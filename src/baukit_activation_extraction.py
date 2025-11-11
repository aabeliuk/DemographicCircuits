"""
Improved activation extraction using baukit
Extracts head-wise outputs instead of just attention weights
"""

import torch
import numpy as np
from typing import List
from tqdm import tqdm

# Try to import baukit, fall back to manual hooks if not available
try:
    from baukit import TraceDict
    BAUKIT_AVAILABLE = True
except ImportError:
    BAUKIT_AVAILABLE = False
    print("⚠️ baukit not installed. Install with: pip install baukit")


def extract_activations_baukit(model, tokenizer, prompts: List[str], device: str) -> np.ndarray:
    """
    Extract attention head activations using baukit.

    This captures the OUTPUT of each attention head (after value projection),
    which provides richer information than just attention weights.

    Args:
        model: LLaMA model
        tokenizer: Tokenizer
        prompts: List of prompt strings
        device: Device ('cuda', 'mps', 'cpu')

    Returns:
        (num_layers, num_heads) array of activation magnitudes
    """
    if not BAUKIT_AVAILABLE:
        raise ImportError("baukit is required. Install with: pip install baukit")

    # Define layer names for head outputs
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    # Baukit hook names
    HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(num_layers)]

    head_activations_list = []

    model.eval()
    for prompt in tqdm(prompts, desc="Extracting activations", leave=False):
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            with TraceDict(model, HEADS) as ret:
                # Forward pass
                output = model(**inputs)

                # Extract head outputs
                head_outputs = []
                for head_name in HEADS:
                    head_out = ret[head_name].output  # Shape: (batch, seq_len, num_heads, head_dim)

                    # Average over sequence length, then compute magnitude per head
                    # Shape: (num_heads, head_dim) -> (num_heads,)
                    head_magnitude = head_out.squeeze(0).mean(dim=0).norm(dim=-1)
                    head_outputs.append(head_magnitude.cpu().numpy())

                # Stack into (num_layers, num_heads)
                head_activations = np.stack(head_outputs, axis=0)
                head_activations_list.append(head_activations)

    # Average across all prompts
    return np.mean(head_activations_list, axis=0)


def extract_full_activations_baukit(
    model,
    tokenizer,
    prompts: List[str],
    device: str,
    aggregation: str = 'mean',
    batch_size: int = 1
) -> torch.Tensor:
    """
    Extract full attention head activations (with head dimension) using baukit.

    This version returns the full activation tensors for linear probing,
    not just magnitudes.

    Args:
        model: LLaMA model
        tokenizer: Tokenizer
        prompts: List of prompt strings
        device: Device ('cuda', 'mps', 'cpu')
        aggregation: 'mean' (average over tokens), 'last' (last token), or 'all' (keep all tokens)
        batch_size: Number of prompts to process in parallel

    Returns:
        If aggregation='all': (n_prompts, n_layers, n_heads, seq_len, head_dim)
        If aggregation='mean' or 'last': (n_prompts, n_layers, n_heads, head_dim)
    """
    if not BAUKIT_AVAILABLE:
        raise ImportError("baukit is required. Install with: pip install baukit")

    # Define layer names for head outputs
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    # Hook into the self_attn output (standard LLaMA attribute)
    # We'll reshape the output to get per-head activations
    ATTN_OUTPUTS = [f"model.layers.{i}.self_attn" for i in range(num_layers)]

    all_activations = []

    model.eval()

    # Process in batches
    for batch_start in tqdm(range(0, len(prompts), batch_size), desc="Extracting full activations"):
        batch_prompts = prompts[batch_start:batch_start + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            with TraceDict(model, ATTN_OUTPUTS) as ret:
                # Forward pass
                output = model(**inputs)

                # Extract head outputs for this batch
                batch_activations = []
                for layer_idx, attn_name in enumerate(ATTN_OUTPUTS):
                    # Get the attention output: (batch, seq_len, hidden_size)
                    attn_out = ret[attn_name].output

                    # Handle tuple output (some models return (hidden_states, attention_weights))
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]

                    # Reshape to (batch, seq_len, num_heads, head_dim)
                    batch, seq_len, hidden_size = attn_out.shape
                    head_out = attn_out.view(batch, seq_len, num_heads, head_dim)

                    if aggregation == 'mean':
                        # Average over sequence: (batch, num_heads, head_dim)
                        head_out = head_out.mean(dim=1)
                    elif aggregation == 'last':
                        # Take last token: (batch, num_heads, head_dim)
                        head_out = head_out[:, -1, :, :]
                    # else aggregation == 'all': keep all tokens

                    batch_activations.append(head_out.cpu())

                # Stack layers: (batch, n_layers, n_heads, [seq_len], head_dim)
                batch_tensor = torch.stack(batch_activations, dim=1)
                all_activations.append(batch_tensor)

    # Concatenate all batches: (n_prompts, n_layers, n_heads, [seq_len], head_dim)
    return torch.cat(all_activations, dim=0)


def extract_mlp_activations_baukit(
    model,
    tokenizer,
    prompts: List[str],
    device: str,
    aggregation: str = 'mean',
    batch_size: int = 1
) -> torch.Tensor:
    """
    Extract MLP activations using baukit.

    Args:
        model: LLaMA model
        tokenizer: Tokenizer
        prompts: List of prompt strings
        device: Device ('cuda', 'mps', 'cpu')
        aggregation: 'mean' (average over tokens), 'last' (last token), or 'all' (keep all tokens)
        batch_size: Number of prompts to process in parallel

    Returns:
        If aggregation='all': (n_prompts, n_layers, seq_len, hidden_size)
        If aggregation='mean' or 'last': (n_prompts, n_layers, hidden_size)
    """
    if not BAUKIT_AVAILABLE:
        raise ImportError("baukit is required. Install with: pip install baukit")

    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    # Hook into MLP outputs
    MLP_OUTPUTS = [f"model.layers.{i}.mlp" for i in range(num_layers)]

    all_activations = []

    model.eval()

    # Process in batches
    for batch_start in tqdm(range(0, len(prompts), batch_size), desc="Extracting MLP activations"):
        batch_prompts = prompts[batch_start:batch_start + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            with TraceDict(model, MLP_OUTPUTS) as ret:
                # Forward pass
                output = model(**inputs)

                # Extract MLP outputs for this batch
                batch_activations = []
                for layer_idx, mlp_name in enumerate(MLP_OUTPUTS):
                    # Get the MLP output: (batch, seq_len, hidden_size)
                    mlp_out = ret[mlp_name].output

                    # Handle tuple output if necessary
                    if isinstance(mlp_out, tuple):
                        mlp_out = mlp_out[0]

                    if aggregation == 'mean':
                        # Average over sequence: (batch, hidden_size)
                        mlp_out = mlp_out.mean(dim=1)
                    elif aggregation == 'last':
                        # Take last token: (batch, hidden_size)
                        mlp_out = mlp_out[:, -1, :]
                    # else aggregation == 'all': keep all tokens

                    batch_activations.append(mlp_out.cpu())

                # Stack layers: (batch, n_layers, [seq_len], hidden_size)
                batch_tensor = torch.stack(batch_activations, dim=1)
                all_activations.append(batch_tensor)

    # Concatenate all batches: (n_prompts, n_layers, [seq_len], hidden_size)
    return torch.cat(all_activations, dim=0)


def extract_activations_manual_hooks(model, tokenizer, prompts: List[str], device: str) -> np.ndarray:
    """
    Extract attention head activations using manual hooks (no baukit required).

    This version uses PyTorch hooks to capture attention head outputs.
    Similar to baukit but doesn't require external dependency.

    Args:
        model: LLaMA model
        tokenizer: Tokenizer
        prompts: List of prompt strings
        device: Device ('cuda', 'mps', 'cpu')

    Returns:
        (num_layers, num_heads) array of activation magnitudes
    """
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    head_activations_list = []

    model.eval()
    for prompt in prompts:
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Storage for captured activations
        captured_outputs = {}
        hooks = []

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # output[0] is the attention output (hidden states)
                # Shape: (batch, seq_len, hidden_size)
                if isinstance(output, tuple):
                    attn_output = output[0]
                else:
                    attn_output = output

                # Reshape to (batch, seq_len, num_heads, head_dim)
                batch_size, seq_len, hidden_size = attn_output.shape
                head_dim = hidden_size // num_heads

                head_out = attn_output.view(batch_size, seq_len, num_heads, head_dim)

                # Average over sequence, compute magnitude per head
                # Shape: (num_heads,)
                head_magnitudes = head_out.squeeze(0).mean(dim=0).norm(dim=-1)
                captured_outputs[layer_idx] = head_magnitudes.cpu()

            return hook_fn

        # Register hooks on all self-attention layers
        try:
            for layer_idx in range(num_layers):
                layer = model.model.layers[layer_idx].self_attn
                hook = layer.register_forward_hook(make_hook(layer_idx))
                hooks.append(hook)

            # Forward pass
            with torch.no_grad():
                _ = model(**inputs)

            # Collect activations in order
            head_activations = []
            for layer_idx in range(num_layers):
                if layer_idx in captured_outputs:
                    head_activations.append(captured_outputs[layer_idx].numpy())
                else:
                    # Fallback: zeros if layer wasn't captured
                    head_activations.append(np.zeros(num_heads))

            head_activations = np.stack(head_activations, axis=0)
            head_activations_list.append(head_activations)

        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()

    # Average across all prompts
    return np.mean(head_activations_list, axis=0)


def extract_activations_improved(model, tokenizer, prompts: List[str], device: str) -> np.ndarray:
    """
    Improved activation extraction with automatic fallback.

    Tries baukit first, falls back to manual hooks if unavailable.

    Args:
        model: LLaMA model
        tokenizer: Tokenizer
        prompts: List of prompt strings
        device: Device ('cuda', 'mps', 'cpu')

    Returns:
        (num_layers, num_heads) array of activation magnitudes
    """
    if BAUKIT_AVAILABLE:
        print("  Using baukit for activation extraction")
        return extract_activations_baukit(model, tokenizer, prompts, device)
    else:
        print("  Using manual hooks for activation extraction")
        return extract_activations_manual_hooks(model, tokenizer, prompts, device)


# Example usage and comparison
if __name__ == "__main__":
    print(f"Baukit available: {BAUKIT_AVAILABLE}")

    print("\nComparison of methods:")
    print("="*70)
    print("1. ORIGINAL (simple attention averaging):")
    print("   - Extracts attention weights only")
    print("   - Fast but limited information")
    print("   - May give weak signals")
    print()
    print("2. BAUKIT (head-wise output magnitudes):")
    print("   - Extracts actual attention head outputs")
    print("   - Richer information about what heads are computing")
    print("   - Requires 'pip install baukit'")
    print()
    print("3. MANUAL HOOKS (similar to baukit):")
    print("   - Extracts attention head outputs via PyTorch hooks")
    print("   - No external dependencies")
    print("   - Good fallback option")
    print("="*70)

    print("\nTo use in experiment_2:")
    print("1. Install baukit (optional): pip install baukit")
    print("2. Replace extract_activations in Cell 14 with extract_activations_improved")
    print("3. Should get stronger circuit signals than simple attention averaging")
