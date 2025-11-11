"""
Activation Extraction Utilities

This module provides convenience wrappers around the baukit library for extracting
internal activations from transformer models. It handles tokenization, batching,
and reshaping to provide clean tensor outputs for circuit analysis.

Baukit (https://github.com/davidbau/baukit) does the heavy lifting of hooking into
model internals. This module adds:
- Tokenization and batch processing
- Token aggregation (mean/last/all)
- Reshaping for attention heads (hidden_size -> num_heads × head_dim)
- Standardized output format for the experiment pipeline

Note: This is a convenience wrapper. You could use baukit's TraceDict directly,
but this provides a cleaner API for our specific use case.
"""

import torch
from typing import List
from tqdm import tqdm

# Try to import baukit
try:
    from baukit import TraceDict
    BAUKIT_AVAILABLE = True
except ImportError:
    BAUKIT_AVAILABLE = False
    print("⚠️  baukit not installed. Install with: pip install baukit")


def extract_full_activations_baukit(
    model,
    tokenizer,
    prompts: List[str],
    device: str,
    aggregation: str = 'mean',
    batch_size: int = 1
) -> torch.Tensor:
    """
    Extract attention head activations using baukit.

    Uses baukit's TraceDict to hook into self-attention layers and extract their
    outputs. The raw hidden_size outputs are reshaped into per-head activations
    for linear probing.

    Args:
        model: Transformer model (e.g., LLaMA)
        tokenizer: HuggingFace tokenizer
        prompts: List of text prompts to process
        device: Device ('cuda', 'mps', 'cpu')
        aggregation: How to aggregate across tokens
            - 'mean': Average activations across all tokens (recommended)
            - 'last': Use only the last token's activations
            - 'all': Keep all tokens (for sequence-level tasks)
        batch_size: Number of prompts to process in parallel (default: 1)

    Returns:
        Tensor of shape:
        - If aggregation='mean' or 'last': (n_prompts, n_layers, n_heads, head_dim)
        - If aggregation='all': (n_prompts, n_layers, n_heads, seq_len, head_dim)

    Example:
        >>> activations = extract_full_activations_baukit(
        ...     model, tokenizer, ["Hello world"], device="cuda"
        ... )
        >>> # Shape: (1, 16_layers, 32_heads, 64_head_dim) for LLaMA-3.2-1B
    """
    if not BAUKIT_AVAILABLE:
        raise ImportError(
            "baukit is required for activation extraction.\n"
            "Install with: pip install baukit"
        )

    # Model architecture parameters
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    # Define which layers to hook into
    # For LLaMA: model.layers.{i}.self_attn outputs the attention layer result
    ATTN_OUTPUTS = [f"model.layers.{i}.self_attn" for i in range(num_layers)]

    all_activations = []
    model.eval()

    # Process prompts in batches
    for batch_start in tqdm(range(0, len(prompts), batch_size), desc="Extracting attention activations"):
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
            # Use baukit's TraceDict to capture layer outputs
            with TraceDict(model, ATTN_OUTPUTS) as ret:
                # Forward pass
                _ = model(**inputs)

                # Extract activations for each layer
                batch_activations = []
                for layer_idx, attn_name in enumerate(ATTN_OUTPUTS):
                    # Get attention layer output: (batch, seq_len, hidden_size)
                    attn_out = ret[attn_name].output

                    # Some models return tuples (hidden_states, attention_weights)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]

                    # Reshape from (batch, seq_len, hidden_size)
                    # to (batch, seq_len, num_heads, head_dim)
                    # This separates out each attention head's contribution
                    batch, seq_len, hidden_size = attn_out.shape
                    head_out = attn_out.view(batch, seq_len, num_heads, head_dim)

                    # Aggregate across tokens
                    if aggregation == 'mean':
                        # Average over all tokens: (batch, num_heads, head_dim)
                        head_out = head_out.mean(dim=1)
                    elif aggregation == 'last':
                        # Use only last token: (batch, num_heads, head_dim)
                        head_out = head_out[:, -1, :, :]
                    # else: keep all tokens (aggregation == 'all')

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
    Extract MLP (feed-forward) layer activations using baukit.

    Uses baukit's TraceDict to hook into MLP layers and extract their outputs.
    MLPs are the feed-forward networks between attention layers.

    Args:
        model: Transformer model (e.g., LLaMA)
        tokenizer: HuggingFace tokenizer
        prompts: List of text prompts to process
        device: Device ('cuda', 'mps', 'cpu')
        aggregation: How to aggregate across tokens
            - 'mean': Average activations across all tokens (recommended)
            - 'last': Use only the last token's activations
            - 'all': Keep all tokens (for sequence-level tasks)
        batch_size: Number of prompts to process in parallel (default: 1)

    Returns:
        Tensor of shape:
        - If aggregation='mean' or 'last': (n_prompts, n_layers, hidden_size)
        - If aggregation='all': (n_prompts, n_layers, seq_len, hidden_size)

    Example:
        >>> activations = extract_mlp_activations_baukit(
        ...     model, tokenizer, ["Hello world"], device="cuda"
        ... )
        >>> # Shape: (1, 16_layers, 2048_hidden) for LLaMA-3.2-1B
    """
    if not BAUKIT_AVAILABLE:
        raise ImportError(
            "baukit is required for activation extraction.\n"
            "Install with: pip install baukit"
        )

    # Model architecture parameters
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    # Define which MLP layers to hook into
    # For LLaMA: model.layers.{i}.mlp outputs the feed-forward layer result
    MLP_OUTPUTS = [f"model.layers.{i}.mlp" for i in range(num_layers)]

    all_activations = []
    model.eval()

    # Process prompts in batches
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
            # Use baukit's TraceDict to capture layer outputs
            with TraceDict(model, MLP_OUTPUTS) as ret:
                # Forward pass
                _ = model(**inputs)

                # Extract MLP outputs for each layer
                batch_activations = []
                for layer_idx, mlp_name in enumerate(MLP_OUTPUTS):
                    # Get MLP output: (batch, seq_len, hidden_size)
                    mlp_out = ret[mlp_name].output

                    # Some models may return tuples
                    if isinstance(mlp_out, tuple):
                        mlp_out = mlp_out[0]

                    # Aggregate across tokens
                    if aggregation == 'mean':
                        # Average over all tokens: (batch, hidden_size)
                        mlp_out = mlp_out.mean(dim=1)
                    elif aggregation == 'last':
                        # Use only last token: (batch, hidden_size)
                        mlp_out = mlp_out[:, -1, :]
                    # else: keep all tokens (aggregation == 'all')

                    batch_activations.append(mlp_out.cpu())

                # Stack layers: (batch, n_layers, [seq_len], hidden_size)
                batch_tensor = torch.stack(batch_activations, dim=1)
                all_activations.append(batch_tensor)

    # Concatenate all batches: (n_prompts, n_layers, [seq_len], hidden_size)
    return torch.cat(all_activations, dim=0)


# Example usage
if __name__ == "__main__":
    print(f"Baukit available: {BAUKIT_AVAILABLE}")
    print()
    print("This module provides convenience wrappers for baukit activation extraction.")
    print()
    print("Functions:")
    print("  - extract_full_activations_baukit(): Extract attention head activations")
    print("  - extract_mlp_activations_baukit(): Extract MLP layer activations")
    print()
    print("Both functions handle:")
    print("  ✓ Tokenization and batching")
    print("  ✓ Token aggregation (mean/last/all)")
    print("  ✓ Reshaping for per-head analysis")
    print("  ✓ Progress bars via tqdm")
    print()
    print("See function docstrings for detailed usage.")
