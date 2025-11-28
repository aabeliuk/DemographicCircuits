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
    batch_size: int = 1,
    answer_texts: List[str] = None
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
            - 'answer_tokens': Average only over tokens corresponding to answer text
        batch_size: Number of prompts to process in parallel (default: 1)
        answer_texts: List of answer strings (required when aggregation='answer_tokens')
                     Must have same length as prompts

    Returns:
        Tensor of shape:
        - If aggregation='mean', 'last', or 'answer_tokens': (n_prompts, n_layers, n_heads, head_dim)
        - If aggregation='all': (n_prompts, n_layers, n_heads, seq_len, head_dim)

    Example:
        >>> activations = extract_full_activations_baukit(
        ...     model, tokenizer, ["Hello world"], device="cuda"
        ... )
        >>> # Shape: (1, 16_layers, 32_heads, 64_head_dim) for LLaMA-3.2-1B

        >>> # Extract from answer tokens only
        >>> activations = extract_full_activations_baukit(
        ...     model, tokenizer,
        ...     ["A person is asked: Question? They answer: Yes"],
        ...     device="cuda",
        ...     aggregation='answer_tokens',
        ...     answer_texts=["Yes"]
        ... )
    """
    if not BAUKIT_AVAILABLE:
        raise ImportError(
            "baukit is required for activation extraction.\n"
            "Install with: pip install baukit"
        )

    # Validate answer_tokens aggregation
    if aggregation == 'answer_tokens':
        if answer_texts is None:
            raise ValueError("answer_texts must be provided when aggregation='answer_tokens'")
        if len(answer_texts) != len(prompts):
            raise ValueError(f"answer_texts length ({len(answer_texts)}) must match prompts length ({len(prompts)})")

    # Model architecture parameters
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    # Define which layers to hook into
    # For LLaMA: model.layers.{i}.self_attn outputs the attention layer result
    ATTN_OUTPUTS = [f"model.layers.{i}.self_attn" for i in range(num_layers)]

    # Helper function to find answer token positions
    def find_answer_token_positions(prompt: str, answer: str, tokenizer) -> List[int]:
        """Find token positions corresponding to the answer in the prompt."""
        # Tokenize full prompt
        full_tokens = tokenizer.encode(prompt, add_special_tokens=True)

        # Tokenize answer alone to see what tokens it produces
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)

        # Find where answer tokens appear in the full sequence
        # We look for the answer tokens at the end of the prompt
        if len(answer_tokens) == 0:
            # Fallback to last token if answer is empty or becomes empty after tokenization
            return [len(full_tokens) - 1]

        # Search for answer tokens as a subsequence in the full tokens
        # Start from the end since answers typically appear at the end
        for start_pos in range(len(full_tokens) - len(answer_tokens), -1, -1):
            if full_tokens[start_pos:start_pos + len(answer_tokens)] == answer_tokens:
                return list(range(start_pos, start_pos + len(answer_tokens)))

        # Fallback: if exact match not found, use last N tokens where N = len(answer_tokens)
        # This handles cases where tokenization differs slightly in context
        return list(range(max(0, len(full_tokens) - len(answer_tokens)), len(full_tokens)))

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
                    elif aggregation == 'answer_tokens':
                        # Average over answer tokens only: (batch, num_heads, head_dim)
                        batch_answer_activations = []
                        for batch_idx in range(batch):
                            prompt_idx = batch_start + batch_idx
                            answer = answer_texts[prompt_idx]
                            prompt = batch_prompts[batch_idx]

                            # Find answer token positions
                            answer_positions = find_answer_token_positions(prompt, answer, tokenizer)

                            # Extract activations for answer tokens and average
                            # head_out shape: (batch, seq_len, num_heads, head_dim)
                            answer_acts = head_out[batch_idx, answer_positions, :, :]  # (n_answer_tokens, num_heads, head_dim)
                            avg_answer_acts = answer_acts.mean(dim=0)  # (num_heads, head_dim)
                            batch_answer_activations.append(avg_answer_acts)

                        # Stack batch: (batch, num_heads, head_dim)
                        head_out = torch.stack(batch_answer_activations, dim=0)
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
    batch_size: int = 1,
    answer_texts: List[str] = None
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
            - 'answer_tokens': Average only over tokens corresponding to answer text
        batch_size: Number of prompts to process in parallel (default: 1)
        answer_texts: List of answer strings (required when aggregation='answer_tokens')
                     Must have same length as prompts

    Returns:
        Tensor of shape:
        - If aggregation='mean', 'last', or 'answer_tokens': (n_prompts, n_layers, hidden_size)
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

    # Validate answer_tokens aggregation
    if aggregation == 'answer_tokens':
        if answer_texts is None:
            raise ValueError("answer_texts must be provided when aggregation='answer_tokens'")
        if len(answer_texts) != len(prompts):
            raise ValueError(f"answer_texts length ({len(answer_texts)}) must match prompts length ({len(prompts)})")

    # Model architecture parameters
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    # Define which MLP layers to hook into
    # For LLaMA: model.layers.{i}.mlp outputs the feed-forward layer result
    MLP_OUTPUTS = [f"model.layers.{i}.mlp" for i in range(num_layers)]

    # Helper function to find answer token positions (same as in attention extraction)
    def find_answer_token_positions(prompt: str, answer: str, tokenizer) -> List[int]:
        """Find token positions corresponding to the answer in the prompt."""
        # Tokenize full prompt
        full_tokens = tokenizer.encode(prompt, add_special_tokens=True)

        # Tokenize answer alone to see what tokens it produces
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)

        # Find where answer tokens appear in the full sequence
        # We look for the answer tokens at the end of the prompt
        if len(answer_tokens) == 0:
            # Fallback to last token if answer is empty or becomes empty after tokenization
            return [len(full_tokens) - 1]

        # Search for answer tokens as a subsequence in the full tokens
        # Start from the end since answers typically appear at the end
        for start_pos in range(len(full_tokens) - len(answer_tokens), -1, -1):
            if full_tokens[start_pos:start_pos + len(answer_tokens)] == answer_tokens:
                return list(range(start_pos, start_pos + len(answer_tokens)))

        # Fallback: if exact match not found, use last N tokens where N = len(answer_tokens)
        # This handles cases where tokenization differs slightly in context
        return list(range(max(0, len(full_tokens) - len(answer_tokens)), len(full_tokens)))

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
                    elif aggregation == 'answer_tokens':
                        # Average over answer tokens only: (batch, hidden_size)
                        batch_answer_activations = []
                        for batch_idx in range(mlp_out.shape[0]):
                            prompt_idx = batch_start + batch_idx
                            answer = answer_texts[prompt_idx]
                            prompt = batch_prompts[batch_idx]

                            # Find answer token positions
                            answer_positions = find_answer_token_positions(prompt, answer, tokenizer)

                            # Extract activations for answer tokens and average
                            # mlp_out shape: (batch, seq_len, hidden_size)
                            answer_acts = mlp_out[batch_idx, answer_positions, :]  # (n_answer_tokens, hidden_size)
                            avg_answer_acts = answer_acts.mean(dim=0)  # (hidden_size,)
                            batch_answer_activations.append(avg_answer_acts)

                        # Stack batch: (batch, hidden_size)
                        mlp_out = torch.stack(batch_answer_activations, dim=0)
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
