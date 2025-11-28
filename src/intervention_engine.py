"""
Circuit Intervention Engine

This module implements intervention strategies for manipulating attention head outputs
to causally test and steer model behavior based on identified circuits.

Based on RepresentationPoliticalLLM-main intervention approach:
- Intervene on top-k attention heads
- Modulate activations using learned ridge coefficients
- Test causal effects on generation
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class InterventionConfig:
    """Configuration for circuit intervention"""
    intervention_strength: float = 1.0  # Alpha parameter: scaling of intervention
    top_k_heads: int = 20  # Number of top heads to intervene on
    intervention_direction: str = 'maximize'  # 'maximize', 'minimize', or 'suppress'


@dataclass
class InterventionResult:
    """Results from a single intervention"""
    original_output: str
    intervened_output: str
    intervention_config: InterventionConfig
    affected_heads: List[Tuple[int, int]]  # (layer, head) pairs
    intervention_magnitude: float


class CircuitInterventionEngine:
    """
    Performs causal interventions on attention circuits.

    Intervention types:
    1. Activation steering: head_output += alpha * ridge_coef * std(features)
    2. Head suppression: head_output = 0
    3. Head amplification: head_output *= alpha
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        intervention_weights: Dict[Tuple[int, int], Tuple[np.ndarray, float, float]],
        device: str = 'cpu'
    ):
        """
        Args:
            model: The language model to intervene on
            intervention_weights: Dict mapping (layer, head) -> (ridge_coef, intercept, feature_std)
            device: Device for computation
        """
        self.model = model
        self.intervention_weights = intervention_weights
        self.device = device
        self.hooks = []

    def intervene_activation_steering_logits(
        self,
        prompt: str,
        tokenizer: AutoTokenizer,
        config: InterventionConfig
    ) -> torch.Tensor:
        """
        Steer model and return LOGITS (not generated text).

        This is for evaluation where we need to compare logits directly,
        not generate new tokens.

        Args:
            prompt: Input prompt
            tokenizer: Tokenizer
            config: Intervention configuration

        Returns:
            Logits tensor: (1, vocab_size) for the last token
        """
        # Get top-k heads to intervene on
        top_k_heads = list(self.intervention_weights.items())[:config.top_k_heads]

        # Prepare intervention hooks
        self._clear_hooks()

        for (layer, head), (ridge_coef, intercept, feature_std) in top_k_heads:
            hook = self._create_steering_hook(
                layer, head, ridge_coef, feature_std, config
            )
            self.hooks.append(hook)

        # Forward pass with intervention (NO generation, just get logits)
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]  # Last token logits

        # Clean up hooks
        self._clear_hooks()

        return logits

    def intervene_activation_steering_logits_batched(
        self,
        prompts: List[str],
        tokenizer: AutoTokenizer,
        config: InterventionConfig
    ) -> torch.Tensor:
        """
        Steer model and return LOGITS for multiple prompts in a single batched forward pass.

        This is significantly faster than calling intervene_activation_steering_logits()
        multiple times, as hooks are set up once and all prompts are processed together.

        Args:
            prompts: List of input prompts
            tokenizer: Tokenizer
            config: Intervention configuration

        Returns:
            Logits tensor: (batch_size, vocab_size) for the last token of each prompt
        """
        if len(prompts) == 0:
            return torch.empty(0, self.model.config.vocab_size).to(self.device)

        # Get top-k heads to intervene on
        top_k_heads = list(self.intervention_weights.items())[:config.top_k_heads]

        # Prepare intervention hooks (set up once for all prompts)
        self._clear_hooks()

        for (layer, head), (ridge_coef, intercept, feature_std) in top_k_heads:
            hook = self._create_steering_hook(
                layer, head, ridge_coef, feature_std, config
            )
            self.hooks.append(hook)

        # Tokenize and batch all prompts with padding
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Single batched forward pass with intervention
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Extract logits for the last non-padded token of each sequence
            # For left-padded sequences, last token is always at position -1
            logits = outputs.logits[:, -1, :]  # (batch_size, vocab_size)

        # Clean up hooks
        self._clear_hooks()

        return logits

    def intervene_activation_steering(
        self,
        prompt: str,
        tokenizer: AutoTokenizer,
        config: InterventionConfig,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> InterventionResult:
        """
        Steer model outputs by modulating top-k head activations.

        Based on RepresentationPoliticalLLM method:
        head_output += alpha * ridge_coef * std(features)

        Args:
            prompt: Input prompt
            tokenizer: Tokenizer
            config: Intervention configuration
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            InterventionResult with original and intervened outputs
        """
        # First, generate without intervention (baseline)
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        original_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Get top-k heads to intervene on
        top_k_heads = list(self.intervention_weights.items())[:config.top_k_heads]

        # Prepare intervention hooks
        self._clear_hooks()

        for (layer, head), (ridge_coef, intercept, feature_std) in top_k_heads:
            hook = self._create_steering_hook(
                layer, head, ridge_coef, feature_std, config
            )
            self.hooks.append(hook)

        # Generate with intervention
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        intervened_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean up hooks
        self._clear_hooks()

        # Calculate intervention magnitude
        intervention_magnitude = config.intervention_strength * np.mean([std for _, _, std in [w for _, w in top_k_heads]])

        return InterventionResult(
            original_output=original_output,
            intervened_output=intervened_output,
            intervention_config=config,
            affected_heads=[(layer, head) for (layer, head), _ in top_k_heads],
            intervention_magnitude=intervention_magnitude
        )

    def _create_steering_hook(
        self,
        target_layer: int,
        target_head: int,
        ridge_coef: np.ndarray,
        feature_std: float,
        config: InterventionConfig
    ) -> Callable:
        """
        Create a hook that steers a specific attention head.

        Intervention formula (from RepresentationPoliticalLLM):
        head_output += alpha * (ridge_coef / ||ridge_coef||) * std(features)
        """
        # Normalize ridge coefficient (direction only)
        ridge_coef_normalized = ridge_coef / (np.linalg.norm(ridge_coef) + 1e-8)
        # Match model dtype (float16 or float32)
        model_dtype = next(self.model.parameters()).dtype
        ridge_coef_tensor = torch.from_numpy(ridge_coef_normalized).to(dtype=model_dtype, device=self.device)

        # Determine intervention direction
        if config.intervention_direction == 'minimize':
            direction_multiplier = -1.0
        elif config.intervention_direction == 'suppress':
            direction_multiplier = 0.0  # Will be handled differently
        else:  # maximize
            direction_multiplier = 1.0

        def hook_fn(module, input, output):
            """
            Hook function to modify attention head output.

            Output shape: (batch, seq_len, num_heads, head_dim)
            """
            if isinstance(output, tuple):
                attn_output = output[0]
            else:
                attn_output = output

            batch_size, seq_len, hidden_size = attn_output.shape
            num_heads = self.model.config.num_attention_heads
            head_dim = hidden_size // num_heads

            # Reshape to (batch, seq_len, num_heads, head_dim)
            head_out = attn_output.view(batch_size, seq_len, num_heads, head_dim)

            # Intervene on target head
            if config.intervention_direction == 'suppress':
                # Suppress: zero out head
                head_out[:, :, target_head, :] = 0
            else:
                # Steer: add scaled direction
                intervention = (
                    config.intervention_strength *
                    direction_multiplier *
                    feature_std *
                    ridge_coef_tensor
                )
                head_out[:, :, target_head, :] += intervention

            # Reshape back
            modified_output = head_out.view(batch_size, seq_len, hidden_size)

            # Return in same format as input
            if isinstance(output, tuple):
                return (modified_output,) + output[1:]
            else:
                return modified_output

        # Register hook on target layer
        layer_module = self.model.model.layers[target_layer].self_attn
        handle = layer_module.register_forward_hook(hook_fn)

        return handle

    def intervene_suppression(
        self,
        prompt: str,
        tokenizer: AutoTokenizer,
        config: InterventionConfig,
        max_new_tokens: int = 50
    ) -> InterventionResult:
        """
        Suppress (zero out) top-k head activations to test necessity.

        This is a simpler intervention that tests whether specific heads
        are causally necessary for the behavior.
        """
        config_suppression = InterventionConfig(
            intervention_strength=0.0,
            top_k_heads=config.top_k_heads,
            intervention_direction='suppress'
        )

        return self.intervene_activation_steering(
            prompt, tokenizer, config_suppression, max_new_tokens
        )

    def intervene_amplification(
        self,
        prompt: str,
        tokenizer: AutoTokenizer,
        config: InterventionConfig,
        max_new_tokens: int = 50
    ) -> InterventionResult:
        """
        Amplify top-k head activations by multiplying by alpha > 1.
        """
        # First, generate without intervention
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        original_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Get top-k heads
        top_k_heads = list(self.intervention_weights.items())[:config.top_k_heads]

        # Create amplification hooks
        self._clear_hooks()

        for (layer, head), _ in top_k_heads:
            hook = self._create_amplification_hook(layer, head, config.intervention_strength)
            self.hooks.append(hook)

        # Generate with intervention
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        intervened_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean up
        self._clear_hooks()

        return InterventionResult(
            original_output=original_output,
            intervened_output=intervened_output,
            intervention_config=config,
            affected_heads=[(layer, head) for (layer, head), _ in top_k_heads],
            intervention_magnitude=config.intervention_strength
        )

    def _create_amplification_hook(self, target_layer: int, target_head: int, alpha: float) -> Callable:
        """Create a hook that amplifies a specific head's output"""

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                attn_output = output[0]
            else:
                attn_output = output

            batch_size, seq_len, hidden_size = attn_output.shape
            num_heads = self.model.config.num_attention_heads
            head_dim = hidden_size // num_heads

            # Reshape
            head_out = attn_output.view(batch_size, seq_len, num_heads, head_dim)

            # Amplify target head
            head_out[:, :, target_head, :] *= alpha

            # Reshape back
            modified_output = head_out.view(batch_size, seq_len, hidden_size)

            if isinstance(output, tuple):
                return (modified_output,) + output[1:]
            else:
                return modified_output

        layer_module = self.model.model.layers[target_layer].self_attn
        handle = layer_module.register_forward_hook(hook_fn)

        return handle

    def _clear_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __del__(self):
        """Ensure hooks are cleaned up"""
        self._clear_hooks()


# ============================================================================
# INTERVENTION EVALUATION
# ============================================================================

def evaluate_intervention_effects(
    engine: CircuitInterventionEngine,
    tokenizer: AutoTokenizer,
    test_prompts: List[str],
    intervention_configs: List[InterventionConfig],
    max_new_tokens: int = 50
) -> Dict:
    """
    Systematically evaluate different intervention configurations.

    Args:
        engine: CircuitInterventionEngine
        tokenizer: Tokenizer
        test_prompts: List of prompts to test
        intervention_configs: List of intervention configurations to try
        max_new_tokens: Max tokens to generate

    Returns:
        Dictionary with evaluation metrics
    """
    results = {
        'prompts': test_prompts,
        'configs': intervention_configs,
        'outputs': [],
        'changes': []
    }

    for prompt in test_prompts:
        prompt_results = []

        for config in intervention_configs:
            result = engine.intervene_activation_steering(
                prompt, tokenizer, config, max_new_tokens
            )

            # Calculate change magnitude (simple text difference)
            change = len(set(result.original_output.split()) - set(result.intervened_output.split()))

            prompt_results.append({
                'original': result.original_output,
                'intervened': result.intervened_output,
                'change_magnitude': change,
                'config': config
            })

        results['outputs'].append(prompt_results)

    return results


def print_intervention_results(result: InterventionResult):
    """Pretty-print intervention results"""
    print("\n" + "="*70)
    print("INTERVENTION RESULT")
    print("="*70)
    print(f"Affected heads: {len(result.affected_heads)}")
    print(f"Top 5 heads: {result.affected_heads[:5]}")
    print(f"Intervention strength: {result.intervention_config.intervention_strength}")
    print(f"Direction: {result.intervention_config.intervention_direction}")
    print(f"Magnitude: {result.intervention_magnitude:.4f}")
    print("\n" + "-"*70)
    print("ORIGINAL OUTPUT:")
    print("-"*70)
    print(result.original_output)
    print("\n" + "-"*70)
    print("INTERVENED OUTPUT:")
    print("-"*70)
    print(result.intervened_output)
    print("="*70)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("CircuitInterventionEngine Example")
    print("="*70)
    print("\nThis module provides tools for causal intervention on attention circuits.")
    print("\nKey functions:")
    print("  1. intervene_activation_steering() - Steer using ridge coefficients")
    print("  2. intervene_suppression() - Zero out head activations")
    print("  3. intervene_amplification() - Amplify head activations")
    print("\nExample workflow:")
    print("  1. Train probes with probing_classifier.py")
    print("  2. Extract intervention_weights from top-k heads")
    print("  3. Create CircuitInterventionEngine")
    print("  4. Test interventions on prompts")
    print("  5. Evaluate causal effects")
    print("="*70)
