"""
MLP Circuit Intervention Engine

This module implements intervention strategies for manipulating MLP layer outputs
to causally test and steer model behavior based on identified MLP circuits.

Similar to CircuitInterventionEngine but for MLP layers instead of attention heads.
"""

import torch
import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class MLPInterventionConfig:
    """Configuration for MLP layer intervention"""
    intervention_strength: float = 1.0  # Alpha parameter: scaling of intervention
    top_k_layers: int = 10  # Number of top layers to intervene on
    intervention_direction: str = 'maximize'  # 'maximize', 'minimize', or 'suppress'


class MLPInterventionEngine:
    """
    Performs causal interventions on MLP circuits.

    Intervention: mlp_output += alpha * ridge_coef * std(features)
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        intervention_weights: Dict[int, Tuple[np.ndarray, float, float]],
        device: str = 'cpu'
    ):
        """
        Args:
            model: The language model to intervene on
            intervention_weights: Dict mapping layer_idx -> (ridge_coef, intercept, feature_std)
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
        config: MLPInterventionConfig
    ) -> torch.Tensor:
        """
        Steer model by modulating MLP activations and return LOGITS.

        Args:
            prompt: Input prompt
            tokenizer: Tokenizer
            config: Intervention configuration

        Returns:
            Logits tensor: (1, vocab_size) for the last token
        """
        # Get top-k layers to intervene on
        top_k_layers = list(self.intervention_weights.items())[:config.top_k_layers]

        # Prepare intervention hooks
        self._clear_hooks()

        for layer_idx, (ridge_coef, intercept, feature_std) in top_k_layers:
            hook = self._create_steering_hook(
                layer_idx, ridge_coef, feature_std, config
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
        config: MLPInterventionConfig
    ) -> torch.Tensor:
        """
        Steer model by modulating MLP activations for multiple prompts in a single batched forward pass.

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

        # Get top-k layers to intervene on
        top_k_layers = list(self.intervention_weights.items())[:config.top_k_layers]

        # Prepare intervention hooks (set up once for all prompts)
        self._clear_hooks()

        for layer_idx, (ridge_coef, intercept, feature_std) in top_k_layers:
            hook = self._create_steering_hook(
                layer_idx, ridge_coef, feature_std, config
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
            logits = outputs.logits[:, -1, :]  # (batch_size, vocab_size)

        # Clean up hooks
        self._clear_hooks()

        return logits

    def _create_steering_hook(
        self,
        target_layer: int,
        ridge_coef: np.ndarray,
        feature_std: float,
        config: MLPInterventionConfig
    ):
        """
        Create a hook function for MLP activation steering.

        Intervention formula:
        mlp_output += alpha * (ridge_coef / ||ridge_coef||) * std(features)
        """
        # Normalize ridge coefficient (direction only)
        ridge_coef_normalized = ridge_coef / (np.linalg.norm(ridge_coef) + 1e-8)
        ridge_coef_tensor = torch.from_numpy(ridge_coef_normalized).float().to(self.device)

        # Determine intervention direction
        if config.intervention_direction == 'minimize':
            direction_multiplier = -1.0
        elif config.intervention_direction == 'suppress':
            direction_multiplier = 0.0  # Will be handled differently
        else:  # maximize
            direction_multiplier = 1.0

        def hook_fn(module, input, output):
            """
            Hook function to modify MLP output.

            Output shape: (batch, seq_len, hidden_size)
            """
            if isinstance(output, tuple):
                mlp_output = output[0]
            else:
                mlp_output = output

            if config.intervention_direction == 'suppress':
                # Suppress: zero out MLP output
                modified_output = torch.zeros_like(mlp_output)
            else:
                # Steer: add scaled direction
                intervention = (
                    config.intervention_strength *
                    direction_multiplier *
                    feature_std *
                    ridge_coef_tensor
                )

                # Broadcast intervention across batch and sequence dimensions
                # intervention shape: (hidden_size,)
                # mlp_output shape: (batch, seq_len, hidden_size)
                modified_output = mlp_output + intervention

            # Return in same format as input
            if isinstance(output, tuple):
                return (modified_output,) + output[1:]
            else:
                return modified_output

        # Register hook on target MLP layer
        layer_module = self.model.model.layers[target_layer].mlp
        handle = layer_module.register_forward_hook(hook_fn)

        return handle

    def _clear_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
