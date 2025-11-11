"""
Linear Probing Classifier for Political Circuit Analysis

This module implements the linear probing methodology from RepresentationPoliticalLLM-main,
adapted for ANES survey data. It trains Ridge regression classifiers per attention head
to predict political attitudes from model activations.

Key features:
- Ridge regression per attention head
- k-fold cross-validation
- Spearman correlation evaluation
- Top-k head selection for intervention
"""

import numpy as np
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal
import torch


@dataclass
class ProbingResult:
    """Results from probing a single attention head"""
    layer: int
    head: int
    train_score: float
    val_score: float
    spearman_r: float
    spearman_p: float
    ridge_coef: np.ndarray
    ridge_intercept: float
    feature_std: float  # Standard deviation of training features


@dataclass
class CircuitProbingResults:
    """Complete results from probing all attention heads"""
    head_results: List[ProbingResult]
    top_k_heads: List[Tuple[int, int]]  # (layer, head) pairs
    top_k_scores: List[float]  # Spearman correlations
    num_heads_probed: int
    task_type: Literal['regression', 'classification']


class AttentionHeadProber:
    """
    Trains linear probes on attention head outputs to predict political attitudes.

    Based on RepresentationPoliticalLLM-main methodology:
    1. Extract attention head outputs for each prompt
    2. Train Ridge regression/classifier per head independently
    3. Evaluate using cross-validation
    4. Rank heads by predictive performance (Spearman r)
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        alpha: float = 1.0,
        n_folds: int = 2,
        task_type: Literal['regression', 'classification'] = 'regression',
        random_state: int = 42
    ):
        """
        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads per layer
            head_dim: Dimension of each attention head
            alpha: Ridge regression regularization parameter
            n_folds: Number of cross-validation folds
            task_type: 'regression' for continuous targets, 'classification' for binary
            random_state: Random seed for reproducibility
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.alpha = alpha
        self.n_folds = n_folds
        self.task_type = task_type
        self.random_state = random_state

    def probe_single_head(
        self,
        head_features: np.ndarray,
        targets: np.ndarray,
        layer: int,
        head: int
    ) -> ProbingResult:
        """
        Train and evaluate a linear probe for a single attention head.

        Args:
            head_features: Shape (n_samples, head_dim)
            targets: Shape (n_samples,) - continuous or binary
            layer: Layer index
            head: Head index

        Returns:
            ProbingResult with cross-validation metrics
        """
        n_samples = head_features.shape[0]

        # Choose model based on task type
        if self.task_type == 'regression':
            model_class = Ridge
        else:
            model_class = RidgeClassifier

        # Cross-validation
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        train_scores = []
        val_scores = []
        val_predictions = np.zeros(n_samples)
        val_indices = []

        for train_idx, val_idx in kf.split(head_features):
            X_train, X_val = head_features[train_idx], head_features[val_idx]
            y_train, y_val = targets[train_idx], targets[val_idx]

            # Train ridge model
            model = model_class(alpha=self.alpha, random_state=self.random_state)
            model.fit(X_train, y_train)

            # Evaluate
            train_scores.append(model.score(X_train, y_train))
            val_scores.append(model.score(X_val, y_val))

            # Store predictions for Spearman correlation
            if self.task_type == 'regression':
                val_predictions[val_idx] = model.predict(X_val)
            else:
                # For classification (binary or multi-class), use predicted class labels
                # This works for both binary (returns 0/1) and multi-class (returns 0/1/2/...)
                val_predictions[val_idx] = model.predict(X_val)

            val_indices.extend(val_idx)

        # Calculate Spearman correlation on all validation predictions
        spearman_r, spearman_p = spearmanr(targets[val_indices], val_predictions[val_indices])

        # Train final model on all data for coefficient extraction
        final_model = model_class(alpha=self.alpha, random_state=self.random_state)
        final_model.fit(head_features, targets)

        # Extract coefficients
        if hasattr(final_model, 'coef_'):
            ridge_coef = final_model.coef_
            if ridge_coef.ndim == 2:  # For multi-class, take first class
                ridge_coef = ridge_coef[0]
        else:
            ridge_coef = np.zeros(self.head_dim)

        ridge_intercept = final_model.intercept_ if hasattr(final_model, 'intercept_') else 0.0
        if isinstance(ridge_intercept, np.ndarray):
            ridge_intercept = ridge_intercept[0]

        # Calculate feature standard deviation for intervention scaling
        feature_std = np.std(head_features)

        return ProbingResult(
            layer=layer,
            head=head,
            train_score=np.mean(train_scores),
            val_score=np.mean(val_scores),
            spearman_r=spearman_r,
            spearman_p=spearman_p,
            ridge_coef=ridge_coef,
            ridge_intercept=ridge_intercept,
            feature_std=feature_std
        )

    def probe_all_heads(
        self,
        activations: torch.Tensor,
        targets: np.ndarray,
        aggregation: Literal['mean', 'last_token'] = 'mean'
    ) -> CircuitProbingResults:
        """
        Probe all attention heads across all layers.

        Args:
            activations: Shape (n_samples, n_layers, n_heads, seq_len, head_dim)
                        or (n_samples, n_layers, n_heads, head_dim) if already aggregated
            targets: Shape (n_samples,) - political attitude labels/scores
            aggregation: How to aggregate across sequence length

        Returns:
            CircuitProbingResults with all head results and top-k ranking
        """
        # Convert to numpy
        if isinstance(activations, torch.Tensor):
            activations = activations.cpu().numpy()

        # Aggregate over sequence dimension if needed
        if activations.ndim == 5:
            if aggregation == 'mean':
                activations = np.mean(activations, axis=3)  # Shape: (n_samples, n_layers, n_heads, head_dim)
            elif aggregation == 'last_token':
                activations = activations[:, :, :, -1, :]  # Take last token

        n_samples, n_layers, n_heads, head_dim = activations.shape
        assert n_layers == self.num_layers
        assert n_heads == self.num_heads
        assert head_dim == self.head_dim

        # Probe each head
        head_results = []
        for layer in range(n_layers):
            for head in range(n_heads):
                # Extract features for this head
                head_features = activations[:, layer, head, :]  # Shape: (n_samples, head_dim)

                # Probe this head
                result = self.probe_single_head(head_features, targets, layer, head)
                head_results.append(result)

        # Rank heads by Spearman correlation (absolute value)
        head_results.sort(key=lambda x: abs(x.spearman_r), reverse=True)

        # Extract top-k heads
        top_k_heads = [(r.layer, r.head) for r in head_results]
        top_k_scores = [r.spearman_r for r in head_results]

        return CircuitProbingResults(
            head_results=head_results,
            top_k_heads=top_k_heads,
            top_k_scores=top_k_scores,
            num_heads_probed=len(head_results),
            task_type=self.task_type
        )

    def get_intervention_weights(
        self,
        results: CircuitProbingResults,
        top_k: int = 20
    ) -> Dict[Tuple[int, int], Tuple[np.ndarray, float, float]]:
        """
        Extract intervention weights for top-k heads.

        Args:
            results: CircuitProbingResults from probe_all_heads
            top_k: Number of top heads to return

        Returns:
            Dictionary mapping (layer, head) -> (ridge_coef, intercept, feature_std)
        """
        intervention_weights = {}

        for result in results.head_results[:top_k]:
            key = (result.layer, result.head)
            intervention_weights[key] = (
                result.ridge_coef,
                result.ridge_intercept,
                result.feature_std
            )

        return intervention_weights

    def print_top_heads(self, results: CircuitProbingResults, top_k: int = 20):
        """Print summary of top-k heads"""
        print(f"\nTop {top_k} Attention Heads (by |Spearman r|):")
        print(f"{'Rank':<6} {'Layer':<8} {'Head':<8} {'Spearman r':<12} {'p-value':<12} {'Val Score':<12}")
        print("-" * 70)

        for rank, result in enumerate(results.head_results[:top_k], 1):
            print(f"{rank:<6} {result.layer:<8} {result.head:<8} "
                  f"{result.spearman_r:<12.4f} {result.spearman_p:<12.6f} "
                  f"{result.val_score:<12.4f}")


class MatchedPairProber(AttentionHeadProber):
    """
    Extension of AttentionHeadProber for matched-pair experiments.

    Instead of predicting absolute values, predicts differences between matched pairs.
    This isolates the causal effect of the demographic variable (e.g., gender).
    """

    def probe_matched_pairs(
        self,
        male_activations: torch.Tensor,
        female_activations: torch.Tensor,
        male_targets: np.ndarray,
        female_targets: np.ndarray,
        aggregation: Literal['mean', 'last_token'] = 'mean'
    ) -> CircuitProbingResults:
        """
        Probe attention heads using matched-pair differences.

        Args:
            male_activations: Shape (n_pairs, n_layers, n_heads, seq_len, head_dim)
            female_activations: Shape (n_pairs, n_layers, n_heads, seq_len, head_dim)
            male_targets: Shape (n_pairs,)
            female_targets: Shape (n_pairs,)
            aggregation: How to aggregate across sequence length

        Returns:
            CircuitProbingResults with matched-pair analysis
        """
        # Convert to numpy
        if isinstance(male_activations, torch.Tensor):
            male_activations = male_activations.cpu().numpy()
        if isinstance(female_activations, torch.Tensor):
            female_activations = female_activations.cpu().numpy()

        # Aggregate over sequence dimension if needed
        if male_activations.ndim == 5:
            if aggregation == 'mean':
                male_activations = np.mean(male_activations, axis=3)
                female_activations = np.mean(female_activations, axis=3)
            elif aggregation == 'last_token':
                male_activations = male_activations[:, :, :, -1, :]
                female_activations = female_activations[:, :, :, -1, :]

        # Compute differences (male - female)
        activation_diffs = male_activations - female_activations  # Shape: (n_pairs, n_layers, n_heads, head_dim)
        target_diffs = male_targets - female_targets  # Shape: (n_pairs,)

        # Use standard probing on differences
        return self.probe_all_heads(
            torch.from_numpy(activation_diffs),
            target_diffs,
            aggregation='mean'  # Already aggregated
        )


def compare_direct_vs_probing(
    direct_circuit: Dict[Tuple[int, int], float],
    probing_results: CircuitProbingResults,
    top_k: int = 20
) -> Dict:
    """
    Compare direct circuit extraction vs. probing approach.

    Args:
        direct_circuit: Dictionary mapping (layer, head) -> importance score
        probing_results: Results from linear probing
        top_k: Number of top heads to compare

    Returns:
        Comparison metrics
    """
    # Get top heads from each method
    direct_top = sorted(direct_circuit.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    direct_heads = set([head for head, score in direct_top])

    probing_heads = set(probing_results.top_k_heads[:top_k])

    # Calculate overlap
    overlap = direct_heads & probing_heads
    overlap_ratio = len(overlap) / top_k

    # Rank correlation (if same heads appear in both)
    common_heads = direct_heads & probing_heads
    if len(common_heads) > 0:
        direct_ranks = {head: rank for rank, (head, _) in enumerate(direct_top)}
        probing_ranks = {head: rank for rank, head in enumerate(probing_results.top_k_heads)}

        common_direct_ranks = [direct_ranks[h] for h in common_heads]
        common_probing_ranks = [probing_ranks[h] for h in common_heads]

        rank_corr, rank_p = spearmanr(common_direct_ranks, common_probing_ranks)
    else:
        rank_corr, rank_p = 0.0, 1.0

    return {
        'overlap_count': len(overlap),
        'overlap_ratio': overlap_ratio,
        'direct_unique': len(direct_heads - probing_heads),
        'probing_unique': len(probing_heads - direct_heads),
        'rank_correlation': rank_corr,
        'rank_p_value': rank_p,
        'common_heads': overlap
    }
