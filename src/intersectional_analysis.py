"""
Intersectional Analysis Utilities for CCA-based Interventions

This module provides functions for user-specific, profile-based interventions
that leverage the full multi-dimensional structure of CCA results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def compute_user_specific_weights(
    user_profile: pd.Series,
    intervention_weights: Dict,
    demographic_columns: List[str]
) -> Dict:
    """
    Compute intervention weights specific to individual user's demographic profile.

    Instead of category-based steering (same weights for all men, all women, etc.),
    this uses the user's actual demographic vector to determine the optimal
    steering direction based on their specific intersectional identity.

    Args:
        user_profile: User's demographic features (one-hot encoded)
        intervention_weights: Dict from get_intervention_weights_multidim() containing:
            'u_vectors': (feature_dim, n_dims) - Neural directions
            'v_vectors': (demo_dim, n_dims) - Demographic patterns
            'canonical_corrs': (n_dims,) - Correlation strengths
        demographic_columns: List of demographic attribute column names

    Returns:
        User-specific intervention weights for each component in format:
        Dict[component_id] -> (intervention_direction, intercept, std)
    """
    # Extract user's demographic vector
    user_demo_vector = user_profile[demographic_columns].values  # Shape: (demo_dim,)

    user_weights = {}

    for component_id, weight_data in intervention_weights.items():
        u_vectors = weight_data['u_vectors']  # (feature_dim, n_dims)
        v_vectors = weight_data['v_vectors']  # (demo_dim, n_dims)
        canonical_corrs = weight_data['canonical_corrs']  # (n_dims,)

        # Project user's demographic profile onto demographic canonical space
        # This tells us how much the user "loads" on each canonical dimension
        user_demo_canonical = v_vectors.T @ user_demo_vector  # (n_dims,)

        # Compute intervention direction weighted by:
        # 1. User's demographic profile (user_demo_canonical)
        # 2. Canonical correlation strength (canonical_corrs)
        # This creates a direction that amplifies the user's specific demographic combination
        intervention_direction = u_vectors @ (canonical_corrs * user_demo_canonical)

        user_weights[component_id] = (intervention_direction, 0.0, 1.0)

    return user_weights


def compute_targeted_demographic_shift(
    source_profile: pd.Series,
    target_demographics: Dict[str, str],
    intervention_weights: Dict,
    demographic_columns: List[str],
    demographic_encodings: Dict
) -> Dict:
    """
    Compute weights to shift from source demographics toward target demographics.

    This enables targeted interventions like:
    - Shift "Old Man" toward "Young Woman" (gender + age shift)
    - Shift "White" toward "Black" (race shift only)
    - Shift any profile toward a specific intersectional target

    Args:
        source_profile: User's current demographic profile (one-hot encoded)
        target_demographics: Desired demographic values to shift toward
            Example: {'gender': 'Woman', 'age': 'Young'}
        intervention_weights: Dict from get_intervention_weights_multidim()
        demographic_columns: List of demographic attribute column names
        demographic_encodings: Mapping of demographic values to one-hot column names
            Example: {'gender': {'Man': 'gender_Man', 'Woman': 'gender_Woman'}, ...}

    Returns:
        Intervention weights that steer toward target demographic combination
        Dict[component_id] -> (intervention_direction, intercept, std)
    """
    # Create target demographic vector
    target_vector = create_demographic_vector(
        target_demographics,
        demographic_encodings,
        demographic_columns
    )
    source_vector = source_profile[demographic_columns].values

    # Compute demographic shift direction in demographic space
    demo_shift = target_vector - source_vector

    user_weights = {}

    for component_id, weight_data in intervention_weights.items():
        u_vectors = weight_data['u_vectors']  # Neural directions
        v_vectors = weight_data['v_vectors']  # Demographic patterns
        canonical_corrs = weight_data['canonical_corrs']

        # Project demographic shift onto canonical space
        # This tells us which canonical dimensions encode the desired shift
        shift_canonical = v_vectors.T @ demo_shift  # (n_dims,)

        # Compute neural intervention direction
        # Weight by canonical correlations (stronger dimensions have more influence)
        intervention_direction = u_vectors @ (canonical_corrs * shift_canonical)

        user_weights[component_id] = (intervention_direction, 0.0, 1.0)

    return user_weights


def create_demographic_vector(
    target_demographics: Dict[str, str],
    demographic_encodings: Dict[str, Dict[str, str]],
    demographic_columns: List[str]
) -> np.ndarray:
    """
    Create a one-hot encoded demographic vector from target demographic values.

    Args:
        target_demographics: Desired demographic values
            Example: {'gender': 'Woman', 'age': 'Young'}
        demographic_encodings: Mapping of demographic values to one-hot column names
            Example: {'gender': {'Man': 'gender_Man', 'Woman': 'gender_Woman'}, ...}
        demographic_columns: List of all demographic column names (for vector shape)

    Returns:
        One-hot encoded vector (demo_dim,) with 1s at target positions
    """
    # Create zero vector
    target_vector = np.zeros(len(demographic_columns))

    # Set 1s for target demographics
    for demo_attr, demo_value in target_demographics.items():
        if demo_attr in demographic_encodings:
            if demo_value in demographic_encodings[demo_attr]:
                col_name = demographic_encodings[demo_attr][demo_value]
                if col_name in demographic_columns:
                    col_idx = demographic_columns.index(col_name)
                    target_vector[col_idx] = 1.0

    return target_vector


def select_relevant_dimensions(
    user_profile: pd.Series,
    v_vectors: np.ndarray,
    canonical_corrs: np.ndarray,
    demographic_columns: List[str],
    threshold: float = 0.3
) -> np.ndarray:
    """
    Select which canonical dimensions are relevant for this user's demographics.

    Different dimensions encode different demographic patterns:
    - Dim 0 might encode gender-primary variance
    - Dim 1 might encode race-primary variance
    - Dim 2 might encode age-primary variance

    For a user who is [Woman, White, Young], we weight dimensions based on
    how much each dimension captures these specific attributes.

    Args:
        user_profile: User's demographic profile
        v_vectors: Demographic patterns (demo_dim, n_canonical_dims)
        canonical_corrs: Canonical correlations (n_canonical_dims,)
        demographic_columns: List of demographic column names
        threshold: Minimum weight to keep dimension (default: 0.3)

    Returns:
        Dimension weights (n_canonical_dims,) indicating relevance for this user
    """
    user_demo_vector = user_profile[demographic_columns].values

    # Project user onto each canonical dimension
    user_projections = v_vectors.T @ user_demo_vector  # (n_canonical_dims,)

    # Weight by canonical correlations and user relevance
    dimension_weights = canonical_corrs * np.abs(user_projections)

    # Normalize
    dimension_weights = dimension_weights / (dimension_weights.sum() + 1e-8)

    # Zero out dimensions below threshold (sparsity)
    dimension_weights[dimension_weights < threshold] = 0

    # Re-normalize after thresholding
    dimension_weights = dimension_weights / (dimension_weights.sum() + 1e-8)

    return dimension_weights


def batch_users_by_profile_similarity(
    users_df: pd.DataFrame,
    demographic_columns: List[str],
    n_prototypes: int = 10
) -> Dict[int, List[int]]:
    """
    Batch users into groups with similar demographic profiles.

    This is a performance optimization for profile-based interventions.
    Instead of computing unique weights for each of 500 users, we can:
    1. Cluster users into ~10 demographic prototypes
    2. Compute weights once per prototype
    3. Reuse hooks for all users in that prototype group

    Args:
        users_df: DataFrame with user profiles
        demographic_columns: List of demographic column names
        n_prototypes: Number of prototype groups to create

    Returns:
        Dict mapping prototype_id -> list of user indices
    """
    from sklearn.cluster import KMeans

    # Extract demographic vectors
    demo_vectors = users_df[demographic_columns].values  # (n_users, demo_dim)

    # Cluster users by demographic similarity
    kmeans = KMeans(n_clusters=n_prototypes, random_state=42)
    prototype_labels = kmeans.fit_predict(demo_vectors)

    # Group user indices by prototype
    prototype_groups = {i: [] for i in range(n_prototypes)}
    for user_idx, prototype_id in enumerate(prototype_labels):
        prototype_groups[prototype_id].append(user_idx)

    return prototype_groups


def get_demographic_encodings(
    df: pd.DataFrame,
    demographic_attrs: List[str]
) -> Dict[str, Dict[str, str]]:
    """
    Extract demographic encodings from a DataFrame.

    This creates a mapping from demographic attribute values to
    their one-hot encoded column names.

    Args:
        df: DataFrame with one-hot encoded demographics
        demographic_attrs: List of demographic attributes (e.g., ['gender', 'age', 'race'])

    Returns:
        Dict mapping demographic_attr -> {value: column_name}
        Example: {
            'gender': {'Man': 'gender_Man', 'Woman': 'gender_Woman'},
            'age': {'Young': 'age_Young', 'Middle-aged': 'age_Middle-aged', ...}
        }
    """
    encodings = {}

    for attr in demographic_attrs:
        # Find all columns that start with this attribute
        attr_cols = [col for col in df.columns if col.startswith(f"{attr}_")]

        if len(attr_cols) > 0:
            # Extract value from column name (e.g., 'gender_Woman' -> 'Woman')
            encodings[attr] = {
                col.replace(f"{attr}_", ""): col
                for col in attr_cols
            }

    return encodings
