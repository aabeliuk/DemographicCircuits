"""
Test script to verify CSV output format from intervention functions.

This script creates mock data to test the CSV export functionality
without running the full model inference.
"""

import pandas as pd
import numpy as np

# Expected columns for single demographic CSV
expected_single_cols = [
    'question',
    'question_label',
    'user_id',
    'demographic',
    'demographic_value',
    'true_label',
    'baseline_prediction',
    'intervention_prediction',
    'baseline_correct',
    'intervention_correct',
    'changed'
]

# Expected columns for intersectional CSV (demographics will vary)
expected_intersectional_base_cols = [
    'question',
    'question_label',
    'user_id',
    'true_label',
    'baseline_prediction',
    'intervention_prediction',
    'baseline_correct',
    'intervention_correct',
    'changed'
]

def create_mock_single_demographic_csv():
    """Create mock CSV data for single demographic intervention"""
    data = []
    for i in range(10):
        data.append({
            'user_id': i,
            'demographic_value': 'Liberal' if i % 2 == 0 else 'Conservative',
            'true_label': 'Favor',
            'baseline_prediction': 'Neither',
            'intervention_prediction': 'Favor'
        })

    # Create DataFrame
    df = pd.DataFrame(data)
    df.insert(0, 'question', 'abortion')
    df.insert(1, 'question_label', 'Abortion Policy')
    df.insert(3, 'demographic', 'ideology')
    df['baseline_correct'] = df['true_label'] == df['baseline_prediction']
    df['intervention_correct'] = df['true_label'] == df['intervention_prediction']
    df['changed'] = df['baseline_prediction'] != df['intervention_prediction']

    return df

def create_mock_intersectional_csv():
    """Create mock CSV data for intersectional intervention"""
    data = []
    for i in range(10):
        data.append({
            'user_id': i,
            'true_label': 'Favor',
            'baseline_prediction': 'Neither',
            'intervention_prediction': 'Favor',
            'demographic_age': '18-29' if i % 2 == 0 else '65+',
            'demographic_gender': 'Female' if i % 3 == 0 else 'Male',
            'demographic_ideology': 'Liberal' if i % 2 == 0 else 'Conservative'
        })

    # Create DataFrame
    df = pd.DataFrame(data)
    df.insert(0, 'question', 'abortion')
    df.insert(1, 'question_label', 'Abortion Policy')
    df['baseline_correct'] = df['true_label'] == df['baseline_prediction']
    df['intervention_correct'] = df['true_label'] == df['intervention_prediction']
    df['changed'] = df['baseline_prediction'] != df['intervention_prediction']

    return df

def verify_csv_format():
    """Verify CSV format matches expectations"""
    print("=" * 70)
    print("CSV Format Verification")
    print("=" * 70)

    # Test single demographic
    print("\n1. Single Demographic CSV Format:")
    print("-" * 70)
    single_df = create_mock_single_demographic_csv()
    print(f"   Columns: {list(single_df.columns)}")
    print(f"   Shape: {single_df.shape}")
    print(f"\n   Sample rows:")
    print(single_df.head(3).to_string(index=False))

    # Verify columns
    missing_cols = set(expected_single_cols) - set(single_df.columns)
    extra_cols = set(single_df.columns) - set(expected_single_cols)

    if missing_cols:
        print(f"\n   ❌ Missing columns: {missing_cols}")
    if extra_cols:
        print(f"\n   ❌ Extra columns: {extra_cols}")
    if not missing_cols and not extra_cols:
        print(f"\n   ✓ All expected columns present!")

    # Test intersectional
    print("\n\n2. Intersectional CSV Format:")
    print("-" * 70)
    intersect_df = create_mock_intersectional_csv()
    print(f"   Columns: {list(intersect_df.columns)}")
    print(f"   Shape: {intersect_df.shape}")
    print(f"\n   Sample rows:")
    print(intersect_df.head(3).to_string(index=False))

    # Verify base columns (demographic columns will vary)
    demographic_cols = [col for col in intersect_df.columns if col.startswith('demographic_')]
    other_cols = [col for col in intersect_df.columns if not col.startswith('demographic_')]

    missing_cols = set(expected_intersectional_base_cols) - set(other_cols)
    extra_cols = set(other_cols) - set(expected_intersectional_base_cols)

    if missing_cols:
        print(f"\n   ❌ Missing base columns: {missing_cols}")
    if extra_cols:
        print(f"\n   ❌ Extra base columns: {extra_cols}")
    if not missing_cols and not extra_cols:
        print(f"\n   ✓ All expected base columns present!")

    print(f"\n   Demographic columns: {demographic_cols}")

    # Test CSV statistics
    print("\n\n3. CSV Statistics:")
    print("-" * 70)
    print(f"   Single demographic:")
    print(f"      Baseline accuracy: {single_df['baseline_correct'].mean():.2%}")
    print(f"      Intervention accuracy: {single_df['intervention_correct'].mean():.2%}")
    print(f"      Changed predictions: {single_df['changed'].sum()} / {len(single_df)}")

    print(f"\n   Intersectional:")
    print(f"      Baseline accuracy: {intersect_df['baseline_correct'].mean():.2%}")
    print(f"      Intervention accuracy: {intersect_df['intervention_correct'].mean():.2%}")
    print(f"      Changed predictions: {intersect_df['changed'].sum()} / {len(intersect_df)}")

    print("\n" + "=" * 70)
    print("Verification Complete!")
    print("=" * 70)

if __name__ == "__main__":
    verify_csv_format()
