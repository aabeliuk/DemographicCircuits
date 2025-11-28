"""
ANES 2024 Correlation Analysis Script

Performs comprehensive correlation analysis between demographic variables
and political responses using the ANES 2024 dataset.

Statistical Tests:
- Cramér's V (primary association measure for categorical data)
- Chi-square test for independence
- Multiple testing correction (Bonferroni/FDR)

Outputs:
- CSV: Detailed correlation results
- JSON: Structured results with metadata
- Heatmap: Visual correlation matrix
- Text Report: Summary statistics

Usage:
    python analyze_anes_correlations.py --data path/to/anes.csv --output results/
    python analyze_anes_correlations.py --demographics gender age --questions abortion
    python analyze_anes_correlations.py --correction bonferroni --alpha 0.05
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from datetime import datetime
from collections import Counter

# Import existing ANES data loader
from src.anes_association_learning import ANESAssociationLearner


# Default configuration
DEFAULT_DEMOGRAPHICS = [
    'gender', 'age', 'race', 'education', 'marital_status',
    'income', 'ideology', 'religion', 'urban_rural'
]

DEFAULT_QUESTIONS = [
    'abortion', 'death_penalty', 'military_force', 'defense_spending',
    'govt_jobs', 'govt_help_blacks', 'colleges_opinion', 'dei_opinion',
    'journalist_access', 'transgender_bathrooms', 'birthright_citizenship',
    'immigration_policy'
]

DEFAULT_DATA_PATH = 'data/anes_timeseries_2024_csv_20250808/anes_timeseries_2024_csv_20250808.csv'
DEFAULT_OUTPUT_DIR = 'correlation_results'


class ANESCorrelationAnalyzer:
    """
    Analyzes correlations between demographic variables and political responses
    in the ANES 2024 dataset.

    Uses Cramér's V as the primary association measure for categorical data,
    with chi-square tests for statistical significance.
    """

    def __init__(
        self,
        data_path: str,
        demographics: Optional[List[str]] = None,
        questions: Optional[List[str]] = None,
        alpha: float = 0.05,
        correction_method: str = 'bonferroni'
    ):
        """
        Initialize the correlation analyzer.

        Args:
            data_path: Path to ANES CSV file
            demographics: List of demographic variables to analyze (None = all)
            questions: List of political questions to analyze (None = all)
            alpha: Significance level for statistical tests
            correction_method: Multiple testing correction ('bonferroni', 'fdr_bh', or 'none')
        """
        self.data_path = data_path
        self.demographics = demographics or DEFAULT_DEMOGRAPHICS
        self.questions = questions or DEFAULT_QUESTIONS
        self.alpha = alpha
        self.correction_method = correction_method

        self.data = None
        self.results = {}
        self.analysis_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        print(f"Initializing ANES Correlation Analyzer")
        print(f"  Demographics: {len(self.demographics)}")
        print(f"  Questions: {len(self.questions)}")
        print(f"  Total tests: {len(self.demographics) * len(self.questions)}")
        print(f"  Significance level: {self.alpha}")
        print(f"  Correction method: {self.correction_method}")

    def load_data(self):
        """Load ANES data using existing infrastructure."""
        print(f"\nLoading data from {self.data_path}...")

        try:
            learner = ANESAssociationLearner(self.data_path)
            self.data = learner.anes_data

            # Filter for binary gender if needed
            if 'gender' in self.data.columns:
                original_size = len(self.data)
                self.data = self.data[self.data['gender'].isin(['Man', 'Woman'])].copy()
                filtered_size = len(self.data)
                print(f"  Filtered gender: {original_size} → {filtered_size} samples")

            # Verify all requested variables exist
            missing_demos = [d for d in self.demographics if d not in self.data.columns]
            missing_questions = [q for q in self.questions if q not in self.data.columns]

            if missing_demos:
                print(f"  Warning: Missing demographics: {missing_demos}")
                self.demographics = [d for d in self.demographics if d in self.data.columns]

            if missing_questions:
                print(f"  Warning: Missing questions: {missing_questions}")
                self.questions = [q for q in self.questions if q in self.data.columns]

            print(f"  Loaded {len(self.data)} samples")
            print(f"  Available demographics: {self.demographics}")
            print(f"  Available questions: {self.questions}")

        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading data: {e}")

    def compute_cramers_v(
        self,
        demo: str,
        question: str
    ) -> Optional[Dict]:
        """
        Compute Cramér's V association between a demographic and a question.

        Args:
            demo: Demographic variable name
            question: Political question variable name

        Returns:
            Dictionary with correlation statistics, or None if insufficient data
        """
        # Get valid data (remove NaN)
        valid_data = self.data[[demo, question]].dropna()

        if len(valid_data) < 10:
            return None  # Insufficient data

        # Create contingency table
        try:
            contingency = pd.crosstab(valid_data[demo], valid_data[question])

            # Chi-square test
            chi2, p_value, dof, expected = chi2_contingency(contingency)

            # Cramér's V
            n = contingency.sum().sum()
            min_dim = min(contingency.shape) - 1

            if min_dim == 0:
                return None  # One variable has only one category

            cramers_v = np.sqrt(chi2 / (n * min_dim))

            # Confidence interval (approximate)
            se = np.sqrt(cramers_v * (1 - cramers_v) / n)
            ci_lower = max(0, cramers_v - 1.96 * se)
            ci_upper = min(1, cramers_v + 1.96 * se)

            # Effect size classification
            if cramers_v < 0.1:
                effect_size = 'small'
            elif cramers_v < 0.3:
                effect_size = 'medium'
            else:
                effect_size = 'large'

            # Phi coefficient (for 2x2 tables)
            phi = None
            if contingency.shape == (2, 2):
                phi = np.sqrt(chi2 / n)

            return {
                'demographic': demo,
                'question': question,
                'cramers_v': cramers_v,
                'phi': phi,
                'chi_square': chi2,
                'p_value': p_value,
                'df': dof,
                'effect_size': effect_size,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_samples': n,
                'contingency_shape': contingency.shape,
                'contingency_table': contingency.to_dict()
            }

        except Exception as e:
            print(f"  Warning: Could not compute correlation for {demo} × {question}: {e}")
            return None

    def compute_all_correlations(self):
        """Compute correlations for all demographic-question pairs."""
        print(f"\nComputing correlations...")

        total_tests = len(self.demographics) * len(self.questions)
        completed = 0

        for demo in self.demographics:
            for question in self.questions:
                result = self.compute_cramers_v(demo, question)
                if result is not None:
                    self.results[(demo, question)] = result

                completed += 1
                if completed % 10 == 0 or completed == total_tests:
                    print(f"  Progress: {completed}/{total_tests} ({100*completed/total_tests:.1f}%)")

        print(f"\nCompleted {len(self.results)} valid correlations")

        # Apply multiple testing correction
        if self.correction_method != 'none' and len(self.results) > 0:
            self._apply_multiple_testing_correction()

    def _apply_multiple_testing_correction(self):
        """Apply multiple testing correction to p-values."""
        print(f"\nApplying {self.correction_method} correction...")

        # Extract p-values
        p_values = [result['p_value'] for result in self.results.values()]

        # Apply correction
        if self.correction_method == 'bonferroni':
            _, p_corrected, _, _ = multipletests(p_values, alpha=self.alpha, method='bonferroni')
        elif self.correction_method == 'fdr_bh':
            _, p_corrected, _, _ = multipletests(p_values, alpha=self.alpha, method='fdr_bh')
        else:
            p_corrected = p_values

        # Update results
        for i, (key, result) in enumerate(self.results.items()):
            result['p_value_corrected'] = p_corrected[i]
            result['significant'] = p_corrected[i] < self.alpha

        n_significant = sum(1 for r in self.results.values() if r['significant'])
        print(f"  Significant associations: {n_significant}/{len(self.results)} ({100*n_significant/len(self.results):.1f}%)")

    def export_to_csv(self, output_path: Path):
        """Export results to CSV format."""
        print(f"\nExporting to CSV: {output_path}")

        rows = []
        for result in self.results.values():
            row = {
                'demographic': result['demographic'],
                'question': result['question'],
                'cramers_v': result['cramers_v'],
                'phi': result['phi'] if result['phi'] is not None else np.nan,
                'chi_square': result['chi_square'],
                'p_value': result['p_value'],
                'p_value_corrected': result.get('p_value_corrected', result['p_value']),
                'significant': result.get('significant', result['p_value'] < self.alpha),
                'effect_size': result['effect_size'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'n_samples': result['n_samples'],
                'df': result['df'],
                'contingency_shape': f"{result['contingency_shape'][0]}x{result['contingency_shape'][1]}"
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values('cramers_v', ascending=False)
        df.to_csv(output_path, index=False, float_format='%.6f')

        print(f"  Exported {len(rows)} results")

    def export_to_json(self, output_path: Path):
        """Export results to JSON format with metadata."""
        print(f"\nExporting to JSON: {output_path}")

        # Count effect sizes
        effect_sizes = Counter(r['effect_size'] for r in self.results.values())
        n_significant = sum(1 for r in self.results.values() if r.get('significant', False))

        output = {
            'metadata': {
                'analysis_date': self.analysis_date,
                'data_path': self.data_path,
                'n_demographics': len(self.demographics),
                'n_questions': len(self.questions),
                'total_tests': len(self.results),
                'significance_level': self.alpha,
                'correction_method': self.correction_method,
                'n_significant': n_significant,
                'effect_size_distribution': dict(effect_sizes)
            },
            'demographics': self.demographics,
            'questions': self.questions,
            'correlations': []
        }

        for result in self.results.values():
            # Create clean result dict (exclude large contingency table)
            clean_result = {k: v for k, v in result.items() if k != 'contingency_table'}
            output['correlations'].append(clean_result)

        # Sort by Cramér's V
        output['correlations'].sort(key=lambda x: x['cramers_v'], reverse=True)

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"  Exported {len(output['correlations'])} results")

    def create_heatmap(self, output_path: Path):
        """Create correlation heatmap visualization."""
        print(f"\nCreating heatmap: {output_path}")

        # Build matrix: demographics × questions
        matrix = np.full((len(self.demographics), len(self.questions)), np.nan)
        sig_matrix = np.zeros((len(self.demographics), len(self.questions)), dtype=bool)

        for i, demo in enumerate(self.demographics):
            for j, question in enumerate(self.questions):
                if (demo, question) in self.results:
                    result = self.results[(demo, question)]
                    matrix[i, j] = result['cramers_v']
                    sig_matrix[i, j] = result.get('significant', False)

        # Create plot
        fig, ax = plt.subplots(figsize=(16, 10))

        # Heatmap
        sns.heatmap(
            matrix,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            xticklabels=self.questions,
            yticklabels=self.demographics,
            cbar_kws={'label': "Cramér's V"},
            vmin=0,
            vmax=0.5,
            ax=ax,
            linewidths=0.5,
            linecolor='gray'
        )

        # Add significance markers
        for i in range(len(self.demographics)):
            for j in range(len(self.questions)):
                if sig_matrix[i, j]:
                    ax.text(j + 0.5, i + 0.8, '*',
                           ha='center', va='center',
                           color='black', fontsize=16, fontweight='bold')

        plt.title(f"Demographic × Political Question Associations (Cramér's V)\n* indicates p < {self.alpha} after {self.correction_method} correction",
                 fontsize=14, pad=20)
        plt.xlabel('Political Questions', fontsize=12)
        plt.ylabel('Demographics', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved heatmap")

    def create_summary_report(self, output_path: Path):
        """Create text summary report."""
        print(f"\nCreating summary report: {output_path}")

        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ANES 2024 CORRELATION ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Analysis Date: {self.analysis_date}\n")
            f.write(f"Data Path: {self.data_path}\n")
            f.write(f"Significance Level: {self.alpha}\n")
            f.write(f"Correction Method: {self.correction_method}\n\n")

            # Overall statistics
            f.write("-" * 80 + "\n")
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 80 + "\n\n")

            total_tests = len(self.results)
            n_significant = sum(1 for r in self.results.values() if r.get('significant', False))

            f.write(f"Total Associations Tested: {total_tests}\n")
            f.write(f"Significant Associations (corrected p < {self.alpha}): {n_significant} ({100*n_significant/total_tests:.1f}%)\n\n")

            # Effect size distribution
            effect_sizes = Counter(r['effect_size'] for r in self.results.values())
            f.write("Effect Size Distribution:\n")
            f.write(f"  Small (V < 0.1):   {effect_sizes['small']:3d} ({100*effect_sizes['small']/total_tests:.1f}%)\n")
            f.write(f"  Medium (0.1-0.3):  {effect_sizes['medium']:3d} ({100*effect_sizes['medium']/total_tests:.1f}%)\n")
            f.write(f"  Large (V ≥ 0.3):   {effect_sizes['large']:3d} ({100*effect_sizes['large']/total_tests:.1f}%)\n\n")

            # Top 10 strongest associations
            f.write("-" * 80 + "\n")
            f.write("TOP 10 STRONGEST ASSOCIATIONS\n")
            f.write("-" * 80 + "\n\n")

            sorted_results = sorted(self.results.values(), key=lambda x: x['cramers_v'], reverse=True)
            for i, result in enumerate(sorted_results[:10], 1):
                sig_marker = "***" if result.get('significant', False) else ""
                f.write(f"{i:2d}. {result['demographic']} × {result['question']}:\n")
                f.write(f"    Cramér's V = {result['cramers_v']:.3f} {sig_marker}\n")
                f.write(f"    Effect Size: {result['effect_size']}\n")
                f.write(f"    p-value: {result['p_value']:.6f}\n")
                f.write(f"    Samples: {result['n_samples']}\n\n")

            # Demographics with most associations
            f.write("-" * 80 + "\n")
            f.write("DEMOGRAPHICS WITH MOST SIGNIFICANT ASSOCIATIONS\n")
            f.write("-" * 80 + "\n\n")

            demo_counts = {}
            for demo in self.demographics:
                sig_count = sum(1 for (d, q), r in self.results.items()
                              if d == demo and r.get('significant', False))
                total_count = sum(1 for (d, q) in self.results.keys() if d == demo)
                demo_counts[demo] = (sig_count, total_count)

            sorted_demos = sorted(demo_counts.items(), key=lambda x: x[1][0], reverse=True)
            for i, (demo, (sig, total)) in enumerate(sorted_demos, 1):
                pct = 100 * sig / total if total > 0 else 0
                f.write(f"{i:2d}. {demo}: {sig}/{total} significant ({pct:.1f}%)\n")

            # Questions with most demographic variation
            f.write("\n" + "-" * 80 + "\n")
            f.write("QUESTIONS WITH MOST DEMOGRAPHIC VARIATION\n")
            f.write("-" * 80 + "\n\n")

            question_counts = {}
            for question in self.questions:
                sig_count = sum(1 for (d, q), r in self.results.items()
                              if q == question and r.get('significant', False))
                total_count = sum(1 for (d, q) in self.results.keys() if q == question)
                question_counts[question] = (sig_count, total_count)

            sorted_questions = sorted(question_counts.items(), key=lambda x: x[1][0], reverse=True)
            for i, (question, (sig, total)) in enumerate(sorted_questions, 1):
                pct = 100 * sig / total if total > 0 else 0
                f.write(f"{i:2d}. {question}: {sig}/{total} demographics ({pct:.1f}%)\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"  Saved summary report")


def main():
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(
        description='Analyze correlations between demographics and political responses in ANES 2024',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all demographics and questions
  python analyze_anes_correlations.py --data path/to/anes.csv --output results/

  # Analyze specific demographics
  python analyze_anes_correlations.py --demographics gender age race

  # Apply Bonferroni correction
  python analyze_anes_correlations.py --correction bonferroni --alpha 0.05

  # Generate only CSV and heatmap
  python analyze_anes_correlations.py --no-json --no-report
        """
    )

    parser.add_argument('--data', type=str, default=DEFAULT_DATA_PATH,
                       help=f'Path to ANES CSV file (default: {DEFAULT_DATA_PATH})')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                       help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--demographics', nargs='+', default=None,
                       help='List of demographics to analyze (default: all)')
    parser.add_argument('--questions', nargs='+', default=None,
                       help='List of questions to analyze (default: all)')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance level (default: 0.05)')
    parser.add_argument('--correction', type=str, default='bonferroni',
                       choices=['bonferroni', 'fdr_bh', 'none'],
                       help='Multiple testing correction method (default: bonferroni)')
    parser.add_argument('--no-csv', action='store_true',
                       help='Skip CSV export')
    parser.add_argument('--no-json', action='store_true',
                       help='Skip JSON export')
    parser.add_argument('--no-heatmap', action='store_true',
                       help='Skip heatmap generation')
    parser.add_argument('--no-report', action='store_true',
                       help='Skip summary report')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize analyzer
    analyzer = ANESCorrelationAnalyzer(
        data_path=args.data,
        demographics=args.demographics,
        questions=args.questions,
        alpha=args.alpha,
        correction_method=args.correction
    )

    # Load data and compute correlations
    analyzer.load_data()
    analyzer.compute_all_correlations()

    # Export results
    if not args.no_csv:
        analyzer.export_to_csv(output_dir / 'anes_correlations.csv')

    if not args.no_json:
        analyzer.export_to_json(output_dir / 'anes_correlations.json')

    if not args.no_heatmap:
        analyzer.create_heatmap(output_dir / 'correlation_heatmap.png')

    if not args.no_report:
        analyzer.create_summary_report(output_dir / 'correlation_summary.txt')

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"\nFiles created:")
    if not args.no_csv:
        print(f"  - anes_correlations.csv")
    if not args.no_json:
        print(f"  - anes_correlations.json")
    if not args.no_heatmap:
        print(f"  - correlation_heatmap.png")
    if not args.no_report:
        print(f"  - correlation_summary.txt")


if __name__ == '__main__':
    main()
