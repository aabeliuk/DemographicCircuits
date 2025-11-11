import pandas as pd
import numpy as np
from scipy.stats import pearsonr, chi2_contingency
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

# ANES 2024 Variable Mappings
ANES_2024_VARIABLES = {
    # Demographics
    'gender': {
        'code': 'V241551',
        'label': 'Gender',
        'values': {1: 'Man', 2: 'Woman', 3: 'Nonbinary', 4: 'Other'},
        'recode': {'Man': 'Man', 'Woman': 'Woman', 'Nonbinary': 'Other', 'Other': 'Other'}
    },
    'age': {
        'code': 'V241458x',
        'label': 'Age',
        'type': 'continuous',
        'bins': [(18, 26, 'Young Adult'), (27, 59, 'Adult'), (60, 120, 'Senior')]
    },
    'marital_status': {
        'code': 'V241459',
        'label': 'Marital Status',
        'values': {
            1: 'Married: spouse present',
            2: 'Married: spouse absent',
            3: 'Widowed',
            4: 'Divorced',
            5: 'Separated',
            6: 'Never married'
        },
        'recode': {
            'Married: spouse present': 'Married',
            'Married: spouse absent': 'Married',
            'Widowed': 'Previously married',
            'Divorced': 'Previously married',
            'Separated': 'Previously married',
            'Never married': 'Never married'
        }
    },
    'education': {
        'code': 'V241463',
        'label': 'Education Level',
        'values': {
            1: 'Less than 1st grade',
            2: '1st-4th grade',
            3: '5th-6th grade',
            4: '7th-8th grade',
            5: '9th grade',
            6: '10th grade',
            7: '11th grade',
            8: '12th grade no diploma',
            9: 'High school graduate',
            10: 'Some college',
            11: 'Associate occupational',
            12: 'Associate academic',
            13: 'Bachelor degree',
            14: 'Master degree',
            15: 'Professional degree',
            16: 'Doctorate degree',
            95: 'Other'
        },
        'recode': {
            'Less than 1st grade': 'Low',
            '1st-4th grade': 'Low',
            '5th-6th grade': 'Low',
            '7th-8th grade': 'Low',
            '9th grade': 'Low',
            '10th grade': 'Low',
            '11th grade': 'Low',
            '12th grade no diploma': 'Low',
            'High school graduate': 'Medium',
            'Some college': 'Medium',
            'Associate occupational': 'Medium',
            'Associate academic': 'Medium',
            'Bachelor degree': 'High',
            'Master degree': 'High',
            'Professional degree': 'High',
            'Doctorate degree': 'High',
            'Other': 'Other'
        }
    },
    'race': {
        'code': 'V241501x',
        'label': 'Race/Ethnicity',
        'values': {
            1: 'White', 2: 'Black', 3: 'Hispanic',
            4: 'Asian/PI', 5: 'Native American', 6: 'Multiple'
        },
        'recode': {
            'White': 'White',
            'Black': 'Black',
            'Hispanic': 'Latino',
            'Asian/PI': 'Asian American',
            'Native American': 'Native American',
            'Multiple': 'Multiracial'
        }
    },
    'urban_rural': {
        'code': 'V242341',
        'label': 'Urban/Rural',
        'values': {
            1: 'City person',
            2: 'Suburban person',
            3: 'Small-town person',
            4: 'Country (or rural) person',
            5: 'Something else'
        },
        'recode': {
            'City person': 'Urban',
            'Suburban person': 'Urban',
            'Small-town person': 'Rural',
            'Country (or rural) person': 'Rural'
            # 'Something else' excluded (only 74 respondents)
        }
    },
    'income': {
        'code': 'V241567x',
        'label': 'Household Income',
        'values': {
            1: 'Under $9,999',
            2: '$10,000 to $29,999',
            3: '$30,000 to $59,999',
            4: '$60,000 to $99,999',
            5: '$100,000 to $249,999',
            6: '$250,000 or more'
        },
        'recode': {
            'Under $9,999': 'Low',
            '$10,000 to $29,999': 'Low',
            '$30,000 to $59,999': 'Middle',
            '$60,000 to $99,999': 'Middle',
            '$100,000 to $249,999': 'High',
            '$250,000 or more': 'High'
        }
    },

    # Political Identity
    'ideology': {
        'code': 'V241709',
        'label': 'Political Ideology',
        'values': {
            1: 'Extremely liberal', 2: 'Liberal', 3: 'Slightly liberal',
            4: 'Moderate', 5: 'Slightly conservative', 6: 'Conservative',
            7: 'Extremely conservative'
        },
        'recode': {
            'Extremely liberal': 'Left', 'Liberal': 'Left', 'Slightly liberal': 'Left',
            'Moderate': 'Center',
            'Slightly conservative': 'Right', 'Conservative': 'Right',
            'Extremely conservative': 'Right'
        }
    },
    'party_id': {
        'code': 'V241707',
        'label': 'Party Identification',
        'values': {
            1: 'Strong Democrat', 2: 'Not strong Democrat', 3: 'Lean Democrat',
            4: 'Independent', 5: 'Lean Republican', 6: 'Not strong Republican',
            7: 'Strong Republican'
        },
        'recode': {
            'Strong Democrat': 'Democrat', 'Not strong Democrat': 'Democrat',
            'Lean Democrat': 'Democrat', 'Independent': 'Independent',
            'Lean Republican': 'Republican', 'Not strong Republican': 'Republican',
            'Strong Republican': 'Republican'
        }
    },
    'religion': {
        'code': 'V241723',
        'label': 'Religious Affiliation',
        'values': {
            1: 'Protestant', 2: 'Catholic', 3: 'Jewish', 4: 'Muslim',
            5: 'Atheist', 6: 'Agnostic', 7: 'Other', 8: 'Nothing'
        },
        'recode': {
            'Protestant': 'Religious', 'Catholic': 'Religious', 'Jewish': 'Religious',
            'Muslim': 'Religious', 'Other': 'Religious',
            'Atheist': 'Not Religious', 'Agnostic': 'Not Religious', 'Nothing': 'Not Religious'
        }
    },

    # Political Engagement
    'political_interest': {
        'code': 'V242400',
        'label': 'Interest in Politics',
        'values': {1: 'Very interested', 2: 'Somewhat interested',
                   3: 'Not very interested', 4: 'Not at all interested'}
    },
    'political_attention': {
        'code': 'V241004',
        'label': 'Attention to Politics',
        'values': {1: 'Always', 2: 'Most of the time',
                   3: 'About half', 4: 'Some of the time'}
    },

    # Policy Issues
    'abortion': {
        'code': 'V241302',
        'label': 'There has been some discussion about abortion during recent years. Which one of the opinions on this page best agrees with your view?',
        'values': {
            1: 'Never permitted', 2: 'Only rape/incest/danger',
            3: 'After need established', 4: 'Always choice', 5: 'Other'
        },
        'recode': {
            'Never permitted': 'Always illegal',
            'Only rape/incest/danger': 'Legal specific cases',
            'After need established': 'Legal specific cases',
            'Always choice': 'Always legal',
            'Other': 'Always legal'
        }
    },
    'death_penalty': {
        'code': 'V241306',
        'label': 'Do you favor or oppose the death penalty for persons convicted of murder?',
        'values': {1: 'Favor', 2: 'Oppose'}
    },
    'military_force': {
        'code': 'V241313',
        'label': 'How willing should the United States be to use military force to solve international problems?',
        'values': {
            1: 'Extremely willing', 2: 'Very willing', 3: 'Moderately willing',
            4: 'A little willing', 5: 'Not at all willing'
        }
    },
    'defense_spending': {
        'code': 'V241242',
        'label': 'Where would you place yourself on this scale, or haven’t you thought much about this',
        'type': '7pt_scale',
        'endpoints': ('Greatly decrease defense spending', 'Greatly increase defense spending')
    },
    'govt_jobs': {
        'code': 'V241252',
        'label': 'Where would you place yourself on this scale, or haven’t you thought much about this',
        'type': '7pt_scale',
        'endpoints': ('Government should see to jobs and standard of living', 'Government should let each person get ahead on own')
    },
    'govt_help_blacks': {
        'code': 'V241255',
        'label': 'Where would you place yourself on this scale, or haven’t you thought much about this?',
        'type': '7pt_scale',
        'endpoints': ('Government should help blacks', 'Blacks should help themselves')
    },
    'colleges_opinion': {
        'code': 'V241285',
        'label': 'Do you approve, disapprove, or neither approve nor disapprove of how most colleges and universities are run these days?',
        'values': {1: 'Approve', 2: 'Disapprove', 3: 'Neither'}
    },
    'dei_opinion': {
        'code': 'V241289',
        'label': 'How strongly approve or disapprove Diversity, Equity, and Inclusion (DEI)',
        'values': {1: 'A great deal', 2: 'A moderate amount', 3: 'A little', 4: "Don't know"}
    },
    'journalist_access': {
        'code': 'V241331',
        'label': "Do you favor, oppose, or neither favor nor oppose elected officials restricting journalists' access to information about government decision-making?",
        'values': {1: 'Favor', 2: 'Oppose', 3: 'Neither'}
    },
    'transgender_bathrooms': {
        'code': 'V241370',
        'label': "Do you favor, oppose, or neither favor nor oppose allowing transgender people to use public bathrooms that match the gender they identify with?",
        'values': {1: 'Favor', 2: 'Oppose', 3: 'Neither'}
    },
    'birthright_citizenship': {
        'code': 'V241387',
        'label': "Some people have proposed that the U.S. Constitution should be changed so that the children of unauthorized immigrants do not automatically get citizenship if they are born in this country. Do you favor, oppose, or neither favor nor oppose this proposal?",
        'values': {1: 'Favor', 2: 'Oppose', 3: 'Neither'}
    },
    'immigration_policy': {
        'code': 'V241386',
        'label': "Which comes closest to your view about what government policy should be toward unauthorized immigrants now living in the United States?",
        'values': {
            1: 'Make all unauthorized immigrants felons and send them back to their home country',
            2: 'Have guest worker program that allows unauthorized immigrants to remain in the US to work but only for a limited amount of time',
            3: 'Allow unauthorized immigrants to remain in US & qualify for citizenship if they meet certain requirements',
            4: 'Allow unauthorized immigrants to remain in US & qualify for citizenship without penalties'
        },
        'recode': {
            'Make all unauthorized immigrants felons and send them back to their home country': 'Restrictive',
            'Have guest worker program that allows unauthorized immigrants to remain in the US to work but only for a limited amount of time': 'Moderate',
            'Allow unauthorized immigrants to remain in US & qualify for citizenship if they meet certain requirements': 'Permissive',
            'Allow unauthorized immigrants to remain in US & qualify for citizenship without penalties': 'Highly Permissive'
        }
    }
}

# Missing value codes in ANES
MISSING_CODES = [-9, -8, -7, -6, -5, -4, -3, -2, -1, 99]

@dataclass
class DemographicAssociation:
    """Represents a demographic-political association with strength and direction"""
    demographic_var: str
    political_var: str
    association_strength: float
    direction: int  # 1 or -1
    confidence_interval: Tuple[float, float]
    sample_size: int
    effect_size: str  # 'small', 'medium', 'large'

class ANESAssociationLearner:
    """Learn demographic-political associations from ANES survey data"""
    
    def __init__(self, anes_data_path: Optional[str] = None):
        """Initialize with ANES 2024 data

        Args:
            anes_data_path: Path to ANES CSV file. If None, uses default path.
        """
        if anes_data_path is None:
            anes_data_path = 'anes_timeseries_2024_csv_20250808/anes_timeseries_2024_csv_20250808.csv'

        self.anes_data = self._load_real_anes_data(anes_data_path)
        self.associations = {}
        self.demographic_vars = []
        self.political_vars = []

    def _load_real_anes_data(self, csv_path: str) -> pd.DataFrame:
        """Load and preprocess real ANES 2024 data"""
        print(f"Loading ANES 2024 data from {csv_path}...")

        # Load only the columns we need
        required_cols = [var_info['code'] for var_info in ANES_2024_VARIABLES.values()]
        df = pd.read_csv(csv_path, usecols=required_cols, low_memory=False)

        print(f"Loaded {len(df)} respondents with {len(required_cols)} variables")

        # Clean missing values
        df = self._clean_missing_values(df)

        # Recode variables
        df = self._recode_variables(df)

        print(f"After preprocessing: {len(df)} valid respondents")
        return df

    def _clean_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace ANES missing value codes with NaN"""
        df_clean = df.copy()

        for col in df_clean.columns:
            # Replace missing codes with NaN
            df_clean[col] = df_clean[col].replace(MISSING_CODES, np.nan)

        return df_clean

    def _recode_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recode ANES variables into meaningful categories"""
        df_recoded = pd.DataFrame()

        for var_name, var_info in ANES_2024_VARIABLES.items():
            code = var_info['code']

            if code not in df.columns:
                continue

            # Handle continuous variables (e.g., age)
            if var_info.get('type') == 'continuous':
                if var_name == 'age':
                    # Bin age into categories
                    df_recoded[var_name] = pd.cut(
                        df[code],
                        bins=[b[0] for b in var_info['bins']] + [var_info['bins'][-1][1]],
                        labels=[b[2] for b in var_info['bins']],
                        include_lowest=True
                    )
                else:
                    df_recoded[var_name] = df[code]

            # Handle categorical variables with value labels
            elif 'values' in var_info:
                # Map numeric codes to labels
                df_recoded[var_name] = df[code].map(var_info['values'])

                # Apply recoding if specified
                if 'recode' in var_info:
                    df_recoded[var_name] = df_recoded[var_name].map(var_info['recode'])

            # Handle 7-point scales
            elif var_info.get('type') == '7pt_scale':
                # For 7-point scales, recode to 3 categories: Low (1-3), Mid (4), High (5-7)
                def recode_7pt(val):
                    if pd.isna(val):
                        return np.nan
                    elif val <= 3:
                        return var_info['endpoints'][0]
                    elif val == 4:
                        return 'Moderate'
                    else:
                        return var_info['endpoints'][1]

                df_recoded[var_name] = df[code].apply(recode_7pt)

            # Handle other categorical variables
            else:
                df_recoded[var_name] = df[code]

        return df_recoded

    
    def calculate_categorical_association(self, demo_var: str, pol_var: str) -> DemographicAssociation:
        """Calculate association strength between categorical demographic and political variables"""
        
        # Create contingency table
        contingency_table = pd.crosstab(self.anes_data[demo_var], self.anes_data[pol_var])
        
        # Calculate Cramér's V (association strength)
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        n = contingency_table.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
        
        # Determine effect size
        if cramers_v < 0.1:
            effect_size = 'small'
        elif cramers_v < 0.3:
            effect_size = 'medium'
        else:
            effect_size = 'large'
            
        # Calculate direction (simplified for binary political outcomes)
        # For more complex categoricals, this would need refinement
        direction = 1  # placeholder
        
        # Calculate confidence interval (approximate)
        se = np.sqrt(cramers_v * (1 - cramers_v) / n)
        ci_lower = max(0, cramers_v - 1.96 * se)
        ci_upper = min(1, cramers_v + 1.96 * se)
        
        return DemographicAssociation(
            demographic_var=demo_var,
            political_var=pol_var,
            association_strength=cramers_v,
            direction=direction,
            confidence_interval=(ci_lower, ci_upper),
            sample_size=n,
            effect_size=effect_size
        )
    
    def calculate_all_associations(self, demographic_vars: Optional[List[str]] = None,
                                   political_vars: Optional[List[str]] = None) -> Dict[str, DemographicAssociation]:
        """Calculate associations between all demographic and political variables

        Args:
            demographic_vars: List of demographic variable names. If None, uses default set.
            political_vars: List of political variable names. If None, uses default set.
        """

        # Use default variable sets if not specified
        if demographic_vars is None:
            demographic_vars = ['gender', 'age', 'race', 'education', 'marital_status', 'income']

        if political_vars is None:
            political_vars = ['ideology', 'party_id', 'abortion', 'immigration_policy',
                            'death_penalty', 'dei_opinion', 'transgender_bathrooms']

        # Filter to only variables that exist in the dataset
        demographic_vars = [v for v in demographic_vars if v in self.anes_data.columns]
        political_vars = [v for v in political_vars if v in self.anes_data.columns]

        associations = {}

        print("Calculating demographic-political associations from ANES data...")
        print("=" * 60)
        print(f"Demographics: {demographic_vars}")
        print(f"Political: {political_vars}")
        print("=" * 60)

        for demo_var in demographic_vars:
            for pol_var in political_vars:
                try:
                    association = self.calculate_categorical_association(demo_var, pol_var)
                    key = f"{demo_var}_{pol_var}"
                    associations[key] = association

                    print(f"{demo_var} → {pol_var}:")
                    print(f"  Association strength: {association.association_strength:.3f} ({association.effect_size})")
                    print(f"  Sample size: {association.sample_size}")
                    print(f"  95% CI: [{association.confidence_interval[0]:.3f}, {association.confidence_interval[1]:.3f}]")
                    print()
                except Exception as e:
                    print(f"Error calculating {demo_var} → {pol_var}: {e}")
                    print()

        self.associations = associations
        return associations
    
    

    def visualize_associations(self, save_path: Optional[str] = None,
                               demographic_vars: Optional[List[str]] = None,
                               political_vars: Optional[List[str]] = None):
        """Visualize ANES demographic-political associations

        Args:
            save_path: Path to save the visualization
            demographic_vars: List of demographic variables to visualize
            political_vars: List of political variables to visualize
        """

        if not self.associations:
            self.calculate_all_associations()

        # Auto-detect variables if not specified
        if demographic_vars is None or political_vars is None:
            # Extract from associations keys
            all_keys = list(self.associations.keys())
            if all_keys:
                demo_set = set()
                pol_set = set()
                for key in all_keys:
                    parts = key.rsplit('_', 1)
                    if len(parts) == 2:
                        demo_part, pol_part = parts
                        # Handle multi-word variables
                        if '_' in demo_part:
                            demo_set.add(demo_part)
                        pol_set.add(pol_part)

                if demographic_vars is None:
                    demographic_vars = sorted(list(demo_set))[:6]  # Limit to 6 for readability
                if political_vars is None:
                    political_vars = sorted(list(pol_set))[:7]  # Limit to 7 for readability

        # Create association strength matrix
        matrix = np.zeros((len(demographic_vars), len(political_vars)))

        for i, demo_var in enumerate(demographic_vars):
            for j, pol_var in enumerate(political_vars):
                key = f"{demo_var}_{pol_var}"
                if key in self.associations:
                    matrix[i, j] = self.associations[key].association_strength

        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(matrix,
                   xticklabels=[v.replace('_', ' ').title() for v in political_vars],
                   yticklabels=[v.replace('_', ' ').title() for v in demographic_vars],
                   annot=True,
                   cmap='RdYlBu_r',
                   center=0.2,
                   fmt='.3f',
                   cbar_kws={'label': "Cramér's V"})

        plt.title('Demographic-Political Association Strengths (ANES 2024 Data)')
        plt.xlabel('Political Variables')
        plt.ylabel('Demographic Variables')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        plt.show()
    
    def generate_correction_targets(self) -> Dict[str, float]:
        """Generate target circuit strengths for bias correction"""
        
        target_strengths = self.create_target_circuit_strengths()
        
        print("\nTarget Circuit Strengths for Bias Correction:")
        print("=" * 50)
        
        for key, strength in target_strengths.items():
            print(f"{key}: {strength:.3f}")
            
        return target_strengths



