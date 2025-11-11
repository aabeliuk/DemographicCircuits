# Data Directory

Place your ANES (American National Election Studies) data files here.

## Required File

- `anes_timeseries_2024_csv_20250808.csv` - ANES 2024 Timeseries dataset

## Download Instructions

1. Visit the ANES website: https://electionstudies.org/
2. Navigate to Data Center
3. Download the ANES 2024 Time Series Study
4. Extract the CSV file to this directory

## Alternative: Custom Data Path

You can specify a custom path instead:

```bash
python robust_demographic_experiment.py --anes_data_path /path/to/your/anes_data.csv
```

## Data Format

The ANES CSV should contain columns for:
- Demographics: gender, age, race, education, etc.
- Political questions: abortion, immigration_policy, etc.

See the ANES codebook for detailed variable descriptions.
