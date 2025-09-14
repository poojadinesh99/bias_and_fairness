# Bias and Fairness Analysis

This repository contains a comprehensive bias and fairness analysis of medical student burnout data.

## Environment Setup

This project uses a Python virtual environment with the following packages:
- pandas 2.3.2
- numpy 2.3.3  
- scikit-learn 1.7.2
- seaborn 0.13.2
- matplotlib 3.10.6
- scipy 1.16.2

## Files

### Analysis Scripts
- `analysis_script_complete.py` - Complete bias and fairness analysis with mitigation
- `analysis_script_simple.py` - Simplified version of the analysis
- `analysis_script.py` - Original analysis script

### Data
- `medteach.csv` - Medical student survey data (886 participants)

### Outputs
- `bias_report_count_sex.png` - Distribution by sex
- `bias_report_count_glang.png` - Distribution by language groups  
- `bias_report_box_mbi_sex.png` - Emotional exhaustion by sex
- `bias_report_group_metrics_glang.csv` - Fairness metrics before/after mitigation

## Analysis Summary

The analysis examines bias and fairness in predicting medical student burnout across demographic groups:

- **Dataset**: 886 medical students
- **Outcome**: High burnout (emotional exhaustion ≥ 17.0)
- **Key Finding**: Significant gender differences in burnout levels (p < 0.001)
- **Model Performance**: AUC = 0.58 (baseline), 0.576 (after bias mitigation)

## Usage

To run the complete analysis:

```bash
python analysis_script_complete.py
```

This will generate all visualizations and fairness metrics.
