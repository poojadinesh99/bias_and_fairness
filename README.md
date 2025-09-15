# Bias and Fairness Analysis in Medical Student Burnout Prediction

A comprehensive analysis of bias and fairness in machine learning models predicting medical student burnout, with implementation of bias mitigation techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Analysis Components](#analysis-components)
- [Results](#results)
- [Files Description](#files-description)
- [Methodology](#methodology)
- [Requirements](#requirements)
- [Contributing](#contributing)

## 🔍 Overview

This project examines bias and fairness in predicting medical student burnout using the Maslach Burnout Inventory (MBI) emotional exhaustion scores. The analysis focuses on identifying and mitigating bias across demographic groups, particularly by sex and native language.

### Key Features
- **Demographic Bias Analysis**: Examination of model performance across sex and language groups
- **Statistical Testing**: T-tests for group differences in burnout levels
- **Bias Mitigation**: Implementation of reweighting techniques to improve fairness
- **Comprehensive Visualizations**: Charts and plots for presentation and analysis
- **Fairness Metrics**: Calculation of False Negative Rates (FNR), AUC, and other fairness indicators

## 📊 Dataset

**Source**: Medical student survey data (`medteach.csv`)
- **Size**: 886 participants
- **Target Variable**: Emotional exhaustion (mbi_ex) - binarized at median (≥17.0)
- **Protected Attributes**: 
  - Sex (coded as 1=Man, 2=Woman, 3=Non-binary)
  - Native Language (glang) - top 3 languages + "other" category

### Demographics
- **Women**: 606 (68.4%)
- **Men**: 275 (31.0%)
- **Non-binary**: 5 (0.6%)

## 🚀 Installation

### Prerequisites
- Python 3.13+
- Virtual environment (recommended)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/poojadinesh99/bias_and_fairness.git
   cd bias_and_fairness
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate     # On Windows
   ```

3. **Install required packages**:
   ```bash
   pip install pandas numpy scikit-learn seaborn matplotlib scipy
   ```

## 📈 Usage

### Running the Complete Analysis

Execute the main analysis script:
```bash
python analysis_script_complete.py
```

### Alternative Scripts

- **Simple Analysis**: `python analysis_script_simple.py`
- **Working Version with Slides**: `python analysis_script_working.py`
- **Fairness Add-ons**: `import fairness_analysis_addons`

### Expected Runtime
- Complete analysis: ~30-60 seconds
- Generates 6 visualization files and 1 CSV metrics file

## 🔬 Analysis Components

### 1. Data Preprocessing
- Missing data handling
- Sex category recoding
- Language grouping (top 3 + other)
- Binary outcome creation (high/low burnout)

### 2. Exploratory Data Analysis
- Demographic distributions
- Statistical testing (t-tests)
- Visualization of group differences

### 3. Predictive Modeling
- **Algorithm**: Logistic Regression
- **Features**: Age, year, sex (one-hot), language (one-hot)
- **Evaluation**: AUC, precision, recall, confusion matrix

### 4. Fairness Assessment
- Group-wise performance metrics
- False Negative Rate (FNR) analysis
- AUC comparison across groups

### 5. Bias Mitigation
- **Technique**: Sample reweighting by language groups
- **Evaluation**: Before/after fairness comparison

## 📊 Results

### Key Findings

#### Statistical Differences
- **Significant gender difference** in burnout: Women (17.46) vs Men (15.62), p < 0.001
- **Effect size**: Cohen's d ≈ 0.5 (medium effect)

#### Model Performance
- **Overall AUC**: 0.58 (baseline) → 0.576 (after mitigation)
- **Baseline Precision/Recall**: 0.54 / 0.68

#### Fairness Analysis
**False Negative Rates by Language Group**:
- Language "1": 36.4% → 40.2% (after mitigation)
- "Other" languages: 16.7% → 27.8% (after mitigation)
- Languages "15" & "90": 16.7% → 16.7% (unchanged)

#### Gender-based Performance
- **Women**: Lower precision (0.57), higher recall (0.75)
- **Men**: Higher precision (0.40), lower recall (0.37)
- **Clear disparity** in false negative rates across groups

## 📁 Files Description

### Core Analysis Scripts
| File | Description |
|------|-------------|
| `analysis_script_complete.py` | Complete analysis with bias mitigation |
| `analysis_script_working.py` | Enhanced version with additional slide visualizations |
| `analysis_script_simple.py` | Simplified version for testing |
| `fairness_analysis_addons.py` | Additional fairness utility functions |

### Data Files
| File | Description |
|------|-------------|
| `medteach.csv` | Medical student survey dataset (886 participants) |
| `bias_report_group_metrics_glang.csv` | Fairness metrics before/after mitigation |

### Visualizations
| File | Description |
|------|-------------|
| `bias_report_count_sex.png` | Gender distribution bar chart |
| `bias_report_count_glang.png` | Language distribution bar chart |
| `bias_report_box_mbi_sex.png` | Burnout levels by gender (boxplot) |
| `slide4_fnr_before.png` | FNR by group before mitigation |
| `slide5_fnr_before_after.png` | FNR comparison before vs after |

### Configuration
| File | Description |
|------|-------------|
| `.gitignore` | Git ignore rules for Python projects |
| `README.md` | This file |

## 🔬 Methodology

### Bias Detection
1. **Statistical Testing**: T-tests for group differences
2. **Performance Disparities**: AUC and confusion matrix analysis
3. **Fairness Metrics**: FNR, FPR calculation by protected groups

### Bias Mitigation
1. **Reweighting**: Inverse probability weighting by language groups
2. **Evaluation**: Before/after comparison of fairness metrics
3. **Trade-off Analysis**: Performance vs fairness assessment

### Evaluation Framework
- **Individual Fairness**: Similar individuals receive similar outcomes
- **Group Fairness**: Equal performance across demographic groups
- **Algorithmic Accountability**: Transparent reporting of biases

## 📋 Requirements

### Python Packages
```
pandas==2.3.2
numpy==2.3.3
scikit-learn==1.7.2
seaborn==0.13.2
matplotlib==3.10.6
scipy==1.16.2
```

### System Requirements
- **OS**: macOS, Linux, Windows
- **Python**: 3.13+
- **Memory**: 4GB+ recommended
- **Storage**: 100MB for data and outputs

## 🤝 Contributing

### How to Contribute
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Code Style
- Follow PEP 8 Python style guidelines
- Include docstrings for functions
- Add comments for complex logic
- Ensure reproducibility with random seeds

## 📚 References

1. Maslach, C., & Jackson, S. E. (1981). The measurement of experienced burnout. Journal of Organizational Behavior, 2(2), 99-113.
2. Barocas, S., Hardt, M., & Narayanan, A. (2019). Fairness and Machine Learning. fairmlbook.org
3. Mehrabi, N., et al. (2021). A survey on bias and fairness in machine learning. ACM Computing Surveys, 54(6), 1-35.

---

**Last Updated**: September 15, 2025
**Version**: 1.0.0
