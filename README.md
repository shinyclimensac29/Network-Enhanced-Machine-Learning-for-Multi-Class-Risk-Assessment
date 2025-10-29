# Network-Enhanced Machine Learning for Risk Assessment in International Notification Systems

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

This repository contains the complete implementation and analysis code for the paper:

**"Network-Enhanced Machine Learning for Multi-Class Risk Assessment: A Case Study of International Food Safety Notifications"**

*Authors: [Your Name], [Supervisor Name]*

---

## 📋 Overview

This project develops and evaluates a network-enhanced machine learning framework for multi-class risk assessment in international food safety notification systems. We integrate graph-theoretic analysis with ensemble learning to predict six-class risk decisions from 1,364 notifications across 96 countries (2020-2025).

### Key Contributions

- **Methodological Innovation**: First systematic integration of 11 network centrality metrics with ensemble ML for multi-class risk classification
- **Network Structure Characterization**: Identified 28 nodes with significantly elevated PageRank beyond degree expectations
- **Predictive Benchmarks**: Gradient Boosting with SMOTE achieved F1-macro = 0.492, ROC-AUC = 0.890
- **Network Feature Paradox**: Network features show strong statistical associations (p < 10⁻¹⁶) but minimal predictive contribution (ΔF1 = -0.005)
- **Temporal Drift Quantification**: 16.6% F1-macro degradation under temporal validation
- **Fairness Analysis**: 90% performance gap across network communities (F1: 0.286 to 0.544)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) Conda for environment management

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/[username]/network-risk-assessment.git
cd network-risk-assessment
```

2. **Create virtual environment (recommended)**
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n network-risk python=3.9
conda activate network-risk
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Full Pipeline

Execute the complete analysis pipeline with a single command:
```bash
python run_analysis.py
```

This will:
- Load and preprocess the data
- Construct the notification network
- Compute centrality metrics and null models (100 randomizations)
- Train and evaluate all models
- Perform ablation studies and permutation importance
- Generate all figures and tables
- Save results to `results/` directory

**Estimated runtime**: ~4 hours on 8-core CPU with 32 GB RAM

---

## 📁 Repository Structure
```
network-risk-assessment/
│
├── data/
│   ├── raw/                          # Original notification data (anonymized)
│   ├── processed/                    # Processed feature matrices
│   └── README.md                     # Data documentation
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb    # EDA and visualization
│   ├── 02_network_construction.ipynb    # Network building and metrics
│   ├── 03_model_training.ipynb          # Model training and evaluation
│   ├── 04_temporal_validation.ipynb     # Temporal drift analysis
│   └── 05_fairness_analysis.ipynb       # Community-based fairness
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Data loading and preprocessing
│   ├── network_builder.py           # Network construction
│   ├── centrality_metrics.py        # Centrality computation
│   ├── null_models.py               # Configuration model testing
│   ├── feature_engineering.py       # Feature extraction
│   ├── models.py                    # Model definitions
│   ├── evaluation.py                # Evaluation metrics
│   ├── statistical_tests.py         # Kruskal-Wallis, Dunn, McNemar
│   ├── visualization.py             # Plotting functions
│   └── utils.py                     # Helper functions
│
├── results/
│   ├── figures/                     # All paper figures (PDF/PNG)
│   ├── tables/                      # LaTeX tables and CSV results
│   ├── models/                      # Trained model checkpoints
│   └── logs/                        # Training logs
│
├── tests/
│   ├── test_network_builder.py      # Unit tests for network construction
│   ├── test_centrality.py           # Tests for centrality metrics
│   └── test_models.py               # Tests for ML models
│
├── paper/
│   ├── manuscript.tex               # LaTeX manuscript
│   ├── references.bib               # BibTeX references
│   └── supplementary.pdf            # Supplementary materials
│
├── run_analysis.py                  # Main pipeline script
├── requirements.txt                 # Python dependencies
├── environment.yml                  # Conda environment (optional)
├── setup.py                         # Package installation
├── LICENSE                          # MIT License
└── README.md                        # This file
```

---

## 📊 Dataset

The dataset comprises **1,364 food safety notifications** from January 2020 to October 2025, including:

- **96 countries** (as notifiers or origins)
- **6 risk categories**: no risk (n=10), not serious (n=308), undecided (n=129), potential risk (n=85), potentially serious (n=61), serious (n=771)
- **Imbalance ratio**: 77:1 (majority to minority class)
- **Attributes**: notifying country, origin country, product category, hazard type, risk decision, date, subject text

**Note**: Due to data sensitivity, we provide anonymized aggregate network statistics. Original data can be requested from [data provider] subject to data use agreements.

### Data Files

- `data/raw/notifications_anonymized.csv`: Anonymized notification records
- `data/processed/network_edgelist.csv`: Network edge list (origin → notifier with weights)
- `data/processed/centrality_metrics.csv`: Computed centrality for all nodes
- `data/processed/feature_matrix.csv`: Complete feature matrix (114 features × 1,364 samples)

---

## 🔧 Usage Examples

### 1. Load Data and Build Network
```python
from src.data_loader import load_notifications
from src.network_builder import build_notification_network

# Load data
df = load_notifications('data/raw/notifications_anonymized.csv')

# Build network
G = build_notification_network(df)
print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
```

### 2. Compute Centrality Metrics
```python
from src.centrality_metrics import compute_all_centralities

# Compute all 11 centrality metrics
centrality_df = compute_all_centralities(G)
print(centrality_df.head())
```

### 3. Run Null Model Testing
```python
from src.null_models import run_null_model_analysis

# Generate 100 randomized networks and compute z-scores
results = run_null_model_analysis(G, n_randomizations=100, metric='pagerank')
significant_nodes = results[results['fdr_corrected_p'] < 0.05]
print(f"Found {len(significant_nodes)} significant nodes")
```

### 4. Train Models
```python
from src.models import train_gradient_boosting
from src.evaluation import evaluate_model

# Train model
model = train_gradient_boosting(X_train, y_train, use_smote=True)

# Evaluate
metrics = evaluate_model(model, X_test, y_test)
print(f"F1-Macro: {metrics['f1_macro']:.3f}")
print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
```

### 5. Ablation Study
```python
from src.models import ablation_study

# Quantify feature group contributions
ablation_results = ablation_study(
    X_train, y_train, X_test, y_test,
    feature_groups=['network', 'temporal', 'content', 'categorical']
)
print(ablation_results)
```

### 6. Temporal Validation
```python
from src.evaluation import temporal_validation

# Chronological 80-20 split
temporal_results = temporal_validation(df, date_column='_date_parsed')
print(f"Temporal F1-Macro: {temporal_results['f1_macro']:.3f}")
print(f"Degradation: {temporal_results['degradation_pct']:.1f}%")
```

### 7. Fairness Analysis
```python
from src.evaluation import fairness_analysis

# Stratify by community
fairness_results = fairness_analysis(
    model, X_test, y_test, communities,
    min_samples=5
)
print(f"F1 Range: {fairness_results['f1_range']:.3f}")
print(f"Variance: {fairness_results['f1_variance']:.4f}")
```

---

## 📈 Reproducing Paper Results

### Main Results (Tables and Figures)

All results can be reproduced by running:
```bash
python run_analysis.py --reproduce-paper
```

This generates:

**Figures:**
- `figure_1_temporal_trends.pdf`: Exploratory data analysis
- `figure_2_network_properties.pdf`: Network structure characterization
- `figure_3_null_model.pdf`: Null model significance testing
- `figure_4_confusion_matrix.pdf`: Model performance
- `figure_5_feature_importance.pdf`: Feature importance analysis
- `figure_6_statistical_tests.pdf`: Kruskal-Wallis results
- `figure_7_temporal_comparison.pdf`: Temporal validation
- `figure_8_fairness_analysis.pdf`: Community-based fairness
- `figure_9_cross_validation.pdf`: 5-fold CV results

**Tables:**
- `table_1_class_distribution.csv`: Class imbalance statistics
- `table_2_network_summary.csv`: Global network properties
- `table_3_model_comparison.csv`: Model performance comparison
- `table_4_kruskal_wallis.csv`: Statistical test results
- `table_5_ablation.csv`: Feature group contributions
- `table_6_fairness.csv`: Per-community performance

### Individual Experiments

Run specific analyses:
```bash
# Network analysis only
python run_analysis.py --network-only

# Model training only
python run_analysis.py --models-only

# Temporal validation only
python run_analysis.py --temporal-only

# Fairness analysis only
python run_analysis.py --fairness-only
```

---

## 🧪 Testing

Run unit tests to verify implementation:
```bash
# All tests
pytest tests/

# Specific test modules
pytest tests/test_network_builder.py
pytest tests/test_centrality.py
pytest tests/test_models.py

# With coverage report
pytest --cov=src tests/
```

---

## 📦 Dependencies

### Core Libraries

- **Network Analysis**: `networkx==3.2`, `python-louvain==0.16`
- **Machine Learning**: `scikit-learn==1.4.0`, `imbalanced-learn==0.12.0`
- **Data Processing**: `pandas==1.3.5`, `numpy==1.21.5`
- **Statistical Analysis**: `scipy==1.7.3`, `statsmodels==0.13.5`
- **Visualization**: `matplotlib==3.5.1`, `seaborn==0.11.2`

### Full Requirements

See `requirements.txt` for complete list with pinned versions.
```bash
pip install -r requirements.txt
```

Or use conda:
```bash
conda env create -f environment.yml
```

---

## 📝 Citation

If you use this code or methodology in your research, please cite:
```bibtex

\author[1]{\fnm{Shinyclimensa} \sur{C}}\email{shinyclimensa.c2022@vitstudent.ac.in}
\author*[1]{\fnm{Parthiban} \sur{A}}\email{parthiban.a@vit.ac.in}

\affil[1]{
  \orgdiv{School of Advanced Sciences}, 
  \orgdiv{Department of Mathematics}, 
  \orgname{Vellore Institute of Technology}, 
  \city{Vellore}, 
  \state{Tamil Nadu}, 
  \postcode{632014}, 
  \country{India}
}

```

**Preprint**: [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- Code passes all tests (`pytest tests/`)
- New features include unit tests
- Code follows PEP 8 style guidelines
- Documentation is updated

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 👥 Authors

**[Your Name]**
- Email: your.email@institution.edu
- GitHub: [@yourusername](https://github.com/yourusername)
- ORCID: [0000-0000-0000-0000](https://orcid.org/0000-0000-0000-0000)

**[Supervisor Name]**
- Email: supervisor.email@institution.edu
- ORCID: [0000-0000-0000-0000](https://orcid.org/0000-0000-0000-0000)

---

## 🙏 Acknowledgments

- Data providers for making anonymized notification records available
- [Institution Name] for computational resources
- Reviewers and collaborators for valuable feedback
- Open-source community for excellent scientific computing tools

---

## 📧 Contact

For questions, suggestions, or collaboration opportunities:

- **General inquiries**: your.email@institution.edu
- **Technical issues**: Open an issue on GitHub
- **Data access requests**: Contact data provider at [link]

---

## 🔗 Related Resources

- **Paper**: [Link to published paper]
- **Preprint**: [Link to arXiv]
- **Supplementary Materials**: [Link to supplementary PDF]
- **Presentation Slides**: [Link to slides]
- **Project Website**: [Link if available]

---

## 📊 Performance Summary

| Metric | Value |
|--------|-------|
| **Best Model** | Gradient Boosting + SMOTE |
| **F1-Macro** | 0.492 |
| **Balanced Accuracy** | 0.474 |
| **Cohen's Kappa** | 0.562 |
| **ROC-AUC** | 0.890 |
| **Temporal Degradation** | -16.6% |
| **Training Time** | ~4 hours |

---

## 🗺️ Roadmap

- [x] Initial release with paper code
- [x] Unit tests and documentation
- [ ] Interactive visualization dashboard
- [ ] Real-time prediction API
- [ ] Extension to other notification systems
- [ ] Integration with graph neural networks
- [ ] Fairness-aware training algorithms
- [ ] Online learning for concept drift

---

## 📚 Additional Documentation

- [Installation Guide](docs/installation.md)
- [API Reference](docs/api_reference.md)
- [Tutorial Notebooks](notebooks/README.md)
- [FAQ](docs/faq.md)
- [Troubleshooting](docs/troubleshooting.md)

---

## ⚠️ Disclaimer

This software is provided for research purposes only. The models and predictions should not be used as the sole basis for regulatory decisions without human expert review. Performance may vary on different datasets and domains.

---

**Last Updated**: January 2025

**Version**: 1.0.0

---

⭐ If you find this work useful, please consider giving it a star on GitHub!
