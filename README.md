# BugInsight AI

**Research-grade platform for automated bug severity prediction, explainable AI, and evolutionary neural architecture search in software engineering.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Research Question

> *Can evolutionary architecture search, knowledge transfer, contrastive learning, and retrieval augmentation automatically build superior bug-analysis systems across diverse software projects?*

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/buginsight-ai.git
cd buginsight-ai
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
# Generate synthetic data for smoke testing
python scripts/generate_synthetic_data.py

# Preprocess raw data into standardised format
python -m data.preprocess
```

### 3. Run All Baseline Experiments (E1)

```bash
# Train RF, XGBoost, BiLSTM, CodeBERT across 5 seeds
python scripts/run_all_experiments.py

# Train a single model with a specific seed
python training/train_codebert.py --seed 42
python training/train_bilstm.py --seed 42
```

### 4. Evaluate

```bash
python evaluate.py --model codebert --seed 42
```

### 5. Generate Table 1

```bash
python scripts/run_all_experiments.py --skip-training
```

---

## Cloud Training (Kaggle)

1. Push this repository to GitHub
2. Open Kaggle and create a new notebook
3. Upload `notebooks/kaggle_baseline.ipynb`
4. Enable **GPU T4 x2**
5. Run all cells
6. Close your laptop — training continues in the cloud
7. Return later and download results from `outputs/`

---

## Repository Structure

```
buginsight-ai/
├── configs/               # config.yaml + config loader
├── data/                  # Dataset pipeline (preprocess, tokenize, split)
├── docs/                  # threats_to_validity.md, dataset_report.md
├── evaluation/            # Metrics computation, statistical tests, Table 1
├── models/                # RF, XGBoost, BiLSTM, CodeBERT classifiers
├── notebooks/             # Kaggle & Colab automation notebooks
├── outputs/               # metrics/, models/, logs/, figures/
├── scripts/               # Experiment runners, data generators
├── tests/                 # pytest unit & integration tests
├── training/              # Training loops, per-model training scripts
├── evaluate.py            # Unified evaluation entry point
├── utils.py               # Seeding, logging, device detection
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## Experiment Matrix

| Experiment | Purpose | Phase |
|:-----------|:--------|:------|
| **E1** | RF vs XGBoost vs BiLSTM vs CodeBERT | Phase 1 |
| **E2** | CodeBERT vs CodeBERT + Retrieval | Phase 2 |
| **E3** | CodeBERT vs CodeBERT + KICL | Phase 5 |
| **E4** | Manual Architecture vs NAS | Phase 6 |
| **E5** | Cold Start NAS vs Transfer NAS | Phase 7 |
| **E6** | Transfer NAS vs Self-Improving Memory NAS | Phase 8 |
| **E7** | Full System Ablation | Phase 9 |
| **E8** | Cross-Project Validation | Phase 9 |

---

## Statistical Rigour

- All experiments use **5 random seeds** (42, 123, 456, 789, 999)
- Results reported as **Mean ± Std**
- Statistical significance via **paired t-test** and **Wilcoxon signed-rank test**
- Threats to validity tracked in [`docs/threats_to_validity.md`](docs/threats_to_validity.md)

---

## Phase 1 Target Output: Table 1

| Model | Accuracy | Precision | Recall | F1 |
|:------|:---------|:----------|:-------|:---|
| Random Forest | — | — | — | — |
| XGBoost | — | — | — | — |
| BiLSTM | — | — | — | — |
| CodeBERT | — | — | — | — |

*All values: Mean ± Std across 5 seeds.*

---

## Configuration

All hyperparameters, paths, and settings are in [`configs/config.yaml`](configs/config.yaml). No hardcoded values exist in the codebase.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

If you use BugInsight AI in your research, please cite:

```bibtex
@software{buginsight2026,
  title={BugInsight AI: Evolutionary Architecture Search for Bug Severity Prediction},
  author={Singh},
  year={2026},
  url={https://github.com/YOUR_USERNAME/buginsight-ai}
}
```
