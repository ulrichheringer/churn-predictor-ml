# SaaS Churn Prediction

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)  [![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)  [![MIT License](https://img.shields.io/badge/License-MIT-2ea44f?style=for-the-badge)](LICENSE)  ![Synthetic Data](https://img.shields.io/badge/Data-Synthetic-6b7280?style=for-the-badge)

Estimates SaaS account churn from usage, support, account age, and plan features with logistic regression.

## Data

Deterministic synthetic data is generated locally in `pipeline.py`. No private data or external download is required.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python cli.py evaluate
python cli.py train --model model.joblib
python cli.py predict --model model.joblib
```

## Evaluation results

With seed 42, a stratified holdout of 450 accounts reached 0.6711 accuracy, 0.2515 precision, 0.6462 recall, 0.3621 F1, and 0.7516 ROC AUC. Class weighting favors recall and leads to false positives. The synthetic churn rule does not represent real cancellations.

## Limitations

These scores come from artificial account data. A useful churn model would need a clearly defined churn window, real account history, and validation that respects time.
