"""SaaS churn classification with synthetic usage and account features."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
NUMERIC = ["tenure_months", "weekly_sessions", "support_tickets_90d", "seats", "days_since_login", "monthly_spend"]
FEATURES = NUMERIC + ["plan"]


def _dataset(seed=SEED, rows=1800):
    rng = np.random.default_rng(seed)
    plan = rng.choice(["free", "basic", "pro", "enterprise"], rows, p=[0.18, 0.37, 0.32, 0.13])
    tenure = rng.integers(1, 73, rows)
    sessions = np.clip(rng.normal(8, 4.5, rows), 0, 30)
    tickets = rng.poisson(1.5, rows)
    seats = np.maximum(1, rng.poisson(np.where(plan == "enterprise", 18, 4), rows))
    inactive = np.clip(rng.gamma(2.0, 12.0, rows), 0, 100)
    spend = np.where(plan == "free", 0, np.where(plan == "basic", 35, np.where(plan == "pro", 95, 360))) * rng.uniform(0.8, 1.2, rows)
    X = pd.DataFrame({"tenure_months": tenure, "weekly_sessions": sessions, "support_tickets_90d": tickets,
                      "seats": seats, "days_since_login": inactive, "monthly_spend": spend, "plan": plan})
    logit = (-2.5 + 0.045 * inactive + 0.24 * tickets - 0.11 * sessions - 0.018 * tenure
             + 0.85 * (plan == "free") + 0.35 * (plan == "basic") - 0.45 * (plan == "enterprise")
             + rng.normal(0, 0.8, rows))
    probability = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, probability)
    return X, y


def _model(seed=SEED):
    preprocess = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("plan", OneHotEncoder(handle_unknown="ignore"), ["plan"]),
    ])
    return make_pipeline(preprocess, LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))


def sample_input():
    return {"tenure_months": 8, "weekly_sessions": 1, "support_tickets_90d": 4, "seats": 2,
            "days_since_login": 36, "monthly_spend": 35, "plan": "basic"}


def train_model(seed=SEED):
    X, y = _dataset(seed)
    return _model(seed).fit(X, y)


def predict(model, payload):
    row = pd.DataFrame([payload], columns=FEATURES)
    probability = float(model.predict_proba(row)[0, 1])
    return {"churn_risk": bool(probability >= 0.5), "churn_probability": round(probability, 4)}


def evaluate(seed=SEED):
    X, y = _dataset(seed)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    model = _model(seed).fit(x_train, y_train)
    pred = model.predict(x_test)
    return {"accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])), 4),
            "test_rows": int(len(y_test)), "churn_test_rows": int(y_test.sum())}
