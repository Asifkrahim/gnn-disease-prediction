from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "data" / "processed" / "prediction"

train = pd.read_parquet(PRED / "train_matched_labeled.parquet")
validation = pd.read_parquet(PRED / "validation_matched_labeled.parquet")
test = pd.read_parquet(PRED / "test_matched_labeled.parquet")

# Compute degrees from training positives only.
train_pos = train[train["label"] == 1]
drug_degree = train_pos["drug_id"].value_counts().to_dict()
disease_degree = train_pos["disease_id"].value_counts().to_dict()

def features(df):
    drug = np.log1p(df["drug_id"].map(drug_degree).fillna(0).to_numpy())
    disease = np.log1p(df["disease_id"].map(disease_degree).fillna(0).to_numpy())
    return np.column_stack([drug, disease, drug * disease])

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42),
)

model.fit(features(train), train["label"].to_numpy())

for name, df in [("Validation", validation), ("Test", test)]:
    y = df["label"].to_numpy()
    scores = model.predict_proba(df_features := features(df))[:, 1]
    print(f"\n{name}")
    print(f"ROC-AUC: {roc_auc_score(y, scores):.4f}")
    print(f"Average Precision: {average_precision_score(y, scores):.4f}")