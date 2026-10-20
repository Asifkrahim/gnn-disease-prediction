from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT=Path(__file__).resolve().parents[1]
PRED=ROOT/"data"/"processed"/"prediction"

train=pd.read_parquet(PRED/"train_labeled.parquet")
validation=pd.read_parquet(PRED/"validation_labeled.parquet")
test=pd.read_parquet(PRED/"test_labeled.parquet")

#Estimate entity popularity using training positives only
train_pos=train[train["label"]==1]
drug_degree=train_pos["drug_id"].value_counts().to_dict()
disease_degree=train_pos["disease_id"].value_counts().to_dict()

def make_features(df):
    drug_deg=df["drug_id"].map(drug_degree).fillna(0).to_numpy()
    disease_deg=df["disease_id"].map(disease_degree).fillna(0).to_numpy()

    return np.column_stack([
        np.log1p(drug_deg),
        np.log1p(disease_deg),
        np.log1p(drug_deg)*np.log1p(disease_deg),
    ])

X_train=make_features(train)
y_train=train["label"].to_numpy()

X_val=make_features(validation)
y_val=validation["label"].to_numpy()

X_test=make_features(test)
y_test=test["label"].to_numpy()

model=make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000,random_state=42),
)

model.fit(X_train,y_train)

for name,X,y in [
    ("Validation",X_val,y_val),
    ("Test",X_test,y_test),
]:
    scores=model.predict_proba(X)[:,1]
    print(f"\n{name}")
    print(f"ROC-AUC: {roc_auc_score(y,scores):.4f}")
    print(f"Average Precision: {average_precision_score(y,scores):.4f}")

print("\nBaseline Training complete.")   
    