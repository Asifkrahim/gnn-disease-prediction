from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = DATA / "prediction"

OUT.mkdir(parents=True, exist_ok=True)

edges = pd.read_parquet(DATA / "drug_disease_edges.parquet")

# Keep one record per drug-disease pair.
edges = edges[["source_id", "target_id"]].drop_duplicates()
edges = edges.rename(columns={
    "source_id": "drug_id",
    "target_id": "disease_id",
})

# Split positive pairs into train, validation, and test sets.
train, temp = train_test_split(
    edges, test_size=0.2, random_state=42
)
validation, test = train_test_split(
    temp, test_size=0.5, random_state=42
)

train.to_parquet(OUT / "train_positive.parquet", index=False)
validation.to_parquet(OUT / "validation_positive.parquet", index=False)
test.to_parquet(OUT / "test_positive.parquet", index=False)

print("Prediction dataset prepared")
print("Train positives:", len(train))
print("Validation positives:", len(validation))
print("Test positives:", len(test))
print("Output:", OUT)