from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
PRED = DATA / "prediction"

rng = np.random.default_rng(42)

train = pd.read_parquet(PRED / "train_positive.parquet")
validation = pd.read_parquet(PRED / "validation_positive.parquet")
test = pd.read_parquet(PRED / "test_positive.parquet")
all_known = pd.read_parquet(DATA / "drug_disease_edges.parquet")

all_known = set(zip(all_known["source_id"], all_known["target_id"]))

drugs = sorted(set(train["drug_id"]))
diseases = sorted(set(train["disease_id"]))

def sample_negatives(positive_df, n, seed):
    local_rng = np.random.default_rng(seed)
    positives = set(zip(positive_df["drug_id"], positive_df["disease_id"]))
    negatives = set()
    max_attempts = max(n * 20, 1000)
    attempts = 0

    while len(negatives) < n and attempts < max_attempts:
        drug = drugs[local_rng.integers(len(drugs))]
        disease = diseases[local_rng.integers(len(diseases))]
        pair = (drug, disease)

        if pair not in all_known and pair not in positives:
            negatives.add(pair)

        attempts += 1

    if len(negatives) < n:
        raise RuntimeError(
            f"Could only sample {len(negatives)} of {n} negatives"
        )

    return pd.DataFrame(negatives, columns=["drug_id", "disease_id"])

for name, positives, seed in [
    ("train", train, 101),
    ("validation", validation, 202),
    ("test", test, 303),
]:
    negatives = sample_negatives(positives, len(positives), seed)
    positives = positives.copy()
    positives["label"] = 1
    negatives["label"] = 0

    dataset = pd.concat([positives, negatives], ignore_index=True)
    dataset = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)
    dataset.to_parquet(PRED / f"{name}_labeled.parquet", index=False)

    print(name, "rows:", len(dataset),
          "positive:", int(dataset["label"].sum()),
          "negative:", int((dataset["label"] == 0).sum()))

print("Saved labeled datasets to:", PRED)