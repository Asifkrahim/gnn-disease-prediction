from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
PRED = DATA / "prediction"
PRED.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

train = pd.read_parquet(PRED / "train_positive.parquet")
validation = pd.read_parquet(PRED / "validation_positive.parquet")
test = pd.read_parquet(PRED / "test_positive.parquet")

# Exclude every known positive pair from negative sampling.
all_edges = pd.read_parquet(DATA / "drug_disease_edges.parquet")
all_known = set(zip(all_edges["source_id"], all_edges["target_id"]))

# Estimate entity degrees using training positives only.
drug_degree = train["drug_id"].value_counts().to_dict()
disease_degree = train["disease_id"].value_counts().to_dict()

all_drugs = sorted(all_edges["source_id"].unique())
all_diseases = sorted(all_edges["target_id"].unique())

def degree_bin(degree):
    # Logarithmic bins: 0, 1, 2–3, 4–7, 8–15, etc.
    return int(np.floor(np.log2(degree + 1)))

drug_bins = {d: degree_bin(drug_degree.get(d, 0)) for d in all_drugs}
disease_bins = {d: degree_bin(disease_degree.get(d, 0)) for d in all_diseases}

drug_pool = defaultdict(list)
disease_pool = defaultdict(list)

for drug, b in drug_bins.items():
    drug_pool[b].append(drug)

for disease, b in disease_bins.items():
    disease_pool[b].append(disease)

max_radius = max(
    max(drug_bins.values(), default=0),
    max(disease_bins.values(), default=0),
)

used_negatives = set()

def candidates(pool, target_bin, radius):
    result = []
    for b in range(max(0, target_bin - radius), target_bin + radius + 1):
        result.extend(pool.get(b, []))
    return result

def sample_negative(drug, disease):
    db = drug_bins[drug]
    disb = disease_bins[disease]

    for radius in range(max_radius + 1):
        drugs = candidates(drug_pool, db, radius)
        diseases = candidates(disease_pool, disb, radius)

        if not drugs or not diseases:
            continue

        for _ in range(200):
            candidate = (
                drugs[rng.integers(len(drugs))],
                diseases[rng.integers(len(diseases))],
            )

            if candidate not in all_known and candidate not in used_negatives:
                used_negatives.add(candidate)
                return candidate

    raise RuntimeError(
        f"Could not find matched negative for ({drug}, {disease})"
    )

for name, positives in [
    ("train", train),
    ("validation", validation),
    ("test", test),
]:
    negatives = [
        sample_negative(row.drug_id, row.disease_id)
        for row in positives.itertuples(index=False)
    ]

    negative_df = pd.DataFrame(
        negatives, columns=["drug_id", "disease_id"]
    )

    positive_df = positives.copy()
    positive_df["label"] = 1
    negative_df["label"] = 0

    dataset = pd.concat(
        [positive_df, negative_df], ignore_index=True
    )
    dataset = dataset.sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    output = PRED / f"{name}_matched_labeled.parquet"
    dataset.to_parquet(output, index=False)

    print(
        f"{name}: rows={len(dataset):,}, "
        f"positives={int(dataset['label'].sum()):,}, "
        f"negatives={int((dataset['label'] == 0).sum()):,}"
    )

print("Saved degree-matched datasets:", PRED)