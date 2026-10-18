#inspect identifier overlap and graph size

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

dd = pd.read_parquet(DATA / "drug_disease_edges.parquet")
dt = pd.read_parquet(DATA / "drug_target_edges.parquet")
ctx = pd.read_parquet(DATA / "clinical_target_disease_context.parquet")

drugs_dd = set(dd["source_id"])
diseases_dd = set(dd["target_id"])
drugs_dt = set(dt["source_id"])
targets_dt = set(dt["target_id"])

print("Graph summary")
print("Drug-disease edges:", len(dd))
print("Drug-target edges:", len(dt))
print("Disease-context records:", len(ctx))
print("Unique drugs:", len(drugs_dd | drugs_dt))
print("Unique diseases:", len(diseases_dd | set(ctx["disease_id"])))
print("Unique targets:", len(targets_dt))
print("Drugs shared across tables:", len(drugs_dd & drugs_dt))
print("Diseases in context also in drug-disease table:",
      len(set(ctx["disease_id"]) & diseases_dd))

print("\nClinical stages:")
print(dd["clinical_stage"].value_counts(dropna=False).to_string())

print("\nSample drug-disease edges:")
print(dd[["source_id", "target_id", "clinical_stage"]].head().to_string(index=False))