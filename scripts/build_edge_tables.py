#An edge table is simply a list of relationships between entities — the input representation we'll use to construct the graph


from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "opentargets"
OUT = ROOT / "data" / "processed"

OUT.mkdir(parents=True, exist_ok=True)


def read_dataset(folder: str) -> pd.DataFrame:
    files = sorted((RAW / folder).rglob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No Parquet files found in {RAW / folder}"
        )

    return pd.concat(
        [pd.read_parquet(file) for file in files],
        ignore_index=True,
    )


def make_edges(
    df: pd.DataFrame,
    source_col: str,
    source_type: str,
    relation: str,
    target_col: str,
    target_type: str,
) -> pd.DataFrame:
    edges = pd.DataFrame({
        "source_id": df[source_col].astype("string"),
        "source_type": source_type,
        "relation": relation,
        "target_id": df[target_col].astype("string"),
        "target_type": target_type,
        "evidence_source": "Open Targets clinical data",
        "clinical_stage": (
            df["maxClinicalStage"].astype("string")
            if "maxClinicalStage" in df.columns
            else pd.Series(pd.NA, index=df.index, dtype="string")
        ),
    })

    edges = edges.dropna(subset=["source_id", "target_id"])
    edges = edges[
        edges["source_id"].str.strip().ne("")
        & edges["target_id"].str.strip().ne("")
    ]

    return edges.drop_duplicates(
        subset=["source_id", "relation", "target_id"]
    ).reset_index(drop=True)


def main():
    indications = read_dataset("clinical_indication")
    targets = read_dataset("clinical_target")

    # Drug -> Disease: clinical indication records
    drug_disease = make_edges(
        indications,
        "drugId", "drug",
        "has_clinical_indication_for",
        "diseaseId", "disease",
    )

    # Drug -> Target: clinical target records
    drug_target = make_edges(
        targets,
        "drugId", "drug",
        "has_clinical_target",
        "targetId", "target",
    )

    # Preserve the disease context nested in clinical_target.
    # These are contextual records, not automatically independent
    # proof of a direct disease-target association.
    context_rows = []

    for row in targets[["drugId", "targetId", "diseases"]].itertuples(
        index=False, name=None
    ):
        drug_id, target_id, diseases = row

        if not isinstance(diseases, (list, tuple)):
            continue

        for disease in diseases:
            if not isinstance(disease, dict):
                continue

            disease_id = disease.get("diseaseId")

            if pd.isna(disease_id) if disease_id is not None else True:
                continue

            context_rows.append({
                "drug_id": str(drug_id),
                "target_id": str(target_id),
                "disease_id": str(disease_id),
                "disease_name": disease.get("diseaseFromSource"),
            })

    disease_context = pd.DataFrame(
        context_rows,
        columns=[
            "drug_id", "target_id", "disease_id", "disease_name"
        ],
    ).drop_duplicates()

    drug_disease.to_parquet(
        OUT / "drug_disease_edges.parquet", index=False
    )
    drug_target.to_parquet(
        OUT / "drug_target_edges.parquet", index=False
    )
    disease_context.to_parquet(
        OUT / "clinical_target_disease_context.parquet", index=False
    )

    print("\nEdge table generation complete")
    print(f"Drug-disease edges: {len(drug_disease):,}")
    print(f"Drug-target edges: {len(drug_target):,}")
    print(f"Clinical disease-context records: {len(disease_context):,}")
    print(f"Output directory: {OUT}")


if __name__ == "__main__":
    main()
