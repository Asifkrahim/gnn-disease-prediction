#This script will inspect both downloaded datasets, report their schemas, row counts, missing values, and unique entity IDs. It won't modify the source data.

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "opentargets"

DATASETS = {
    "clinical_indication": {
        "folder": "clinical_indication",
        "ids": ["drugId", "diseaseId"],
    },
    "clinical_target": {
        "folder": "clinical_target",
        "ids": ["drugId", "targetId"],
    },
}


def find_parquet_files(folder: Path) -> list[Path]:
    return sorted(folder.rglob("*.parquet"))


def main():
    report_dir = PROJECT_ROOT / "docs"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_lines = []

    for name, config in DATASETS.items():
        folder = DATA_DIR / config["folder"]
        files = find_parquet_files(folder)

        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")

        if not files:
            print(f"No Parquet files found in: {folder}")
            report_lines.append(f"## {name}\nNo Parquet files found: {folder}")
            continue

        total_rows = 0
        total_missing = {}
        unique_ids = {col: set() for col in config["ids"]}
        columns = None

        for file in files:
            df = pd.read_parquet(file)
            total_rows += len(df)
            columns = list(df.columns)

            for col in config["ids"]:
                if col in df.columns:
                    total_missing[col] = (
                        total_missing.get(col, 0) + int(df[col].isna().sum())
                    )
                    unique_ids[col].update(
                        df[col].dropna().astype(str).unique()
                    )

        print(f"Files: {len(files)}")
        print(f"Rows: {total_rows:,}")
        print(f"Columns: {columns}")

        section = [
            f"## {name}",
            f"- Files: {len(files)}",
            f"- Rows: {total_rows:,}",
            f"- Columns: {columns}",
        ]

        for col in config["ids"]:
            if col in (columns or []):
                unique_count = len(unique_ids[col])
                missing_count = total_missing.get(col, 0)
                print(
                    f"{col}: {unique_count:,} unique non-null IDs; "
                    f"{missing_count:,} missing values"
                )
                section.append(
                    f"- {col}: {unique_count:,} unique non-null IDs; "
                    f"{missing_count:,} missing values"
                )
            else:
                print(f"WARNING: Expected column '{col}' not found")
                section.append(f"- WARNING: Expected column '{col}' not found")

        print("\nSample rows:")
        sample = pd.read_parquet(files[0]).head(3)
        print(sample.to_string(index=False))

        section.append("\nSample rows:\n```text\n")
        section.append(sample.to_string(index=False))
        section.append("\n```")
        report_lines.append("\n".join(section))

    report_path = report_dir / "data_audit.md"
    report_path.write_text("\n\n".join(report_lines), encoding="utf-8")
    print(f"\nAudit report saved to: {report_path}")


if __name__ == "__main__":
    main()
