from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def collect_csvs(root: Path):
    return [p for p in root.rglob("*.csv") if p.is_file()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    csvs = collect_csvs(root)
    rows = []
    for p in csvs:
        try:
            df = pd.read_csv(p)
            rows.append({
                "file": str(p),
                "rows": len(df),
                "cols": ",".join(df.columns[:12]),
            })
        except Exception as e:
            rows.append({
                "file": str(p),
                "rows": -1,
                "cols": f"read_failed:{e}",
            })
    out = pd.DataFrame(rows)
    out_path = root / "all_csv_index.csv"
    out.to_csv(out_path, index=False)
    print(out)

if __name__ == "__main__":
    main()
