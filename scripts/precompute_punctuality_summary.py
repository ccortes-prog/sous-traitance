#!/usr/bin/env python3
# scripts/precompute_punctuality_summary.py

import sys, os

# ─────── Make project root importable ───────
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

import argparse
import pandas as pd
from indicators.punctuality_summary import compute_punctuality_summary

def main():
    parser = argparse.ArgumentParser(
        description="Precompute punctuality summary for all Category/Ligne"
    )
    parser.add_argument(
        "-i", "--csv",
        default="data/soustraitance.csv",
        help="Path to raw soustraitance CSV (semicolon-delimited)"
    )
    parser.add_argument(
        "-o", "--output",
        default="data/punctuality_summary.parquet",
        help="Path to write the precomputed Parquet"
    )
    args = parser.parse_args()

    # 1) Load raw CSV exactly like your notebook
    df = pd.read_csv(
        args.csv,
        sep=";",
        header=None,
        names=[
            "IdCourse","DateCourse","IdArret","HDepartTheo","HSortieFenetreArretReal",
            "Ligne","RangArretAsc","NumVoiture","TypeSens","SensTheo","parcveh",
            "CodeLong","Arret"
        ],
        parse_dates=["DateCourse"],
        dayfirst=True
    )
    print(f"Loaded {len(df)} rows from {args.csv}")

    # 2) Compute summary
    summary_df = compute_punctuality_summary(df)

    # 3) Write to Parquet
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    summary_df.to_parquet(args.output, index=False)
    print(f"Wrote {len(summary_df)} records to {args.output}")

if __name__ == "__main__":
    main()
