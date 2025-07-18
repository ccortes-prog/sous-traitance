#!/usr/bin/env python3
"""
scripts/precompute_delay_percentages.py

Precompute depot/stop delay percentages for each Category and Ligne,
writing results to a Parquet file for fast loading in Streamlit.
"""
import argparse
import pandas as pd
import numpy as np
from indicators.delay_percentages import compute_delay_percentages

# Threshold for lateness (3 minutes)
THRESHOLD_SEC = 180

# Category → list of lignes (modify as needed)
CATEGORIES = {
    "GLOBE":        ['28','47','48','51','57','59','67','69','70','71','72','73','74','78','83','91','92',
                     'A1','A2','A3','A4','A5','A6','C8'],
    "Rdta":         ['60','61'],
    "GEM’BUS":      ['80','66','68'],
    "ODIER":        ['43','47','48','70','75','L','41','52','55','C9'],
    "Genève-Tours": ['31','32','33','34','37','38','39','C3','C4','C5','C6']
}

# Reverse map for quick lookup
line_to_category = {line: cat for cat, lines in CATEGORIES.items() for line in lines}

# CSV column names
COLS = [
    'IdCourse','DateCourse','IdArret','HDepartTheo','HSortieFenetreArretReal',
    'Ligne','RangArretAsc','NumVoiture','TypeSens','SensTheo','parcveh',
    'CodeLong','Arret'
]


def main():
    parser = argparse.ArgumentParser(
        description="Precompute depot & first-stop delay percentages"
    )
    parser.add_argument(
        '-i','--csv', default='data/soustraitance.csv',
        help='Path to raw soustraitance CSV file'
    )
    parser.add_argument(
        '-o','--output', default='data/delay_percentages.parquet',
        help='Output Parquet file path'
    )
    args = parser.parse_args()

    # 1) Load raw data
    df = pd.read_csv(
        args.csv,
        sep=';', header=None,
        names=COLS,
        parse_dates=['DateCourse'],
        dayfirst=True,
        dtype={'Ligne': str, 'CodeLong': str}
    )

    # 2) Map each line to its category
    df['Category'] = df['Ligne'].map(line_to_category).fillna('Unknown')

    # 3) Parse time columns
    df['HDepartTheo']            = pd.to_datetime(df['HDepartTheo'], format='%H:%M:%S', errors='coerce')
    df['HSortieFenetreArretReal'] = pd.to_datetime(df['HSortieFenetreArretReal'], format='%H:%M:%S', errors='coerce')

    # 4) Prepare terminus DataFrames
    # find max stop index per run
    df_sorted = df.copy()
    df_sorted['max_Rang'] = df_sorted.groupby(
        ['parcveh','DateCourse','IdCourse']
    )['RangArretAsc'].transform('max')

    # end-of-run stops
    df_end = df_sorted[df_sorted['RangArretAsc'] == df_sorted['max_Rang']].copy()
    df_end = df_end.rename(columns={
        'CodeLong': 'stop',
        'HDepartTheo': 'end_deptheo',
        'HSortieFenetreArretReal': 'end_real'
    })[[
        'parcveh','DateCourse','IdCourse','stop','end_deptheo','end_real','Category','Ligne'
    ]]

    # start-of-run stops
    df_start = df_sorted[df_sorted['RangArretAsc'] == 1].copy()
    df_start = df_start.rename(columns={
        'CodeLong': 'stop',
        'HDepartTheo': 'start_deptheo',
        'HSortieFenetreArretReal': 'start_real'
    })[[
        'parcveh','DateCourse','IdCourse','stop','start_deptheo','start_real','Category','Ligne'
    ]]

    # 5) Pair each end with the next start of same vehicle+date
    pairs = []
    grouped_ends = df_end.groupby(['parcveh','DateCourse'])
    for (veh, date), ends in grouped_ends:
        starts = df_start[
            (df_start['parcveh'] == veh) &
            (df_start['DateCourse'] == date)
        ].sort_values('start_deptheo').reset_index(drop=True)
        ends = ends.sort_values('end_deptheo').reset_index(drop=True)
        # pair i-th end with (i+1)-th start if same stop
        for i in range(len(ends) - 1):
            prev = ends.iloc[i]
            nxt = starts.iloc[i+1] if i+1 < len(starts) else None
            if nxt is not None and prev['stop'] == nxt['stop']:
                arrival_delay = (prev['end_real'] - prev['end_deptheo']).total_seconds()
                depart_delay  = (nxt['start_real'] - nxt['start_deptheo']).total_seconds()
                pairs.append({
                    'Category': prev['Category'],
                    'Ligne': prev['Ligne'],
                    'dep_delay_sec': depart_delay,
                    'stop_delay_sec': arrival_delay
                })

    if not pairs:
        print("No terminus pairs found. Exiting.")
        return

    pairs_df = pd.DataFrame(pairs)

    # 6) Flag lateness > threshold
    pairs_df['dep_late']  = pairs_df['dep_delay_sec']  > THRESHOLD_SEC
    pairs_df['stop_late'] = pairs_df['stop_delay_sec'] > THRESHOLD_SEC
    pairs_df['both_late'] = pairs_df['dep_late'] & pairs_df['stop_late']

    # 7) Compute and save percentages via indicator
    result = compute_delay_percentages(pairs_df)
    result.to_parquet(args.output, index=False)
    print(f"Wrote {len(result)} records to {args.output}")

if __name__ == '__main__':
    main()
