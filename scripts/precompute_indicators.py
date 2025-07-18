#!/usr/bin/env python3
"""
scripts/precompute_indicators.py

Load raw soustraitance CSV, build terminus departure–arrival pairs with progress,
then precompute two indicators:x
  1) Punctuality summary (arrival/departure/both late %)
  2) Depot/first-stop delay percentages
Save each to a Parquet for Streamlit.
"""
import sys
import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

# ─────── Ensure project root is importable ───────
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from indicators.punctuality_summary import compute_punctuality_summary
from indicators.delay_percentages import compute_delay_percentages
from indicators.overall_delay import compute_overall_delay_percentages
from indicators.first_stop_summary import compute_first_stop_summary

# Late threshold (seconds)
THRESHOLD_SEC = 180          # 3 minutes
# Pairing window (seconds) for next start
PAIRING_WINDOW_SEC = 30 * 60 # 30 minutes

# Category → list of lignes
CATEGORIES = {
    "GLOBE":        ['28','47','48','51','57','59','67','69','70','71','72','73','74','78','83','91','92',
                         'A1','A2','A3','A4','A5','A6','C8'],
    "Rdta":         ['60','61'],
    "GEM’BUS":      ['80','66','68'],
    "ODIER":        ['43','47','48','70','75','L','41','52','55','C9'],
    "Genève-Tours": ['31','32','33','34','37','38','39','C3','C4','C5','C6']
}
line_to_category = {line:cat for cat, lines in CATEGORIES.items() for line in lines}

# Raw CSV columns (semicolon-delimited, no header)
CSV_COLUMNS = [
    'IdCourse','DateCourse','IdArret','HDepartTheo','HSortieFenetreArretReal',
    'Ligne','RangArretAsc','NumVoiture','TypeSens','SensTheo','parcveh',
    'CodeLong','Arret'
]

def main(raw_csv: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load raw CSV and parse
    df = pd.read_csv(
        raw_csv,
        sep=';', header=None,
        names=CSV_COLUMNS,
        dtype={'Ligne': str, 'CodeLong': str},
        parse_dates=['DateCourse']
    )
    df['HDepartTheo'] = pd.to_datetime(df['HDepartTheo'], format='%H:%M:%S', errors='coerce')
    df['HSortieFenetreArretReal'] = pd.to_datetime(
        df['HSortieFenetreArretReal'], format='%H:%M:%S', errors='coerce')
    print(f"Loaded {len(df):,} rows; DateCourse dtype: {df['DateCourse'].dtype}")

    # 2) Map lines to categories
    df['Category'] = df['Ligne'].map(line_to_category).fillna('Unknown')

    # 3) Identify terminus ends and starts
    df['RangArretAsc'] = pd.to_numeric(df['RangArretAsc'], errors='coerce')
    df['WorkWeekend'] = np.where(df['DateCourse'].dt.weekday >= 5, 'Weekend', 'Workday')
    def bucket(hr):
        if 7 <= hr < 9:      return 'HPM'
        elif 16 <= hr < 19:  return 'HPS'
        else:                return 'Off Peak'
    df['Peak'] = df['HDepartTheo'].dt.hour.map(bucket)

    df_sorted = df.copy()
    df_sorted['max_Rang'] = df_sorted.groupby(
        ['parcveh','DateCourse','IdCourse']
    )['RangArretAsc'].transform('max')

    df_end = df_sorted[df_sorted['RangArretAsc'] == df_sorted['max_Rang']].copy()
    df_end.rename(columns={
        'HDepartTheo':'end_deptheo',
        'HSortieFenetreArretReal':'end_real'
    }, inplace=True)
    df_end = df_end[[
        'parcveh','DateCourse','IdCourse','Category','Ligne',
        'end_deptheo','end_real','CodeLong'
    ]]
    df_end = df_end.dropna(subset=['end_deptheo','end_real'])

    df_start = df_sorted[df_sorted['RangArretAsc'] == 1].copy()
    df_start.rename(columns={
        'HDepartTheo':'start_deptheo',
        'HSortieFenetreArretReal':'start_real'
    }, inplace=True)
    df_start = df_start[[
        'parcveh','DateCourse','IdCourse',
        'start_deptheo','start_real','CodeLong',
        'SensTheo','WorkWeekend','Peak'
    ]]
    df_start = df_start.dropna(subset=['start_deptheo','start_real'])

    # 4') LOOP + searchsorted pairing (one-to-one within 30')
    records = []
    gb = df_end.groupby(['parcveh','DateCourse'])
    total_groups = gb.ngroups
    # iterate with progress bar
    for (veh, date), end_grp in tqdm(gb, desc="Pairing runs", total=total_groups):
        start_grp = df_start[(df_start['parcveh'] == veh) & (df_start['DateCourse'] == date)]
        if start_grp.empty:
            continue

        start_grp = start_grp.sort_values('start_deptheo').reset_index(drop=True)
        start_times = start_grp['start_deptheo'].values.astype('datetime64[ns]')
        for _, end_row in end_grp.iterrows():
            end_time = np.datetime64(end_row['end_deptheo'])
            idx = np.searchsorted(start_times, end_time, side='right')
            if idx < len(start_times):
                delta_s = (start_times[idx] - end_time) / np.timedelta64(1, 's')
                if delta_s <= PAIRING_WINDOW_SEC:
                    s = start_grp.iloc[idx]
                    records.append({
                        'parcveh':      veh,
                        'DateCourse':   date,
                        'IdCourse':     end_row['IdCourse'],
                        'Category':     end_row['Category'],
                        'Ligne':        end_row['Ligne'],
                        'SensTheo':     s['SensTheo'],
                        'Peak':         s['Peak'],
                        'WorkWeekend':  s['WorkWeekend'],
                        'end_deptheo':  end_row['end_deptheo'],
                        'end_real':     end_row['end_real'],
                        'start_deptheo':s['start_deptheo'],
                        'start_real':   s['start_real'],
                        'dep_delay_sec':  (s['start_real'] - s['start_deptheo']).total_seconds(),
                        'stop_delay_sec': (end_row['end_real'] - end_row['end_deptheo']).total_seconds(),
                    })

    next_pairs = pd.DataFrame.from_records(records)

    # 5) Build the final pairs_df
    pairs_df = next_pairs[[
        'parcveh','DateCourse','IdCourse','Category','Ligne',
        'SensTheo','Peak','WorkWeekend',
        'start_deptheo','start_real','end_deptheo','end_real',
        'dep_delay_sec','stop_delay_sec'
    ]].copy()

    # 6) Flag lateness
    pairs_df['dep_late']  = pairs_df['dep_delay_sec']  > THRESHOLD_SEC
    pairs_df['stop_late'] = pairs_df['stop_delay_sec'] > THRESHOLD_SEC
    # add arrival/departure synonyms for punctuality_summary
    pairs_df['departure_late'] = pairs_df['dep_late']
    pairs_df['arrival_late']   = pairs_df['stop_late']
    pairs_df['both_late']      = pairs_df['dep_late'] & pairs_df['stop_late']

    # 7) Write out parquet and compute summaries
    os.makedirs(out_dir, exist_ok=True)
    pairs_out = os.path.join(out_dir, 'pairs.parquet')
    pairs_df.to_parquet(pairs_out, index=False)
    print(f"Wrote raw pairs to {pairs_out} ({len(pairs_df):,} records)")

    punctuality_df = compute_punctuality_summary(pairs_df)
    punctuality_out = os.path.join(out_dir, 'punctuality_summary.parquet')
    punctuality_df.to_parquet(punctuality_out, index=False)
    print(f"Wrote punctuality summary to {punctuality_out}")

    delay_pct_df = compute_delay_percentages(pairs_df)
    delay_out = os.path.join(out_dir, 'delay_percentages.parquet')
    delay_pct_df.to_parquet(delay_out, index=False)
    print(f"Wrote delay percentages to {delay_out}")
    # 8) Compute overall “any‐late” percentage and save
    

    overall_df = compute_overall_delay_percentages(pairs_df)
    overall_out = os.path.join(out_dir, 'overall_delay_percentages.parquet')
    overall_df.to_parquet(overall_out, index=False)
    print(f"Wrote overall delay percentages to {overall_out}")

    # 9) Compute premier arrêt commercial par véhicule/jour et save
    first_stop_df = compute_first_stop_summary(pairs_df)
    first_stop_out = os.path.join(out_dir, 'first_stop_summary.parquet')
    first_stop_df.to_parquet(first_stop_out, index=False)
    print(f"Wrote first-stop-of-day summary to {first_stop_out}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Precompute all indicators")
    parser.add_argument('-i','--input', required=True, help="Path to raw CSV")
    parser.add_argument('-o','--output-dir', default='data', help="Output directory")
    args = parser.parse_args()
    main(args.input, args.output_dir)

