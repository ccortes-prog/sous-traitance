# indicators/delay_percentages.py
import pandas as pd
import numpy as np

def compute_delay_percentages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute depot, first-stop, and both late percentages per Category and Ligne.
    Returns a DataFrame with columns ['Category', 'Ligne', 'DepotLatePct', 'FirstStopLatePct', 'BothLatePct'].
    """
    # total runs per category
    total = df.groupby('Category').size().rename('TotalRuns')

    # count late occurrences
    stats = (
        df.groupby(['Category', 'Ligne'])
          .agg(
              depot_late_count=('dep_late', 'sum'),
              stop_late_count=('stop_late', 'sum'),
              both_late_count=('both_late', 'sum')
          )
          .reset_index()
    )

    # merge total to compute percentages
    stats = stats.merge(total.reset_index(), on='Category')
    stats['DepotLatePct']      = stats['depot_late_count']   / stats['TotalRuns'] * 100
    stats['FirstStopLatePct']  = stats['stop_late_count']    / stats['TotalRuns'] * 100
    stats['BothLatePct']       = stats['both_late_count']    / stats['TotalRuns'] * 100

    return stats[['Category', 'Ligne', 'DepotLatePct', 'FirstStopLatePct', 'BothLatePct']]
