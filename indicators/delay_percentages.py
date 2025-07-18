import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

def plot_delay_percentages(df: pd.DataFrame,
    category: str,
    logo_path: str, color: str = "#D35400",
) -> plt.Figure:
    """
    Given the delay‐percentages DataFrame (with columns
    ['Category','Ligne','DepotLatePct','FirstStopLatePct','BothLatePct']),
    produce a 3‐panel bar chart for the specified category.
    """
    # Filter to this category
    df_cat = df[df['Category'] == category]
    if df_cat.empty:
        raise ValueError(f"No data for category {category}")

    stats = df_cat.set_index('Ligne')[['DepotLatePct','FirstStopLatePct','BothLatePct']]
    ylim = np.ceil(stats.values.max() / 10) * 10

    # Load logo
    logo = mpimg.imread(logo_path)

    # Build figure
    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(2, 3, height_ratios=[0.15, 0.85], hspace=0.1)

    # Header row
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis('off')
    ax_logo = ax_header.inset_axes([0.01, 0.2, 0.1, 0.6], transform=ax_header.transAxes)
    ax_logo.imshow(logo)
    ax_logo.axis('off')
    ax_header.text(
        0.12, 0.6,
        f"Ponctualité % {category}",
        fontsize=14, weight="bold", va="center", transform=ax_header.transAxes
    )
    # subtitle
    ax_header.text(
        0.12, 0.3,
        "Service Réseau (Gestion VCOM)",
        fontsize=12, va="center", transform=ax_header.transAxes
    )

    # Three panels
    axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    metrics = ['DepotLatePct','FirstStopLatePct','BothLatePct']
    titles  = ['% Dépôt Retard', '% 1er Arrêt Retard', '% Dépôt & 1er Arrêt Retard']

    for ax, metric, title in zip(axes, metrics, titles):
        bars = ax.bar(stats.index, stats[metric], color=color)
        ax.set_title(f"{category}: {title}")
        ax.set_ylim(0, ylim)
        ax.set_xticklabels(stats.index, rotation=45, ha='right')
        for bar, val in zip(bars, stats[metric]):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                val + 0.5,
                f"{val:.1f}%",
                ha='center', va='bottom', fontsize=9
            )

    return fig


import pandas as pd

def compute_delay_percentages(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque Category+Ligne :
      - compte late_dep_count, late_stop_count, late_both_count
      - divise chacun par le total trips de la Category
    Renvoie ['Category','Ligne','DepotLatePct','FirstStopLatePct','BothLatePct'].
    """
    df = pairs_df.copy()

    # 1) late counts per Category+Ligne
    line_stats = (
        df
        .groupby(['Category', 'Ligne'])
        .agg(
            late_dep_count   = ('dep_late',  'sum'),
            late_stop_count  = ('stop_late', 'sum'),
            late_both_count  = ('both_late','sum'),
        )
        .reset_index()
    )

    # 2) total trips per Category
    cat_totals = (
        df
        .groupby('Category')
        .size()
        .rename('cat_total')
        .reset_index()
    )

    # 3) merge and compute percentages
    stats = pd.merge(line_stats, cat_totals, on='Category')
    stats['DepotLatePct']     = stats['late_dep_count']  / stats['cat_total'] * 100
    stats['FirstStopLatePct'] = stats['late_stop_count'] / stats['cat_total'] * 100
    stats['BothLatePct']      = stats['late_both_count'] / stats['cat_total'] * 100

    return stats[['Category','Ligne','DepotLatePct','FirstStopLatePct','BothLatePct']]
