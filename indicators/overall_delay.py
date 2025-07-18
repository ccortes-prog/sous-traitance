import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
import numpy as np
import pandas as pd


def compute_overall_delay_percentages(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute overall delay percentages per Category and Ligne, including:
      - DepotLatePct: % of trips with departure late
      - FirstStopLatePct: % with first stop late
      - AnyLatePct: % with either departure or first-stop late
    Returns DataFrame with columns:
      ['Category','Ligne','DepotLatePct','FirstStopLatePct','AnyLatePct']
    """
    df = pairs_df.copy()
    # flag any-late
    df['any_late'] = df['dep_late'] | df['stop_late']

    stats = (
        df
        .groupby(['Category', 'Ligne'])
        .agg(
            total_count     = ('any_late', 'size'),
            late_dep_count  = ('dep_late', 'sum'),
            late_stop_count = ('stop_late', 'sum'),
            late_any_count  = ('any_late', 'sum')
        )
        .reset_index()
    )

    # compute percentages
    stats['DepotLatePct']      = stats['late_dep_count']  / stats['total_count'] * 100
    stats['FirstStopLatePct']  = stats['late_stop_count']/ stats['total_count'] * 100
    stats['AnyLatePct']        = stats['late_any_count'] / stats['total_count'] * 100

    # keep only the needed columns
    return stats[['Category','Ligne','DepotLatePct','FirstStopLatePct','AnyLatePct']]


def plot_overall_delay(
    df: pd.DataFrame,
    category: str,
    logo_path: str,color: str = "#D35400",
) -> plt.Figure:
    """
    Given the DataFrame from compute_overall_delay_percentages,
    produce a 3-panel bar chart for the specified category, and
    show the date-range in the header.
    """
    # filter to this category
    df_cat = df[df['Category'] == category]
    if df_cat.empty:
        raise ValueError(f"No data for category {category}")

    stats = df_cat.set_index('Ligne')[['DepotLatePct','FirstStopLatePct','AnyLatePct']]
    ylim = np.ceil(stats.values.max() / 10) * 10

    # load logo
    logo = mpimg.imread(logo_path)

    # build figure
    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(2, 3, height_ratios=[0.15, 0.85], hspace=0.1)

    # header
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis('off')
    ax_logo = ax_header.inset_axes([0.01, 0.2, 0.1, 0.6], transform=ax_header.transAxes)
    ax_logo.imshow(logo)
    ax_logo.axis('off')

    # main title
    ax_header.text(
        0.12, 0.6,
        f"Ponctualité % {category} par ligne",
        fontsize=14, weight="bold", va="center", transform=ax_header.transAxes
    )
    # subtitle
    ax_header.text(
        0.12, 0.3,
        "Service Réseau (Gestion VCOM)",
        fontsize=12, va="center", transform=ax_header.transAxes
    )

    # three panels
    axes   = [fig.add_subplot(gs[1, i]) for i in range(3)]
    metrics = ['DepotLatePct','FirstStopLatePct','AnyLatePct']
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

