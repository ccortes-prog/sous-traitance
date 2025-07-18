import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg


def compute_punctuality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean punctuality metrics per Category and Ligne, expressed as percentages.
    Returns a DataFrame with columns ['Category', 'Ligne', 'arrival_late', 'departure_late', 'both_late'].
    """
    pairs_df = (
        df.groupby(['Category', 'Ligne'])
          .agg(
              arrival_late=('arrival_late', 'mean'),
              departure_late=('departure_late', 'mean'),
              both_late=('both_late', 'mean')
          )
          .reset_index()
    )
    # convert fractions to percentages
    for col in ['arrival_late', 'departure_late', 'both_late']:
        pairs_df[col] = pairs_df[col] * 100
    return pairs_df


def plot_punctuality_summary(pairs_df: pd.DataFrame, category: str, logo_path: str) -> plt.Figure:
    """
    Generate a three-panel bar chart for a given category showing:
      - Arrival Late %
      - Departure Late %
      - Both Late %
    Returns a matplotlib Figure.
    """
    df_cat = pairs_df[pairs_df['Category'] == category]
    stats = df_cat.set_index('Ligne')[['arrival_late', 'departure_late', 'both_late']]
    ylim_max = np.ceil(stats.values.max() / 10) * 10

    # load logo
    logo = mpimg.imread(logo_path)

    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(2, 3, height_ratios=[0.15, 0.85], hspace=0.1)
    color= "#D35400"
    # header
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis('off')
    ax_logo = ax_header.inset_axes([0.01, 0.2, 0.1, 0.6], transform=ax_header.transAxes)
    ax_logo.imshow(logo)
    ax_logo.axis('off')
    ax_header.text(
        0.12, 0.6,
        f"Ponctualité Terminus {category}",
        fontsize=14, weight="bold", va="center", transform=ax_header.transAxes
    )
    ax_header.text(
        0.12, 0.3,
        "Service Réseau (Gestion VCOM)",
        fontsize=12, va="center", transform=ax_header.transAxes
    )


    # bar panels
    axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    metrics = ['arrival_late', 'departure_late', 'both_late']
    titles = ['% Arrivée Terminus Retard', '% Départ Terminus Retard', '% Arrivée & Départ Terminus Retard']

    for ax, metric, title in zip(axes, metrics, titles):
        bars = ax.bar(stats.index, stats[metric], color=color)
        ax.set_title(f"{category}: {title}")
        ax.set_ylim(0, ylim_max)
        ax.set_xticklabels(stats.index, rotation=45, ha='right')
        for bar, val in zip(bars, stats[metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.5,
                f"{val:.1f}%",
                ha='center', va='bottom', fontsize=9
            )

    return fig
