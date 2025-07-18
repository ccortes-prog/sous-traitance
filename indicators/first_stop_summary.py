import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
import numpy as np

def compute_first_stop_summary(pairs_df: pd.DataFrame, threshold_sec: int = 180) -> pd.DataFrame:
    """
    Pour chaque véhicule et jour, sélectionne le premier trajet (le premier arrêt commercial),
    calcule le retard à ce premier arrêt et renvoie le pourcentage de retards
    par catégorie et par ligne.

    Args:
        pairs_df: DataFrame issu de precompute, contenant au moins:
            ['parcveh','DateCourse','start_deptheo','start_real','Category','Ligne']
        threshold_sec: seuil de retard en secondes (par défaut 180s = 3min)

    Returns:
        DataFrame avec colonnes ['Category','Ligne','FirstStopOfDayLatePct']
    """
    # 1) Trouver, pour chaque véhicule et chaque date, l'index du premier trajet
    idx = pairs_df.groupby(['parcveh', 'DateCourse'])['start_deptheo'].idxmin()
    first_df = pairs_df.loc[idx].copy()

    # 2) Calculer le retard au premier arrêt (en secondes)
    first_df['stop_delay_sec'] = (
        first_df['start_real'] - first_df['start_deptheo']
    ).dt.total_seconds()
    first_df['stop_late'] = first_df['stop_delay_sec'] > threshold_sec

    # 3) Moyenne des retards par Catégorie et Ligne
    stats = (
        first_df
        .groupby(['Category', 'Ligne'])['stop_late']
        .mean()
        .mul(100)
        .reset_index(name='FirstStopOfDayLatePct')
    )
    return stats


def plot_first_stop_summary(
    df: pd.DataFrame,
    category: str,
    logo_path: str,
    color: str = "#D35400"
) -> plt.Figure:
    """
    Produit un graphique en deux lignes (en-tête + barre) pour le retard
    du premier arrêt commercial du dépôt, par ligne, pour une catégorie donnée.
    Affiche la période et la mention Service Réseau.

    Args:
        df: DataFrame renvoyé par compute_first_stop_summary
        category: nom de la catégorie à tracer
        logo_path: chemin vers le logo à afficher en en-tête
        start_date, end_date: bornes temporelles à afficher
        color: couleur des barres (par défaut orange)

    Returns:
        Matplotlib Figure
    """
    df_cat = df[df['Category'] == category]
    if df_cat.empty:
        raise ValueError(f"Aucune donnée pour la catégorie {category}")

    stats = df_cat.set_index('Ligne')['FirstStopOfDayLatePct']
    ylim = np.ceil(stats.max() / 10) * 10

    # Chargement du logo
    logo = mpimg.imread(logo_path)

    # Création de la figure
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(2, 1, height_ratios=[0.15, 0.85], hspace=0.1)

    # En-tête
    ax_header = fig.add_subplot(gs[0, 0])
    ax_header.axis('off')
    ax_logo = ax_header.inset_axes([0.01, 0.2, 0.1, 0.6], transform=ax_header.transAxes)
    ax_logo.imshow(logo)
    ax_logo.axis('off')
    # Titre principal
    ax_header.text(
        0.15, 0.6,
        f"Diagramme : Retard premier arrêt commercial — Catégorie {category}",
        fontsize=13, weight="bold", va="center", transform=ax_header.transAxes
    )
    # Mention Service Réseau
    ax_header.text(
        0.15, 0.3,
        "Service Réseau (Gestion VCOM)",
        fontsize=12, va="center", transform=ax_header.transAxes
    )

    # Barres
    ax = fig.add_subplot(gs[1, 0])
    bars = ax.bar(stats.index, stats, color=color)
    ax.set_ylabel("% Retard")
    ax.set_ylim(0, ylim)
    ax.set_xticklabels(stats.index, rotation=45, ha='right')
    for bar, val in zip(bars, stats):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.5,
            f"{val:.1f}%",
            ha='center', va='bottom', fontsize=9
        )

    return fig
