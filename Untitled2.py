#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec


# In[10]:


# --- PARAMETERS ---
CSV_PATH      = '/Users/cloe/Desktop/soustraitance.csv'
THRESHOLD_SEC = 180  # 3 minutes
HCOLOR        = "#D35400"
OUTPUT_FOLDER = '/Users/cloe/Desktop/Soustraitance/'
logo_path  = 'logo.png'

# Category → list of lignes
CATEGORIES = {
    "GLOBE":        ['28','47','48','51','57','59','67','69','70','71','72','73','74','78','83','91','92',
                     'A1','A2','A3','A4','A5','A6','C8'],
    "Rdta":         ['60','61'],
    "GEM’BUS":      ['80','66','68'],
    "ODIER":        ['43','47','48','70','75','L','41','52','55','C9'],
    "Genève-Tours": ['31','32','33','34','37','38','39','C3','C4','C5','C6']
}

PANEL_POSITIONS = {
    ('HPM','Workday'): [0.05, 0.53, 0.45, 0.4],
    ('HPS','Workday'): [0.55, 0.53, 0.45, 0.4],
    ('HPM','Weekend'): [0.05, 0.05, 0.45, 0.4],
    ('HPS','Weekend'): [0.55, 0.05, 0.45, 0.4],
}
PANEL_TITLES = {
    ('HPM','Workday'): 'HPM – Workday',
    ('HPS','Workday'): 'HPS – Workday',
    ('HPM','Weekend'): 'HPM – Weekend',
    ('HPS','Weekend'): 'HPS – Weekend',
}

# --- 1) Load & parse CSV ---
cols = ['IdCourse','DateCourse','IdArret','HDepartTheo','HSortieFenetreArretReal',
        'Ligne','RangArretAsc','NumVoiture','TypeSens', 'SensTheo','parcveh', 'CodeLong', 'Arret']


# In[11]:


# 3. Reverse mapping: line -> category
line_to_category = {line: cat for cat, lines in CATEGORIES.items() for line in lines}

# 4. Load the merged CSV (update path if needed)
df = pd.read_csv(
    '/Users/cloe/Desktop/soustraitance.csv',
    sep=';',
    header=None,        # no existing header row
    names=cols,      # assign these names
    parse_dates=['DateCourse']
)


# 5. Map each Ligne to its category
df['Category'] = df['Ligne'].astype(str).map(line_to_category).fillna('Unknown')

# Parse time columns
df['HDepartTheo']            = pd.to_datetime(df['HDepartTheo'], format='%H:%M:%S', errors='coerce')
df['HSortieFenetreArretReal'] = pd.to_datetime(df['HSortieFenetreArretReal'], format='%H:%M:%S', errors='coerce')

# Tag Workday/Weekend and peaks
df['WorkWeekend'] = np.where(df['DateCourse'].dt.weekday >= 5, 'Weekend', 'Workday')
df['Hour']        = df['HDepartTheo'].dt.hour
df['Peak']        = df['Hour'].map(lambda h: 'HPM' if 6<=h<9 else ('HPS' if 15<=h<18 else np.nan))


# In[12]:


df


# In[13]:


hcolor     = "#D35400"
# 1) Identify end‐stop (terminus) and start‐stop for each run
# 1) Identify end‐stop (terminus) and start‐stop for each run
df_sorted=df
df_sorted['max_Rang'] = df_sorted.groupby(
    ['parcveh','DateCourse','IdCourse']
)['RangArretAsc'].transform('max')
df_sorted.to_csv('soustraitance_terminus.csv')


# In[ ]:


df_sorted=pd.read_csv('soustraitance_terminus.csv')


# In[14]:


# DataFrame of run‐ends
df_end = df_sorted[df_sorted['RangArretAsc'] == df_sorted['max_Rang']].copy()
df_end = df_end.rename(columns={
    'CodeLong': 'stop',
    'HDepartTheo': 'end_deptheo',
    'HSortieFenetreArretReal': 'end_real'
})[
    ['parcveh','DateCourse','IdCourse','stop','end_deptheo','end_real','Category','Ligne']
]
df_end.to_csv('soustraitance_terminusend.csv')

# DataFrame of run‐starts
df_start = df_sorted[df_sorted['RangArretAsc'] == 1].copy()
df_start = df_start.rename(columns={
    'CodeLong': 'stop',
    'HDepartTheo': 'start_deptheo',
    'HSortieFenetreArretReal': 'start_real'
})[
    ['parcveh','DateCourse','IdCourse','stop','start_deptheo','start_real','Category','Ligne']
]
df_start.to_csv('soustraitance_terminusstart.csv')


# In[7]:


df_start=pd.read_csv('soustraitance_terminusstart.csv')
df_end=pd.read_csv('soustraitance_terminusend.csv')


# In[ ]:


# 2) Build terminus pairs: end of run_i with start of run_{i+1} for same vehicle+date
pairs = []
for (veh, date), ends in df_end.groupby(['parcveh','DateCourse']):
    starts = df_start.query("parcveh == @veh & DateCourse == @date")
    # order both by actual times
    ends = ends.sort_values('end_real').reset_index(drop=True)
    starts = starts.sort_values('start_deptheo').reset_index(drop=True)
    # for each i, pair ends.iloc[i] with starts.iloc[i+1]
    for i in range(len(ends)-1):
        prev = ends.iloc[i]
        nxt  = starts.iloc[i+1]
        # check same terminus
        if prev['stop'] == nxt['stop']:
            pairs.append({
                'Category': prev['Category'],
                'Ligne': prev['Ligne'],
                'arrival_delay_min': (prev['end_real'] - prev['end_deptheo']).total_seconds()/60,
                'departure_delay_min': (nxt['start_real'] - nxt['start_deptheo']).total_seconds()/60,
            })

pairs_df = pd.DataFrame(pairs)

# 3) Flag >3min late
pairs_df['arrival_late']   = pairs_df['arrival_delay_min']   > 3
pairs_df['departure_late'] = pairs_df['departure_delay_min'] > 3
pairs_df['both_late']      = pairs_df['arrival_late'] & pairs_df['departure_late']
logo_path = "/Users/cloe/Desktop/logo.png"
logo = mpimg.imread(logo_path)
# Compute a common y‐limit (round up to nearest 10)

for cat in tqdm(pairs_df['Category'].unique(), desc='Categories'):
    df_cat = pairs_df[pairs_df['Category'] == cat]
    stats  = df_cat.groupby('Ligne')[['arrival_late','departure_late','both_late']].mean() * 100

    # Compute common y‐limit
    ylim_max = np.ceil(stats.values.max()/10)*10

    # Build figure and grid
    fig = plt.figure(figsize=(16,6))
    gs  = GridSpec(2, 3, height_ratios=[0.15,0.85], hspace=0.1)

    # Header row spans all 3 cols
    ax_header = fig.add_subplot(gs[0,:])
    ax_header.axis('off')

    # Logo inset inside header
    ax_logo = ax_header.inset_axes([0.01,0.2,0.1,0.6], transform=ax_header.transAxes)
    ax_logo.imshow(logo)
    ax_logo.axis('off')
    ax_logo.set_aspect('equal')
    start='01.10.2025'
    end='31.03.2025'
    ax_header.text(
        0.12, 0.6,
        f"Diagramme : Ponctualité Terminus pour {cat}", 
        fontsize=14, weight="bold", va="center", transform=ax_header.transAxes
    )
    ax_header.text(
        0.12, 0.3, "Service Réseau (Gestion VCOM)",
        fontsize=12, va="center", transform=ax_header.transAxes
    )
    ax_header.text(
        0.85, 0.3, f"{start} → {end}",
        fontsize=10, ha="right", va="center", transform=ax_header.transAxes
    )

    # Three bar‐plots in second row
    ax1 = fig.add_subplot(gs[1,0])
    ax2 = fig.add_subplot(gs[1,1], sharey=ax1)
    ax3 = fig.add_subplot(gs[1,2], sharey=ax1)

    stats['arrival_late'].plot.bar(ax=ax1, color=hcolor)
    ax1.set(title=f'{cat}: % Arrivée Terminus Retard', ylim=(0,55), ylabel='Pourcentage')

    stats['departure_late'].plot.bar(ax=ax2, color=hcolor)
    ax2.set(title=f'{cat}: % Départ Terminus Retard', ylim=(0,55))

    stats['both_late'].plot.bar(ax=ax3, color=hcolor)
    ax3.set(title=f'{cat}: % Arrivée & Départ Terminus Retard', ylim=(0,55))

    plt.tight_layout()
    plt.savefig(f'terminusretard_{cat}_{start}_{end}.png', dpi=300)
    plt.show()


# In[ ]:




