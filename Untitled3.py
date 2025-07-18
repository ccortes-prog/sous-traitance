#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from tqdm import tqdm
from matplotlib.gridspec import GridSpec


# In[ ]:


# --- PARAMETERS ---
CSV_PATH      = 'soustraitance.csv'
THRESHOLD_SEC = 180  # 3 minutes
HCOLOR        = "#D35400"
OUTPUT_FOLDER = 'soustraitance/'
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
        'Ligne','RangArretAsc','numvoit','TypeSens', 'SensTheo','IdVehicule', 'CodeLong', 'Arret']


# 3. Reverse mapping: line -> category
line_to_category = {line: cat for cat, lines in CATEGORIES.items() for line in lines}

# 4. Load the merged CSV (update path if needed)
df = pd.read_csv(
    '/Users/cloe/Documents/GitHub/sous-traitance/data/soustraitance.csv',
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


# In[ ]:


# 1) Define your filters
dates   = [pd.Timestamp('2025-01-01'), pd.Timestamp('2025-06-30')]

# 2) Filter df1_clean
df = (
    df[
      df['DateCourse'].isin(dates)
    ]
    .reset_index(drop=True)
)


# In[ ]:


coords = pd.read_csv('/Users/cloe/Desktop/project/data/stops_df.csv')
# Rename for clarity
coords = coords.rename(columns={
    'MNLP': 'CodeLong',   # code for both depots and stops
    'X':    'coord_X',    # projected X in meters
    'Y':    'coord_Y'     # projected Y in meters
})

# --- 2) Prepare two small DataFrames for merging ---
# Depot coords
depot_coords = coords[['CodeLong','coord_X','coord_Y']].rename(columns={
    'coord_X': 'depot_X',
    'coord_Y': 'depot_Y'
})
# Stop coords
stop_coords = coords[['CodeLong','coord_X','coord_Y']].rename(columns={
    'CodeLong': 'CodeLong_stop',
    'coord_X':  'stop_X',
    'coord_Y':  'stop_Y'
})



# In[ ]:


# import pandas as pd

# # 1) Group by CodeLong and collect unique Ligne values
# lines_per_codelong = (
#     first_df
#     .groupby('CodeLong')['Ligne']
#     .unique()                          # get array of unique lines
#     .reset_index()
#     .rename(columns={'Ligne':'Lines'})
# )

# # 2) (Optional) Turn those arrays into comma-separated strings
# lines_per_codelong['Lines'] = lines_per_codelong['Lines'] \
#     .apply(lambda arr: ', '.join(str(x) for x in sorted(arr)))

# # 3) Inspect
# print(lines_per_codelong.to_string(index=False))


# In[ ]:


# Determine which vehicle column to use
veh_col = 'IdVehicule' if 'IdVehicule' in df.columns else 'NumVoiture'

# — 2) Depot departures only
df_dep = df[
    (df['TypeSens'] == 2) &
    df['CodeLong'].str.startswith('DP')
].copy()
print('NEW SIZE', len(df_dep))
df_dep['dep_delay_min'] = (
    df_dep['HSortieFenetreArretReal'] - df_dep['HDepartTheo']
).dt.total_seconds() / 60
df_dep['dep_late'] = df_dep['dep_delay_min'] > 3

# — 3) Candidate first-stops: non-DP & RangArretAsc 1 or 2
stops_df = df[
    (~df['CodeLong'].str.startswith('DP')) &
    (df['RangArretAsc'].isin([1,2]))
].copy()

# — 4) Group & sort stops for fast lookup
stops_grouped = {
    key: g.sort_values('HSortieFenetreArretReal').reset_index(drop=True)
    for key, g in stops_df.groupby([veh_col,'DateCourse'])
}

# — 5) Match each departure to its first valid stop
records = []
for (veh,date), dep_grp in tqdm(df_dep.groupby([veh_col,'DateCourse']), desc='Matching stops'):
    stop_grp   = stops_grouped.get((veh,date), pd.DataFrame(columns=stops_df.columns))
    stop_times = stop_grp['HSortieFenetreArretReal'].values.astype('datetime64[ns]')

    for _, dep in dep_grp.iterrows():
        dep_time = np.datetime64(dep['HSortieFenetreArretReal'])
        idx      = np.searchsorted(stop_times, dep_time)

        rec = dep.to_dict()
        # initialize stop‐fields as missing
        rec.update({
            'HDepartTheo_stop':            pd.NaT,
            'HSortieFenetreArretReal_stop':pd.NaT,
            'CodeLong_stop':               pd.NA,
            'RangArretAsc_stop':           pd.NA,
            'TypeSens_stop':               pd.NA
        })

        # if we found a stop within 30'
        if idx < len(stop_times) and stop_times[idx] <= dep_time + np.timedelta64(30,'m'):
            s = stop_grp.iloc[idx]
            rec['HDepartTheo_stop']             = s['HDepartTheo']
            rec['HSortieFenetreArretReal_stop'] = s['HSortieFenetreArretReal']
            rec['CodeLong_stop']                = s['CodeLong']
            rec['RangArretAsc_stop']            = s['RangArretAsc']
            rec['TypeSens_stop']                = s['TypeSens']

        records.append(rec)

first_df = pd.DataFrame(records)
#####filter
# --- 3) Remove any old coordinate columns (if you ran merges earlier) ---
for col in ['depot_X','depot_Y','stop_X','stop_Y','dist_km','travel_time_min','actual_diff_min','feasible']:
    if col in first_df.columns:
        first_df.drop(columns=[col], inplace=True)

# --- 4) Merge depot coordinates on first_df.CodeLong ---
first_df = first_df.merge(depot_coords, on='CodeLong', how='left')

# --- 5) Merge stop coordinates on first_df.CodeLong_stop ---
first_df = first_df.merge(stop_coords, on='CodeLong_stop', how='left')

# Verify the merges
print("Got columns:", [c for c in ['depot_X','depot_Y','stop_X','stop_Y'] if c in first_df.columns])

# --- 6) Compute planar distance (km) between depot and stop ---
first_df['dist_km'] = np.sqrt(
    (first_df['stop_X'] - first_df['depot_X'])**2 +
    (first_df['stop_Y'] - first_df['depot_Y'])**2
) / 1000.0

# --- 7) Estimate travel time at 15 km/h (minutes) ---
first_df['travel_time_min'] = first_df['dist_km'] / 60 * 60

# --- 8) Compute actual gap between exit‐depot and arrival‐stop (minutes) ---
first_df['actual_diff_min'] = (
    first_df['HSortieFenetreArretReal_stop'] - first_df['HSortieFenetreArretReal']
).dt.total_seconds() / 60

# --- 9) Flag feasibility ---
first_df['feasible'] = first_df['travel_time_min'] <= first_df['actual_diff_min']

# --- 10) Summarize feasibility ---
total     = len(first_df)
feasible  = first_df['feasible'].sum()
infeasible= total - feasible

print(f"Total runs:     {total}")
print(f"Feasible runs:   {feasible} ({feasible/total*100:.1f}%)")
print(f"Infeasible runs: {infeasible} ({infeasible/total*100:.1f}%)\n")

# --- 11) Show a few examples of each ---
print("Sample feasible runs:")
print(first_df[first_df['feasible']].head(5)[
    ['IdCourse','Category','Ligne','dist_km','travel_time_min','actual_diff_min','CodeLong_stop','RangArretAsc_stop']
].to_string(index=False))

print("\nSample infeasible runs:")
print(first_df[~first_df['feasible']].head(5)[
    ['IdCourse','Category','Ligne','dist_km','travel_time_min','actual_diff_min','CodeLong_stop','RangArretAsc_stop']
].to_string(index=False))
######filter
#
# first_df = first_df[first_df['feasible']].copy()
# — 6) Compute stop lateness
first_df['stop_delay_min'] = (
    first_df['HSortieFenetreArretReal_stop'] - first_df['HDepartTheo_stop']
).dt.total_seconds() / 60
first_df['stop_late'] = first_df['stop_delay_min'] > 3


# — 7) Report match rates
total = len(first_df)
found = first_df['HSortieFenetreArretReal_stop'].notna().sum()
missing = total - found
print(f"Total departures: {total}, Matches: {found} ({found/total:.1%}), Missing: {missing} ({missing/total:.1%})")

# — 8) Show a few examples
print("\nMatched departures:")
print(first_df[first_df['HSortieFenetreArretReal_stop'].notna()]
      [['IdCourse','DateCourse',veh_col,'Ligne',
        'CodeLong','dep_delay_min',
        'CodeLong_stop','RangArretAsc_stop','TypeSens_stop',
        'stop_delay_min']].head(10).to_string(index=False))

print("\nUnmatched departures:")
print(first_df[first_df['HSortieFenetreArretReal_stop'].isna()]
      [['IdCourse','DateCourse',veh_col,'Ligne',
        'CodeLong','dep_delay_min']].head(10).to_string(index=False))


# In[ ]:


first_df=first_df[first_df['HSortieFenetreArretReal_stop'].notna()]
first_df


# In[ ]:


hcolor    = "#D35400"
logo_path = "/Users/cloe/Desktop/logo.png"
logo      = mpimg.imread(logo_path)
first_df['both_late'] = first_df['dep_late'] & first_df['stop_late']

for category in first_df['Category'].unique():
    df_cat    = first_df[first_df['Category'] == category]
    total_cat = len(df_cat)

    # --- Build stats_line: counts per line, then pct of total category ---
    stats_line = (
        df_cat
        .groupby('Ligne')
        .agg(
            late_dep_count      = ('dep_late',  'sum'),
            late_stop_count     = ('stop_late', 'sum'),
            late_both_count     = ('both_late','sum')
        )
        .reset_index()
    )
    stats_line['DepotLatePct']       = stats_line['late_dep_count']   / total_cat * 100
    stats_line['FirstStopLatePct']   = stats_line['late_stop_count']  / total_cat * 100
    stats_line['BothLatePct']        = stats_line['late_both_count']  / total_cat * 100

    # --- Set up figure + GridSpec for header + 3 plots ---
    fig = plt.figure(figsize=(18, 7))
    gs  = GridSpec(2, 3, height_ratios=[0.15,0.85], hspace=0.1)

    # Header
    ax_header = fig.add_subplot(gs[0,:])
    ax_header.axis('off')
    ax_logo = ax_header.inset_axes([0.01,0.2,0.1,0.6], transform=ax_header.transAxes)
    ax_logo.imshow(logo); ax_logo.axis('off'); ax_logo.set_aspect('equal')
    start = first_df['DateCourse'].min().strftime('%d.%m.%Y')
    end   = first_df['DateCourse'].max().strftime('%d.%m.%Y')
    ax_header.text(0.12,0.6,
        f"Diagramme : Ponctualité {category} – % Retards par Ligne (base = total cat.)",
        fontsize=14, weight="bold", va="center", transform=ax_header.transAxes)
    ax_header.text(0.12,0.3,
        "Service Réseau (Gestion VCOM)",
        fontsize=12, va="center", transform=ax_header.transAxes)
    ax_header.text(0.85,0.3,
        f"{start} → {end}",
        fontsize=10, ha="right", va="center", transform=ax_header.transAxes)

    # Compute a common y‐limit
    ylim_max = 30

    # Plot 1: Depot Late %
    ax1 = fig.add_subplot(gs[1,0])
    bars1 = ax1.bar(stats_line['Ligne'], stats_line['DepotLatePct'], color=hcolor)
    ax1.set_title(f'{category}: % Dépôt Retard')
    ax1.set_ylabel('Pourcentage')
    ax1.set_ylim(0, ylim_max)
    ax1.set_xticklabels(stats_line['Ligne'], rotation=45, ha='right')
    for bar, pct in zip(bars1, stats_line['DepotLatePct']):
        ax1.text(bar.get_x()+bar.get_width()/2, pct+0.1, f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

    # Plot 2: First Stop Late %
    ax2 = fig.add_subplot(gs[1,1], sharey=ax1)
    bars2 = ax2.bar(stats_line['Ligne'], stats_line['FirstStopLatePct'], color=hcolor)
    ax2.set_title(f'{category}: % 1er Arrêt Retard')
    ax2.set_ylim(0, ylim_max)
    ax2.set_xticklabels(stats_line['Ligne'], rotation=45, ha='right')
    for bar, pct in zip(bars2, stats_line['FirstStopLatePct']):
        ax2.text(bar.get_x()+bar.get_width()/2, pct+0.1, f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

    # Plot 3: Both Late %
    ax3 = fig.add_subplot(gs[1,2], sharey=ax1)
    bars3 = ax3.bar(stats_line['Ligne'], stats_line['BothLatePct'], color=hcolor)
    ax3.set_title(f'{category}: % Dépôt & 1er Arrêt Retard')
    ax3.set_ylim(0, ylim_max)
    ax3.set_xticklabels(stats_line['Ligne'], rotation=45, ha='right')
    for bar, pct in zip(bars3, stats_line['BothLatePct']):
        ax3.text(bar.get_x()+bar.get_width()/2, pct+0.1, f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'/Users/cloe/Documents/GitHub/sous-traitance/figures/bilan avril2025-juin2025/retards_cattotal_{category}_{start}_{end}.png', dpi=300)
    plt.show()


# In[ ]:


hcolor     = "#D35400"
# 1) Compute the combined "both late" flag
first_df['both_late'] = first_df['dep_late'] & first_df['stop_late']

# Assume logo is preloaded or define the path
logo_path = "/Users/cloe/Desktop/logo.png"
logo = mpimg.imread(logo_path)

# Example loop over categories
for category in first_df['Category'].unique():
    # Aggregate your stats_line as before...
    df_cat = first_df[first_df['Category'] == category]
    # inside your for‐loop, after df_cat = …
    stats_line = (
        df_cat
        .groupby('Ligne')
        .agg(
            DepotLatePct     = ('dep_late',  lambda x: x.mean()*100),
            FirstStopLatePct = ('stop_late', lambda x: x.mean()*100),
            BothLatePct      = ('both_late', lambda x: x.mean()*100),
        )
        .reset_index()
    )


    # Create figure with GridSpec: 2 rows, 3 cols; header row smaller
    fig = plt.figure(figsize=(18, 7))
    gs = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[0.15, 0.85], hspace=0.1)

    # Header spans all 3 columns
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis("off")

    # Place logo inside header
    ax_logo = ax_header.inset_axes([0.01, 0.2, 0.1, 0.6], transform=ax_header.transAxes)
    ax_logo.imshow(logo)
    ax_logo.axis("off")
    ax_logo.set_aspect('equal')

    # Header text
    start = first_df["DateCourse"].min().strftime("%d.%m.%Y")
    end   = first_df["DateCourse"].max().strftime("%d.%m.%Y")
    ax_header.text(0.12, 0.6,
        f"Diagramme : Ponctualité {category} – départ dépôt, départ 1er arrêt et dépôt + 1er arrêt",
        fontsize=14, weight="bold", va="center", transform=ax_header.transAxes
    )
    ax_header.text(0.12, 0.3,
        "Service Réseau (Gestion VCOM)",
        fontsize=12, va="center", transform=ax_header.transAxes
    )
    ax_header.text(0.85, 0.3,
        f"{start} → {end}",
        fontsize=10, ha="right", va="center", transform=ax_header.transAxes
    )

    # Main subplots
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[1, 2])

    # Plot data
    hcolor = "#D35400"
    ax1.bar(stats_line['Ligne'], stats_line['DepotLatePct'], color=hcolor)
    ax1.set_title(f'{category}: Dépôt Retard %')
    ax1.set_ylabel('Pourcentage')
    ax1.set_xticklabels(stats_line['Ligne'], rotation=45, ha='right')
    ax1.set_ylim(0, 110)

    ax2.bar(stats_line['Ligne'], stats_line['FirstStopLatePct'], color=hcolor)
    ax2.set_title(f'{category}: 1er Arrêt Retard %')
    ax2.set_xticklabels(stats_line['Ligne'], rotation=45, ha='right')
    ax2.set_ylim(0, 110)

    ax3.bar(stats_line['Ligne'], stats_line['BothLatePct'], color=hcolor)
    ax3.set_title(f'{category}: Retard Dépôt et 1er Arrêt %')
    ax3.set_xticklabels(stats_line['Ligne'], rotation=45, ha='right')
    ax3.set_ylim(0, 110)
    # Plot Depot Late %
    bars1 = ax1.bar(stats_line['Ligne'], stats_line['DepotLatePct'], color=hcolor)
    for bar in bars1:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2,  # x‐center of bar
            h + 0.5,                          # a little above the top
            f"{h:.1f}%",                      # label
            ha='center', va='bottom', fontsize=9
        )

    # Plot First Stop Late %
    bars2 = ax2.bar(stats_line['Ligne'], stats_line['FirstStopLatePct'], color=hcolor)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            h + 0.5,
            f"{h:.1f}%",
            ha='center', va='bottom', fontsize=9
        )

    # Plot Both Late %
    bars3 = ax3.bar(stats_line['Ligne'], stats_line['BothLatePct'], color=hcolor)
    for bar in bars3:
        h = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width()/2,
            h + 0.5,
            f"{h:.1f}%",
            ha='center', va='bottom', fontsize=9
        )


    # Save and show
    plt.savefig(f'/Users/cloe/Documents/GitHub/sous-traitance/figures/bilan avril2025-juin2025/retards_moyenligne_{category}_{start}_{end}.png', dpi=300)
    plt.show()


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg

# === 1) Filter for TypeSens == 0 and sort to get first stop of day per vehicle+date ===
df0 = stops_df[stops_df['TypeSens'] == 0].copy()
df0_sorted = df0.sort_values(['IdVehicule', 'DateCourse', 'HSortieFenetreArretReal'])
first_stops = df0_sorted.drop_duplicates(subset=['IdVehicule', 'DateCourse'], keep='first')

# === 2) Compute delay at first stop and flag lateness (> 3 min) ===
first_stops['stop_delay_min'] = (
    first_stops['HSortieFenetreArretReal'] - first_stops['HDepartTheo']
).dt.total_seconds() / 60
first_stops['stop_late'] = first_stops['stop_delay_min'] > 3

# === 3) Prepare logo and date range for headers ===
logo = mpimg.imread("/Users/cloe/Desktop/logo.png")
start = first_stops["DateCourse"].min().strftime("%d.%m.%Y")
end   = first_stops["DateCourse"].max().strftime("%d.%m.%Y")
hcolor = "#D35400"

# === 4) Loop per category and plot ===
for category in first_stops['Category'].unique():
    df_cat = first_stops[first_stops['Category'] == category]
    # mean % late per line
    stats_line = (
        df_cat
        .groupby('Ligne')['stop_late']
        .mean()
        .mul(100)
        .reset_index(name='FirstStopLatePct')
    )

    # set up figure with header + bar plot
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(2, 1, height_ratios=[0.15, 0.85], hspace=0.1)

    # header
    ax_header = fig.add_subplot(gs[0])
    ax_header.axis('off')
    ax_logo = ax_header.inset_axes([0.01, 0.2, 0.1, 0.6], transform=ax_header.transAxes)
    ax_logo.imshow(logo)
    ax_logo.axis('off')
    ax_logo.set_aspect('equal')
    ax_header.text(
        0.15, 0.6,
        f"Diagramme : Ponctualité premier arrêt commercial qui suit une sortie de dépôt - {category}",
        fontsize=13, weight="bold", va="center", transform=ax_header.transAxes
    )
    ax_header.text(
        0.15, 0.3,
        "Service Réseau (Gestion VCOM)",
        fontsize=12, va="center", transform=ax_header.transAxes
    )
    ax_header.text(
        0.85, 0.3,
        f"{start} → {end}",
        fontsize=10, ha="right", va="center", transform=ax_header.transAxes
    )

    # bar plot
    ax = fig.add_subplot(gs[1])
    bars = ax.bar(stats_line['Ligne'], stats_line['FirstStopLatePct'], color=hcolor)
    ax.set_ylabel('Pourcentage')
    ax.set_ylim(0, 100)
    ax.set_xticklabels(stats_line['Ligne'], rotation=45, ha='right')

    # annotate bars
    for bar, pct in zip(bars, stats_line['FirstStopLatePct']):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            pct + 0.5,
            f'{pct:.1f}%',
            ha='center',
            va='bottom',
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(f'/Users/cloe/Documents/GitHub/sous-traitance/figures/bilan avril2025-juin2025/retards_moyenligne_premierarret_{category}_{start}_{end}.png', dpi=300)
    plt.show()


# In[ ]:


# --- 1) Compute entry counts per DateCourse, Category, and Ligne ---
counts = (
    first_df
    .groupby(['Category', 'Ligne'])
    .size()                              # count rows
    .reset_index(name='entry_count')    # turn into DataFrame
    .sort_values(['Category','Ligne'])
)

# --- 2) Plot one time‐series per Category ---
for cat in counts['Category'].unique():
    # select only this category
    df_cat = counts[counts['Category'] == cat]

    # pivot so index=DateCourse, columns=Ligne, values=entry_count
    pivot = df_cat.pivot(
        index='Category',
        columns='Ligne',
        values='entry_count'
    ).fillna(0)  # fill missing dates/lines with zero

    # create figure
    fig, ax = plt.subplots(figsize=(10, 4))

    # plot each line separately
    for line in pivot.columns:
        ax.plot(
            pivot.index, 
            pivot[line], 
            marker='o', 
            label=str(line)
        )

    # cosmetics
    ax.set_title(f'Entry Counts Over Time for Category {cat}')
    ax.set_xlabel('Ligne')
    ax.set_ylabel('Number of Entries')
    ax.legend(title='Ligne', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd

# 1) Subset to the problematic categories
cats = ['C3','C4','C5','C6']
df_c = first_df[first_df['Ligne'].isin(cats)].copy()

# 2) Line‐level stats
line_stats = (
    df_c
    .groupby('Ligne')
    .agg(
        total_departures = ('dep_late',      'size'),
        late_pct         = ('dep_late',      lambda x: x.mean()*100),
        avg_dep_delay    = ('dep_delay_min', 'mean'),
        med_dep_delay    = ('dep_delay_min', 'median')
    )
    .sort_values('late_pct', ascending=False)
)
print("=== Line‐by‐Line Depot‐Late Stats for C3–C6 ===")
print(line_stats.to_string())

# 3) Bar chart: top 10 lines by late_pct
top10 = line_stats.head(10)
fig, ax = plt.subplots(figsize=(8,4))
ax.bar(top10.index.astype(str), top10['late_pct'])
ax.set_title("Top 10 Lines by Depot Late % (C3–C6)")
ax.set_xlabel("Line")
ax.set_ylabel("Depot Late %")
ax.set_ylim(0,100)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 4) Histogram of all depot delays
fig, ax = plt.subplots(figsize=(6,4))
ax.hist(df_c['dep_delay_min'], bins=50)
ax.set_title("Distribution of Depot Delays (min) for C3–C6")
ax.set_xlabel("Delay (minutes)")
ax.set_ylabel("Count")
plt.tight_layout()
plt.show()

# 5) Ten largest individual delays
print("\n=== Top 10 Largest Depot Delays (C3–C6) ===")
print(
    df_c
    .sort_values('dep_delay_min', ascending=False)
    [['IdCourse','DateCourse','Ligne','dep_delay_min']]
    .head(10)
    .to_string(index=False)
)


# In[ ]:


# 9) Identify departures with no valid stop in 30 minutes
problems_df = (
    first_df[first_df['HSortieFenetreArretReal_stop'].isna()]
    [['IdVehicule', 'DateCourse', 'IdCourse']]
    .drop_duplicates()
    .reset_index(drop=True)
)

print("Departures with no real stop within 30 minutes:")
print(problems_df)


# In[ ]:




