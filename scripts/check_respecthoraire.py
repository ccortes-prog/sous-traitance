#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import datetime
import numpy as np

# Load the CSV file with specific parameters
df1= pd.read_csv(
    '/Users/cloe/Documents/GitHub/sous-traitance/data/Voyage-25.Prod-20241215-20251213.csv',
    sep=';',                  # semicolon delimiter
    encoding='ISO-8859-1',    # latin-1, as detected
    parse_dates=['Date -duh']
)

print(df1.dtypes)
print(df1.head())


# In[16]:


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

# 1) Build the composite key in df:
df['Numéro -Voit'] = (
    df['Ligne'].astype(str).str.strip()
    + ' - '
    + df['numvoit'].astype(str)
)
df['DateCourse'] = pd.to_datetime(df['DateCourse']).dt.normalize()
df = df[df['TypeSens'] == 0]


# In[17]:


# 1) Build composite key and normalize df:
df['Numéro -Voit'] = (
    df['Ligne'].astype(str).str.strip()
    + ' - '
    + df['numvoit'].astype(str)
)
df['Ligne'] = (
    df['Ligne']
      .astype(str)
      .str.replace(r'\.0$', '', regex=True)
)
df['DateCourse'] = pd.to_datetime(df['DateCourse']).dt.normalize()
df


# In[18]:


# 2) Clean df1 and rename the time columns
df1_clean = (
    df1
    .rename(columns={
        'Date -duh':        'DateCourse',
        'Heure début -Voy': 'sched_start',
        'Heure fin -Voy':   'sched_end',
        'Origine -Voy': 'CodeLong',
        'Type -Voy': 'Type',
        'Ligne -Voy': 'Ligne',
    })
    .assign(
        DateCourse = lambda d: pd.to_datetime(d['DateCourse']).dt.normalize(),
        **{'Numéro -Voit': lambda d: (
              d['Numéro -Voit']
               .str.replace(r'\s*-\s*',' - ',regex=True)
               .str.strip()
          )}
    )
    # keep only the columns we need
    [['DateCourse','Numéro -Voit','sched_start','sched_end','Type','CodeLong','Ligne']]
)
df1_clean['sched_start'] = pd.to_datetime(df1_clean['sched_start'], format='%H:%M:%S', errors='coerce')
df1_clean['sched_end'] = pd.to_datetime(df1_clean['sched_end'], format='%H:%M:%S', errors='coerce')
# total number of rows before
total = len(df1_clean)

# mask of rows where any of the three is NA
mask_nan = df1_clean[['DateCourse','Numéro -Voit','sched_start']].isna().any(axis=1)

# number of bad rows
n_bad = mask_nan.sum()

# percentage
pct_bad = n_bad / total * 100
print(f"{n_bad:,} rows out of {total:,} ({pct_bad:.2f}%) have a missing DateCourse, Numéro -Voit, or sched_start")

# now drop them
df1_clean_cleaned = df1_clean.dropna(subset=['DateCourse','Numéro -Voit','sched_start','Ligne']).reset_index(drop=True)
# 2) Cast everything to string
df1_clean_cleaned['Ligne'] = (
    df1_clean_cleaned['Ligne']
      .astype(str)
      .str.replace(r'\.0$', '', regex=True)
)
# confirm
print(f"After drop: {len(df1_clean_cleaned):,} rows remaining.")


# 3) Build a reverse lookup from code → category
code_to_cat = {
    code: cat
    for cat, codes in CATEGORIES.items()
    for code in codes
}

# 4) Filter and (optionally) assign each row’s category
mask = df1_clean_cleaned['Ligne'].isin(code_to_cat)
df1_clean_cleaned = df1_clean_cleaned.loc[mask].copy()
df1_clean_cleaned['Category'] = df1_clean_cleaned['Ligne'].map(code_to_cat)

# Now df1_filtered contains only the five groups you care about,
# and a new column “Category” tells you which one.
print(df1_clean_cleaned.shape)
df1_clean_cleaned[['Ligne', 'Category']].drop_duplicates().sort_values('Category')

df1_clean=df1_clean_cleaned
df1_clean


# In[19]:


import pandas as pd

# 1) Define your filters
start_date = pd.Timestamp('2025-04-01')
end_date   = pd.Timestamp('2025-06-25')
# line       = '33'
# &
    # (df['Ligne'] == line)

# 2) Filter df1_clean over the full date range
mask1 = (
    (df1_clean['DateCourse'] >= start_date) &
    (df1_clean['DateCourse'] <= end_date) 
)
df1_clean = df1_clean.loc[mask1].reset_index(drop=True)

# 3) Filter df_actual over the same range
mask2 = (
    (df['DateCourse'] >= start_date) &
    (df['DateCourse'] <= end_date) 
)
df = df.loc[mask2].reset_index(drop=True)


print(f"Length of df1_clean after filtering: {len(df1_clean)}")
print(f"Length of df after filtering: {len(df)}")


# In[20]:


# extract the unique values and sort them
unique_dates = np.sort(df1_clean['DateCourse'].unique())

print(unique_dates)


# In[21]:


start_dates = pd.Timestamp('2025-04-07')

mask2 = (
    (df['DateCourse'] == start_dates) &
    (df['CodeLong'] == 'RIVE04') 
)
a = df.loc[mask2].reset_index(drop=True)
a


# In[22]:


# extract the unique values and sort them
unique_dates2 = np.sort(df['DateCourse'].unique())

if np.array_equal(unique_dates, unique_dates2):
    print("They’re identical!")
else:
    print("They differ.")


# In[23]:


# df_actual = (
#     df
#     .sort_values(['DateCourse','IdCourse','RangArretAsc'])
#     .groupby(
#         ['DateCourse','IdCourse','Numéro -Voit'],  # <-- exact column name
#         as_index=False
#     )
#     .first()  # convert the GroupBy into a DataFrame, keeping the first row of each group
# )


# In[24]:


# Force all CodeLong entries to uppercase
df1_clean['CodeLong'] = df1_clean['CodeLong'].str.upper()


# In[25]:


# 4) Extract the actual first‐stop exit time from df:
df_actual = (
    df
    .sort_values(['DateCourse','IdCourse','RangArretAsc'])
)
df_actual =df_actual[['IdCourse','DateCourse','HSortieFenetreArretReal','RangArretAsc','CodeLong','Numéro -Voit','TypeSens','Ligne','HDepartTheo']]
# # Boolean mask for rangarretasc == 1
# mask_r1 = df_actual['RangArretAsc'] == 1

# # Apply it and reset the index
# df_actual = df_actual[mask_r1].reset_index(drop=True)
# this will remove everything that isn’t 0–9, leaving just the digits
# df_actual['Numéro -Voit'] = (
#     df_actual['Numéro -Voit']
#       .astype(str)
#       .str.replace(
#           r'(\d+)[A-Za-z]*(\s*-\s*)(\d+)',  # capture digits, then the “ - ” (with any spacing), then digits
#           r'\1\2\3',                       # rebuild as: group1 + original spaces-hyphen-spaces + group3
#           regex=True
#       )
# )
# df1_clean['Numéro -Voit'] = (
#     df1_clean['Numéro -Voit']
#       .astype(str)
#       .str.replace(
#           r'(\d+)[A-Za-z]*(\s*-\s*)(\d+)',  # capture digits, then the “ - ” (with any spacing), then digits
#           r'\1\2\3',                       # rebuild as: group1 + original spaces-hyphen-spaces + group3
#           regex=True
#       )
# )

# Boolean mask for rangarretasc == 1
mask_r1 = df_actual['RangArretAsc'] == 1

# Apply it and reset the index
df_actual = df_actual[mask_r1].reset_index(drop=True)
# 1) Total rows before
total_actual = len(df_actual)

# 2) Boolean mask: True for rows with ANY NaN in any column
mask_any_nan = df_actual.isna().any(axis=1)

# 3) Count & percentage of “bad” rows
n_bad_actual = mask_any_nan.sum()
pct_bad_actual = n_bad_actual / total_actual * 100
print(f"{n_bad_actual:,} rows out of {total_actual:,} "
      f"({pct_bad_actual:.2f}%) have at least one NaN in df_actual")

# 4) Drop them (dropna defaults to how='any')
df_actual_cleaned = df_actual.dropna().reset_index(drop=True)

# 5) Confirm
print(f"After drop: {len(df_actual_cleaned):,} rows remaining.")
df_actual


# In[26]:


import pandas as pd
from tqdm import tqdm  # make sure you have tqdm installed!

# 1) prepare your two tables
df1 = df1_clean.copy()
df1['sched_start_ts'] = pd.to_datetime(
    df1['DateCourse'].dt.strftime('%Y-%m-%d') + ' ' +
    df1['sched_start'].dt.time.astype(str),
    errors='coerce'
).dropna()

df2 = df_actual.copy()
df2['actual_exit_ts'] = pd.to_datetime(
    df2['DateCourse'].dt.strftime('%Y-%m-%d') + ' ' +
    df2['HDepartTheo'].astype(str),
    errors='coerce'
).dropna()

# Drop the timezone so actual_exit_ts is naive
df2['actual_exit_ts'] = df2['actual_exit_ts'].dt.tz_localize(None)

# Prepare grouping
group_cols = ['DateCourse','Ligne','CodeLong']
grouped = df1.groupby(group_cols)
n_groups = grouped.ngroups
print("Unique groups:", n_groups)
print("Example groups:", list(grouped.groups.keys())[:5])


import pandas as pd
from tqdm import tqdm  # optional, for a progress bar

matches = []
unmatched_sched = []

# grouped is your GroupBy on (date, ligne, code)
for key_vals, sched_grp in tqdm(grouped, total=len(grouped), desc="Groups"):
    date, ligne, code = key_vals
    # 1) Slice out the actuals and schedule for this group
    actual_grp = df2[
        (df2['DateCourse'] == date) &
        (df2['Ligne']      == ligne) &
        (df2['CodeLong']   == code)
    ].sort_values('actual_exit_ts').drop_duplicates('actual_exit_ts').reset_index(drop=True)

    if actual_grp.empty:
        # everything in sched_grp is unmatched
        unmatched_sched.append(sched_grp.copy())
        continue

    sched_grp = sched_grp.sort_values('sched_start_ts').reset_index(drop=True)
    # Remember original row-index so we can find unmatched later
    sched_grp = sched_grp.reset_index().rename(columns={'index':'sched_idx'})

    # 2) Merge-asof
    merged = pd.merge_asof(
        sched_grp,
        actual_grp[['actual_exit_ts','Numéro -Voit','TypeSens']],
        left_on   = 'sched_start_ts',
        right_on  = 'actual_exit_ts',
        tolerance = pd.Timedelta(minutes=5),
        direction = 'nearest'
    )

    # 3) Those with a match will have non-null actual_exit_ts
    matched = merged.dropna(subset=['actual_exit_ts'])
    matches.append(matched)

    # 4) Anything in sched_grp whose sched_idx is NOT in matched['sched_idx'] is unmatched
    unmatched = sched_grp[~sched_grp['sched_idx'].isin(matched['sched_idx'])]
    unmatched_sched.append(unmatched)

# 5) Concatenate results
df_matches   = pd.concat(matches, ignore_index=True)
df_unmatched = pd.concat(unmatched_sched, ignore_index=True)

print(f"Matched  : {len(df_matches)} schedule rows within ±5 minutes")
print(f"Unmatched: {len(df_unmatched)} schedule rows with no nearby actual")

# 6) Save both
df_matches.to_parquet('../data/inter_speed_matched.parquet',   index=False)
df_unmatched.to_parquet('../data/inter_speed_unmatched.parquet', index=False)
