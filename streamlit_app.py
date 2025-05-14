import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.gridspec import GridSpec

# Load your data
@st.cache_data
def load_data():
    df_start = pd.read_csv("data/soustraitance_terminusstart.csv", parse_dates=['DateCourse'])
    df_end = pd.read_csv("data/soustraitance_terminusend.csv", parse_dates=['DateCourse'])
    return df_start, df_end

df_start, df_end = load_data()

# Sidebar filters
st.sidebar.title("Filters")
categories = sorted(df_start['Category'].dropna().unique())
category = st.sidebar.selectbox("Select Category", ['All'] + categories)

min_date = df_start['DateCourse'].min()
max_date = df_start['DateCourse'].max()
start_date, end_date = st.sidebar.date_input("Select Date Range", [min_date, max_date])

# Apply filters
mask = (df_start['DateCourse'] >= pd.to_datetime(start_date)) & (df_start['DateCourse'] <= pd.to_datetime(end_date))
df_start_filt = df_start.loc[mask]
df_end_filt = df_end.loc[mask]

if category != 'All':
    df_start_filt = df_start_filt[df_start_filt['Category'] == category]
    df_end_filt = df_end_filt[df_end_filt['Category'] == category]

# Build pairs
pairs = []
for (veh, date), ends in df_end_filt.groupby(['parcveh','DateCourse']):
    starts = df_start_filt.query("parcveh == @veh & DateCourse == @date")
    ends = ends.sort_values('end_real').reset_index(drop=True)
    starts = starts.sort_values('start_deptheo').reset_index(drop=True)
    for i in range(min(len(ends), len(starts)) - 1):
        prev = ends.iloc[i]
        nxt  = starts.iloc[i+1]
        if prev['stop'] == nxt['stop']:
            pairs.append({
                'Category': prev['Category'],
                'Ligne': prev['Ligne'],
                'arrival_delay_min': (pd.to_datetime(prev['end_real']) - pd.to_datetime(prev['end_deptheo'])).total_seconds()/60,
                'departure_delay_min': (pd.to_datetime(nxt['start_real']) - pd.to_datetime(nxt['start_deptheo'])).total_seconds()/60,
            })

pairs_df = pd.DataFrame(pairs)
if pairs_df.empty:
    st.warning("No matching terminus pairs found for selected filters.")
    st.stop()

pairs_df['arrival_late'] = pairs_df['arrival_delay_min'] > 3
pairs_df['departure_late'] = pairs_df['departure_delay_min'] > 3
pairs_df['both_late'] = pairs_df['arrival_late'] & pairs_df['departure_late']

# Plot
stats = pairs_df.groupby('Ligne')[['arrival_late','departure_late','both_late']].mean() * 100
ylim_max = np.ceil(stats.values.max()/10)*10

fig = plt.figure(figsize=(16,6))
gs = GridSpec(1, 3, hspace=0.4)

ax1 = fig.add_subplot(gs[0,0])
stats['arrival_late'].plot(kind='bar', ax=ax1, color='#D35400')
ax1.set_title("Arrivée Terminus Retard (%)")
ax1.set_ylim(0, ylim_max)

ax2 = fig.add_subplot(gs[0,1])
stats['departure_late'].plot(kind='bar', ax=ax2, color='#D35400')
ax2.set_title("Départ Terminus Retard (%)")
ax2.set_ylim(0, ylim_max)

ax3 = fig.add_subplot(gs[0,2])
stats['both_late'].plot(kind='bar', ax=ax3, color='#D35400')
ax3.set_title("Retard Arrivée & Départ (%)")
ax3.set_ylim(0, ylim_max)

st.pyplot(fig)
