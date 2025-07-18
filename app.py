import os
import streamlit as st
import pandas as pd
import io
from datetime import time
from indicators.punctuality_summary import compute_punctuality_summary, plot_punctuality_summary
from indicators.delay_percentages     import compute_delay_percentages, plot_delay_percentages
from indicators.overall_delay         import compute_overall_delay_percentages, plot_overall_delay
from indicators.first_stop_summary    import compute_first_stop_summary, plot_first_stop_summary

st.set_page_config(page_title="KPI Sous-traitance", layout="wide")

def check_password():
    """Returns True if the user entered the correct password."""
    # Pull the current password from the env
    correct_pw = os.getenv("APP_PASSWORD", "")
    # Initialize session state flag
    if "pw_ok" not in st.session_state:
        st.session_state.pw_ok = False

    if not st.session_state.pw_ok:
        # store the widget value in session_state["password"]
        pw = st.text_input("🔑 Enter password:", type="password", key="password")
        if pw:
            if pw == correct_pw:
                st.session_state.pw_ok = True
                # remove the password from state
                st.session_state.pop("password", None)
            else:
                st.error("❌ Incorrect password")
        return False
    return True

@st.cache_data
def load_pairs(path):
    df = pd.read_parquet(path)
    df['DateCourse'] = pd.to_datetime(df['DateCourse'])
    df['start_real'] = pd.to_datetime(df['start_real'])
    df['WeekdayType'] = df['DateCourse'].dt.weekday.map(
        lambda d: 'Weekend' if d>=5 else 'Workday'
    )
    def bucket(h):
        if 7 <= h < 9:      return 'HPM'
        elif 16 <= h < 19:  return 'HPS'
        else:               return 'Off Peak'
    df['Period'] = df['start_real'].dt.hour.map(bucket)
    return df

def filter_pairs(df, cats, lines, sens, wk, periods, start, end):
    mask = (
        (df['Category'].isin(cats)) &
        (df['Ligne'].isin(lines)) &
        (df['SensTheo'].isin(sens)) &
        (df['WeekdayType'].isin(wk)) &
        (df['Period'].isin(periods)) &
        (df['DateCourse'].between(start, end))
    )
    return df.loc[mask]

def main():
    st.title("Indicateurs de performance (KPI) pour la sous-traitance")

    df = load_pairs("data/pairs.parquet")

    st.sidebar.header("Filters")
    cats   = st.sidebar.multiselect("Category", options=sorted(df['Category'].unique()),
                                    default=sorted(df['Category'].unique()))
    avail_lines = sorted(df[df['Category'].isin(cats)]['Ligne'].unique())
    lines = st.sidebar.multiselect("Ligne", options=avail_lines, default=avail_lines)
    sens = st.sidebar.multiselect("SensTheo", options=sorted(df['SensTheo'].unique()),
                                  default=sorted(df['SensTheo'].unique()))
    wk = st.sidebar.multiselect("Weekday Type",
               options=['Workday','Weekend'], default=['Workday','Weekend'])
    periods = st.sidebar.multiselect("Time Period",
              options=['HPM','HPS','Off Peak'], default=['HPM','HPS','Off Peak'])
    min_d, max_d = df['DateCourse'].min().date(), df['DateCourse'].max().date()
    start, end = st.sidebar.date_input("Date Range", [min_d, max_d])

    tabs = st.tabs(["Ponctualité Terminus", "Sortie Depot et Premier Arret - Total", "Sortie Depot et Premier Arret - Par Ligne", "Retard Premier Arrêt", "Export"])
    filtered = filter_pairs(df, cats, lines, sens, wk, periods,
                             pd.to_datetime(start), pd.to_datetime(end))

    with tabs[0]:
        st.subheader("Ponctualité Terminus")
        if filtered.empty:
            st.info("No data for those filters.")
        else:
            stats = compute_punctuality_summary(filtered)
            for cat in cats:
                if cat in stats['Category'].values:
                    fig = plot_punctuality_summary(stats, cat, "logo.png")
                    st.pyplot(fig)

    with tabs[1]:
        st.subheader("Depot → Premier Arret  % (total sous-traitant)")
        if filtered.empty:
            st.info("No data for those filters.")
        else:
            delay_stats = compute_delay_percentages(filtered)
            for cat in cats:
                if cat in delay_stats['Category'].values:
                    fig = plot_delay_percentages(delay_stats, cat, "logo.png")
                    st.pyplot(fig)
    
    # ─── Tab 3: Overall “any‐late” %
    with tabs[2]:
        st.subheader("Depot → Premier Arret  % (total ligne)%")
        if filtered.empty:
            st.info("No data for those filters.")
        else:
            overall_stats = compute_overall_delay_percentages(filtered)
            for cat in cats:
                if cat in overall_stats['Category'].values:
                    fig = plot_overall_delay(overall_stats, cat, "logo.png")
                    st.pyplot(fig)
    
        # ─── Tab 3 (idx=3): Export all charts as PNGs ────────────
    
    with tabs[3]:
        st.subheader("Retard premier arrêt commercial du dépôt")
        if filtered.empty:
            st.info("No data for those filters.")
        else:  
            first_stats = compute_first_stop_summary(filtered)
            for cat in cats:
                if cat in first_stats['Category'].values:
                    fig = plot_first_stop_summary(first_stats, cat, "logo.png")
                    st.pyplot(fig)

    with tabs[4]:
        st.subheader("📥 Export Charts")
        if filtered.empty:
            st.info("No data to export for those filters.")
        else:
            for cat in cats:
                # 1) punctuality
                stats1 = compute_punctuality_summary(filtered)
                if cat in stats1['Category'].values:
                    fig1 = plot_punctuality_summary(stats1, cat, "logo.png")
                    buf1 = io.BytesIO()
                    fig1.savefig(buf1, format="png", bbox_inches="tight")
                    buf1.seek(0)
                    st.download_button(
                        label=f"Télécharger Retard Terminus — {cat}.png",
                        data=buf1,
                        file_name=f"terminusretard_{cat}.png",
                        mime="image/png"
                    )

                # 2) delay percentages
                stats2 = compute_delay_percentages(filtered)
                if cat in stats2['Category'].values:
                    fig2 = plot_delay_percentages(stats2, cat, "logo.png")
                    buf2 = io.BytesIO()
                    fig2.savefig(buf2, format="png", bbox_inches="tight")
                    buf2.seek(0)
                    st.download_button(
                        label=f"Télécharger Sortie Depot et Premier Arret - Total— {cat}.png",
                        data=buf2,
                        file_name=f"retards_cattotal_{cat}.png",
                        mime="image/png"
                    )

                # 3) overall delay
                stats3 = compute_overall_delay_percentages(filtered)
                if cat in stats3['Category'].values:
                    fig3 = plot_overall_delay(stats3, cat, "logo.png")
                    buf3 = io.BytesIO()
                    fig3.savefig(buf3, format="png", bbox_inches="tight")
                    buf3.seek(0)
                    st.download_button(
                        label=f"Télécharger Sortie Depot et Premier Arret - Par Ligne— {cat}.png",
                        data=buf3,
                        file_name=f"retards_moyenligne_{cat}.png",
                        mime="image/png"
                    )
                # 4) first stop summary
                first_stats = compute_first_stop_summary(filtered)
                if cat in first_stats['Category'].values:
                    fig4 = plot_first_stop_summary(first_stats, cat, "logo.png")
                    buf4 = io.BytesIO()
                    fig4.savefig(buf4, format="png", bbox_inches="tight")
                    buf4.seek(0)
                    st.download_button(
                        label=f"Télécharger First stop summary— {cat}.png",
                        data=buf4,
                        file_name=f"first_stop_summary_{cat}.png",
                        mime="image/png"
                    )

if __name__ == "__main__":
    # Only run the app if password check passes
    if check_password():
        main()
