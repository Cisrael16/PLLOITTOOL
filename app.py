# PLL OIT Dashboard – v7.0.0 (UI Redesign + Career Trajectories)
# Redesign focuses on: Clean UI, Better UX, Highlighted scatter selections, Career tracking

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, Optional

# Configuration for available years and their file paths
YEARS_CONFIG = {
    2022: {
        "touchrate": "PLL_2022_Touch_Rate.csv",
        "stats": "2022_pll-player-stats.csv"
    },
    2023: {
        "touchrate": "PLL_2023_Touch_Rate.csv",
        "stats": "2023_pll-player-stats.csv"
    },
    2024: {
        "touchrate": "PLL_2024_Touch_Rate.csv",
        "stats": "2024_pll-player-stats.csv"
    },
    2025: {
        "touchrate": "PLL_2025_Touch_Rate.csv",
        "stats": "2025_pll-player-stats.csv"
    }
}

# Role color scheme
ROLE_COLORS = {
    "Shooter": "#FF6B6B",
    "Facilitator": "#4ECDC4",
    "Efficient Finisher": "#95E1D3",
    "Ball-Dominant Creator": "#F38181",
    "Risky Creator": "#FFA07A",
    "Other": "#A8A8A8",
}

st.set_page_config(page_title="PLL OIT Dashboard", layout="wide", initial_sidebar_state="expanded")

# Dark mode toggle in sidebar (at the very top)
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Custom CSS with dark mode support
def get_custom_css(dark_mode):
    if dark_mode:
        return """
        <style>
            /* Dark mode colors - proper contrast */
            :root {
                --bg-primary: #0e1117;
                --bg-secondary: #262730;
                --bg-tertiary: #1a1d24;
                --text-primary: #fafafa;
                --text-secondary: #a3a8b8;
                --border-color: #3d4148;
                --accent-color: #4ECDC4;
            }
            
            /* Force light text on dark backgrounds */
            .main {
                background-color: var(--bg-primary) !important;
                color: var(--text-primary) !important;
            }
            
            /* Sidebar */
            [data-testid="stSidebar"] {
                background-color: var(--bg-tertiary) !important;
            }
            
            [data-testid="stSidebar"] * {
                color: var(--text-primary) !important;
            }
            
            /* Content padding */
            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            
            /* Headers - force white */
            h1, h2, h3, h4, h5, h6 {
                color: var(--text-primary) !important;
                font-weight: 500;
            }
            
            h1 {
                font-weight: 600 !important;
                margin-bottom: 1rem;
            }
            
            h2, h3 {
                margin-top: 1.5rem;
                margin-bottom: 0.75rem;
            }
            
            /* Force all paragraph text to be visible */
            p, span, div {
                color: var(--text-primary) !important;
            }
            
            /* Dividers */
            hr {
                margin: 2rem 0;
                border: none;
                border-top: 1px solid var(--border-color) !important;
            }
            
            /* Metrics - CRITICAL FIX */
            [data-testid="stMetricValue"] {
                font-size: 1.5rem;
                color: var(--text-primary) !important;
            }
            
            [data-testid="stMetricLabel"] {
                color: var(--text-secondary) !important;
            }
            
            [data-testid="stMetricDelta"] {
                color: var(--text-secondary) !important;
            }
            
            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {
                gap: 2rem;
                background-color: var(--bg-secondary);
                border-bottom: 1px solid var(--border-color);
            }
            
            .stTabs [data-baseweb="tab"] {
                padding: 0.5rem 1rem;
                font-weight: 500;
                color: var(--text-secondary) !important;
                background-color: transparent;
            }
            
            .stTabs [data-baseweb="tab"][aria-selected="true"] {
                color: var(--accent-color) !important;
                border-bottom: 2px solid var(--accent-color);
            }
            
            /* Dataframes */
            [data-testid="stDataFrame"] {
                background-color: var(--bg-secondary) !important;
            }
            
            /* Table headers */
            [data-testid="stDataFrame"] th {
                background-color: var(--bg-tertiary) !important;
                color: var(--text-primary) !important;
            }
            
            /* Table cells */
            [data-testid="stDataFrame"] td {
                color: var(--text-primary) !important;
            }
            
            /* Selectboxes and inputs */
            [data-baseweb="select"] {
                background-color: var(--bg-secondary) !important;
            }
            
            [data-baseweb="select"] * {
                color: var(--text-primary) !important;
            }
            
            /* Input fields */
            input, textarea {
                background-color: var(--bg-secondary) !important;
                color: var(--text-primary) !important;
                border-color: var(--border-color) !important;
            }
            
            /* Buttons */
            button {
                background-color: var(--bg-secondary) !important;
                color: var(--text-primary) !important;
                border-color: var(--border-color) !important;
            }
            
            button:hover {
                background-color: var(--bg-tertiary) !important;
                border-color: var(--accent-color) !important;
            }
            
            /* Captions */
            .stCaption {
                color: var(--text-secondary) !important;
            }
            
            /* Info/warning boxes */
            .stAlert {
                background-color: var(--bg-secondary) !important;
                color: var(--text-primary) !important;
                border-color: var(--border-color) !important;
            }
            
            /* Expander */
            [data-testid="stExpander"] {
                background-color: var(--bg-secondary) !important;
                border-color: var(--border-color) !important;
            }
            
            [data-testid="stExpander"] * {
                color: var(--text-primary) !important;
            }
            
            /* Markdown */
            .stMarkdown {
                color: var(--text-primary) !important;
            }
            
            /* Sliders */
            [data-baseweb="slider"] {
                background-color: var(--bg-secondary) !important;
            }
            
            /* Checkbox */
            [data-testid="stCheckbox"] label {
                color: var(--text-primary) !important;
            }
        </style>
        """
    else:
        return """
        <style>
            /* Light mode colors */
            :root {
                --bg-primary: #ffffff;
                --bg-secondary: #f8f9fa;
                --text-primary: #262730;
                --text-secondary: #6c757d;
                --border-color: #e0e0e0;
                --accent-color: #4ECDC4;
            }
            
            /* Main background */
            .main {
                background-color: var(--bg-primary);
                color: var(--text-primary);
            }
            
            /* Sidebar */
            [data-testid="stSidebar"] {
                background-color: var(--bg-secondary);
            }
            
            /* Content padding */
            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            
            /* Headers */
            h1, h2, h3 {
                color: var(--text-primary);
                font-weight: 500;
            }
            
            h1 {
                font-weight: 600;
                margin-bottom: 1rem;
            }
            
            h2, h3 {
                margin-top: 1.5rem;
                margin-bottom: 0.75rem;
            }
            
            /* Dividers */
            hr {
                margin: 2rem 0;
                border: none;
                border-top: 1px solid var(--border-color);
            }
            
            /* Metrics */
            [data-testid="stMetricValue"] {
                font-size: 1.5rem;
                color: var(--text-primary);
            }
            
            [data-testid="stMetricLabel"] {
                color: var(--text-secondary);
            }
            
            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {
                gap: 2rem;
            }
            
            .stTabs [data-baseweb="tab"] {
                padding: 0.5rem 1rem;
                font-weight: 500;
                color: var(--text-secondary);
            }
            
            .stTabs [data-baseweb="tab"][aria-selected="true"] {
                color: var(--accent-color);
            }
            
            /* Captions */
            .stCaption {
                color: var(--text-secondary);
            }
        </style>
        """

st.markdown(get_custom_css(st.session_state.dark_mode), unsafe_allow_html=True)

# ---------------- HELPERS ----------------
def pct_to_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if s.endswith("%"):
        s = s[:-1].strip()
        return float(s) / 100.0 if s else np.nan
    try:
        return float(s)
    except:
        return np.nan

def find_file(names):
    for n in names:
        if Path(n).exists():
            return n
    return None

def clean_touchrate(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    first = df.columns[0]
    df = df.rename(columns={first: "raw"})

    is_logo = df["raw"].astype(str).str.contains(r"\blogo\b", case=False, na=False)
    logos = df[is_logo].reset_index(drop=True)
    names = df[~is_logo].reset_index(drop=True)

    if len(logos) == 0 or len(names) == 0:
        st.error("CSV format issue: Could not find team/player rows.")
        st.stop()
    
    if abs(len(logos) - len(names)) > 1:
        st.error(f"Data format issue: Found {len(logos)} team rows but {len(names)} player rows.")
        st.stop()

    n = min(len(logos), len(names))
    logos = logos.iloc[:n].copy()
    names = names.iloc[:n].copy()

    logos["team"] = logos["raw"].astype(str).str.replace(r"\s*logo\s*", "", regex=True).str.strip()
    logos["player"] = names["raw"].astype(str).str.strip()

    cols = {c.strip().lower(): c for c in logos.columns}
    def col_for(*options):
        for o in options:
            if o in cols:
                return cols[o]
        return None

    c_touches = col_for("touches")
    c_goal    = col_for("goal rate", "goal_rate")
    c_ast     = col_for("assist rate", "assist_rate")
    c_pass    = col_for("pass rate", "pass_rate")
    c_shot    = col_for("shot rate", "shot_rate")
    c_to      = col_for("turnover rate", "turnover_rate", "to rate", "turnover%")

    if not c_touches:
        st.error("Missing required column: 'touches'")
        st.stop()

    out = pd.DataFrame({
        "player": logos["player"],
        "team": logos["team"],
        "touches": pd.to_numeric(logos[c_touches] if c_touches else np.nan, errors="coerce"),
        "goal_rate": (logos[c_goal].apply(pct_to_float) if c_goal else np.nan),
        "assist_rate": (logos[c_ast].apply(pct_to_float) if c_ast else np.nan),
        "pass_rate": (logos[c_pass].apply(pct_to_float) if c_pass else np.nan),
        "shot_rate": (logos[c_shot].apply(pct_to_float) if c_shot else np.nan),
        "turnover_rate": (logos[c_to].apply(pct_to_float) if c_to else np.nan),
    })

    out = out.dropna(subset=["player", "touches"])
    out = out[out["player"].astype(str).str.strip().ne("")]
    
    if len(out) == 0:
        st.error("No valid player data found.")
        st.stop()
    
    return out.reset_index(drop=True)

def compute_goal_value(reg_df: pd.DataFrame) -> float:
    if not {"goals", "scoringPoints"}.issubset(reg_df.columns):
        return 1.25
    
    goals = pd.to_numeric(reg_df["goals"], errors="coerce").sum()
    pts   = pd.to_numeric(reg_df["scoringPoints"], errors="coerce").sum()
    
    if goals == 0 or pd.isna(goals):
        return 1.25
    
    calculated = float(pts / goals)
    
    if calculated < 1.0 or calculated > 3.0:
        return 1.25
    
    return calculated

def add_roles(df_in: pd.DataFrame) -> tuple:
    df = df_in.copy()
    numeric_cols = ["touches", "goal_rate", "assist_rate", "pass_rate", "shot_rate", "turnover_rate"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    q75 = df[numeric_cols].quantile(0.75)
    q50 = df[numeric_cols].quantile(0.50)

    def role(r):
        if (r["assist_rate"] >= q75["assist_rate"]) and (r["turnover_rate"] >= q75["turnover_rate"]):
            return "Risky Creator"
        if (r["touches"] >= q75["touches"]) and (r["assist_rate"] >= q75["assist_rate"]) and (r["turnover_rate"] < q75["turnover_rate"]):
            return "Ball-Dominant Creator"
        if (r["assist_rate"] >= q75["assist_rate"]) and (r["pass_rate"] >= q50["pass_rate"]) and (r["shot_rate"] < q75["shot_rate"]):
            return "Facilitator"
        if (r["shot_rate"] >= q75["shot_rate"]) and (r["assist_rate"] < q50["assist_rate"]):
            return "Shooter"
        if (r["goal_rate"] >= q75["goal_rate"]) and (r["touches"] < q50["touches"]):
            return "Efficient Finisher"
        return "Other"

    df["role"] = df.apply(role, axis=1)
    return df, q75, q50

def add_percentiles(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    metrics = ["touches", "oit_index", "goal_rate", "assist_rate", "shot_rate", "pass_rate", "turnover_rate"]
    
    for metric in metrics:
        df[f"{metric}_percentile"] = df[metric].rank(pct=True) * 100
    
    return df

def load_year_data(year: int, assist_weight: float, turnover_weight: float):
    config = YEARS_CONFIG.get(year)
    if not config:
        return None, None
    
    touch_file = find_file([config["touchrate"]])
    stats_file = find_file([config["stats"]])
    
    if not touch_file or not stats_file:
        return None, f"Missing data files for {year}"
    
    try:
        touch_raw = pd.read_csv(touch_file)
        reg_raw = pd.read_csv(stats_file)
        
        touch = clean_touchrate(touch_raw)
        goal_value = compute_goal_value(reg_raw)
        
        df = touch.copy()
        for c in ["goal_rate","assist_rate","pass_rate","shot_rate","turnover_rate"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        df["touches"] = pd.to_numeric(df["touches"], errors="coerce").fillna(0.0)
        
        df["oit"] = (df["goal_rate"] * goal_value) + (df["assist_rate"] * assist_weight) - (df["turnover_rate"] * turnover_weight)
        avg_oit = float(df["oit"].mean()) if len(df) else 0.0
        df["oit_index"] = (df["oit"] / avg_oit) * 100.0 if avg_oit != 0 else 0.0
        df, q75, q50 = add_roles(df)
        df = add_percentiles(df)
        df["year"] = year
        
        return {"df": df, "q75": q75, "q50": q50, "goal_value": goal_value, "avg_oit": avg_oit}, None
        
    except Exception as e:
        return None, f"Error loading {year} data: {str(e)}"

def load_all_years_data(assist_weight: float, turnover_weight: float):
    """Load data for all available years"""
    all_data = {}
    for year in YEARS_CONFIG.keys():
        data, error = load_year_data(year, assist_weight, turnover_weight)
        if data:
            all_data[year] = data
    return all_data

# ---------------- SIDEBAR ----------------
st.sidebar.title("PLL OIT Analytics")

# Dark mode toggle
dark_mode_col1, dark_mode_col2 = st.sidebar.columns([3, 1])
with dark_mode_col1:
    st.sidebar.markdown("**Appearance**")
with dark_mode_col2:
    if st.sidebar.button("🌓", help="Toggle dark mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

st.sidebar.markdown("---")

# About section (collapsed by default for cleaner look)
with st.sidebar.expander("About OIT"):
    st.markdown("""
    **Offensive Impact per Touch (OIT)** quantifies offensive value created per possession.
    
    **Formula:**  
    `OIT = (Goal Rate × Goal Value) + (Assist Rate × Weight) - (Turnover Rate × Penalty)`
    
    **OIT Index:** Player OIT relative to league average (100 = average)
    """)

st.sidebar.markdown("---")

# Season Selection
st.sidebar.subheader("Season")
available_years = sorted(YEARS_CONFIG.keys(), reverse=True)
selected_year = st.sidebar.selectbox(
    "Select season to analyze",
    available_years,
    index=0,
    label_visibility="collapsed"
)

comparison_mode = st.sidebar.checkbox("Enable year comparison")

if comparison_mode:
    compare_year = st.sidebar.selectbox(
        "Compare with",
        [y for y in available_years if y != selected_year]
    )
else:
    compare_year = None

st.sidebar.markdown("---")

# OIT Calculation Parameters
st.sidebar.subheader("OIT Parameters")
assist_weight = st.sidebar.slider(
    "Assist weight", 
    0.0, 2.0, 0.7, 0.05,
    help="Multiplier for assist value in OIT calculation"
)
turnover_weight = st.sidebar.slider(
    "Turnover penalty", 
    0.0, 2.0, 0.6, 0.05,
    help="Penalty multiplier for turnovers in OIT calculation"
)

st.sidebar.markdown("---")

# Filters
st.sidebar.subheader("Filters")
min_touches = st.sidebar.slider(
    "Minimum touches", 
    0, 600, 100, 10,
    help="Filter out players below this touch threshold"
)

# ---------------- LOAD DATA ----------------
primary_data, primary_error = load_year_data(selected_year, assist_weight, turnover_weight)

if primary_error:
    st.error(f"Error loading {selected_year} data: {primary_error}")
    st.info("Ensure CSV files are named correctly and in the same folder as app.py")
    st.stop()

if not primary_data:
    st.error(f"No data available for {selected_year}")
    st.stop()

df = primary_data["df"]
q75 = primary_data["q75"]
q50 = primary_data["q50"]
goal_value = primary_data["goal_value"]
avg_oit = primary_data["avg_oit"]

# Load comparison data if needed
if comparison_mode and compare_year:
    compare_data, compare_error = load_year_data(compare_year, assist_weight, turnover_weight)
    if compare_error:
        st.warning(f"Could not load {compare_year} data: {compare_error}")
        compare_data = None
else:
    compare_data = None

# Apply filters
df = df[df["touches"] >= min_touches].copy()
if df.empty:
    st.warning("No players meet the current filter criteria. Try lowering minimum touches.")
    st.stop()

# Team filter
teams_available = sorted(df["team"].unique())
teams_selected = st.sidebar.multiselect(
    "Filter by team", 
    teams_available,
    help="Select specific teams (leave empty for all)"
)
if teams_selected:
    df = df[df["team"].isin(teams_selected)]
    if df.empty:
        st.warning("No players from selected teams meet the touch minimum.")
        st.stop()

# Role thresholds (collapsed)
with st.sidebar.expander("Role Thresholds"):
    st.caption("**75th percentile:**")
    st.text(f"Assist: {q75['assist_rate']:.1%}")
    st.text(f"Turnover: {q75['turnover_rate']:.1%}")
    st.text(f"Touches: {q75['touches']:.0f}")
    st.caption("**50th percentile:**")
    st.text(f"Assist: {q50['assist_rate']:.1%}")
    st.text(f"Touches: {q50['touches']:.0f}")

avg_touches = float(df["touches"].mean())

# ---------------- MAIN UI ----------------
st.title(f"{selected_year} PLL Season Analysis")

# Cleaner metrics row
c1, c2, c3 = st.columns(3)
c1.metric("League Avg OIT", f"{avg_oit:.3f}")
c2.metric("Players Analyzed", f"{len(df)}")
c3.metric("Goal Value", f"{goal_value:.2f} pts")

st.markdown("---")

# Create tabs
if comparison_mode and compare_data:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Player Analysis", 
        "Team Summary", 
        "Percentile Rankings", 
        "Year Comparison",
        "Career Trajectories"
    ])
else:
    tab1, tab2, tab3, tab5 = st.tabs([
        "Player Analysis", 
        "Team Summary", 
        "Percentile Rankings",
        "Career Trajectories"
    ])
    tab4 = None

# TAB 1: Player Analysis (with highlighted scatter)
with tab1:
    col_left, col_right = st.columns([1.6, 1])
    
    with col_left:
        st.subheader("Usage vs Efficiency")
        
        # Player selector for highlighting
        highlight_player = st.selectbox(
            "Highlight player on chart (optional)",
            ["None"] + sorted(df["player"].unique().tolist()),
            key="scatter_highlight"
        )
        
        # Create base scatter plot
        fig = px.scatter(
            df,
            x="touches",
            y="oit_index",
            size="shot_rate",
            size_max=28,
            color="role",
            color_discrete_map=ROLE_COLORS,
            hover_name="player",
            hover_data={
                "team": True,
                "touches": True,
                "goal_rate": ':.2%',
                "assist_rate": ':.2%',
                "turnover_rate": ':.2%',
                "oit_index": ':.1f',
            },
            labels={"touches": "Touches (Usage)", "oit_index": "OIT Index"},
        )
        
        # Dark mode chart styling
        if st.session_state.dark_mode:
            fig.update_layout(
                paper_bgcolor='#0e1117',
                plot_bgcolor='#0e1117',
                font_color='#fafafa'
            )
            fig.update_xaxes(gridcolor='#2d3139', zerolinecolor='#2d3139')
            fig.update_yaxes(gridcolor='#2d3139', zerolinecolor='#2d3139')
        
        # Add reference lines
        fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.3, annotation_text="League Avg")
        fig.add_vline(x=avg_touches, line_dash="dash", line_color="gray", opacity=0.3)
        
        # Highlight selected player
        if highlight_player != "None":
            player_row = df[df["player"] == highlight_player].iloc[0]
            
            # Fade all other points
            fig.update_traces(opacity=0.4)
            
            # Add highlighted point
            fig.add_trace(go.Scatter(
                x=[player_row["touches"]],
                y=[player_row["oit_index"]],
                mode='markers',
                marker=dict(
                    size=30,
                    color='#FFD700',  # Gold
                    line=dict(width=3, color='#FF4500'),  # Orange outline
                    symbol='star'
                ),
                name=highlight_player,
                showlegend=False,
                hovertemplate=f"<b>{highlight_player}</b><br>" +
                             f"OIT Index: {player_row['oit_index']:.1f}<br>" +
                             f"Touches: {int(player_row['touches'])}<extra></extra>"
            ))
        
        # Annotate top 5 (if no player highlighted)
        if highlight_player == "None":
            top5 = df.nlargest(5, "oit_index")
            for _, row in top5.iterrows():
                name_parts = row["player"].split()
                display_name = name_parts[-1] if len(name_parts) > 1 else row["player"]
                fig.add_annotation(
                    x=row["touches"], 
                    y=row["oit_index"],
                    text=display_name,
                    showarrow=False,
                    yshift=12,
                    font=dict(size=9, color="black"),
                    bgcolor="rgba(255,255,255,0.7)",
                    borderpad=2
                )
        
        fig.update_layout(
            height=600, 
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.subheader("Player Details")
        
        player_select = st.selectbox(
            "Select player",
            sorted(df["player"].unique().tolist()),
            key="player_card"
        )
        
        if player_select:
            p = df[df["player"] == player_select].iloc[0]
            
            # Clean player card
            st.markdown(f"### {p['player']}")
            st.caption(f"{p['team']} • {p['role']}")
            
            # Key metrics in grid
            m1, m2, m3 = st.columns(3)
            m1.metric("OIT Index", f"{p['oit_index']:.1f}", f"Avg: {100:.0f}")
            m2.metric("Touches", f"{int(p['touches'])}", f"Avg: {int(avg_touches)}")
            m3.metric("Role", p['role'])
            
            st.markdown("---")
            
            # Rates table with league averages
            league_avg_goal = df["goal_rate"].mean()
            league_avg_assist = df["assist_rate"].mean()
            league_avg_turnover = df["turnover_rate"].mean()
            league_avg_shot = df["shot_rate"].mean()
            
            rates_df = pd.DataFrame({
                "Metric": ["Goal Rate", "Assist Rate", "Turnover Rate", "Shot Rate"],
                "Value": [
                    f"{p['goal_rate']:.1%}",
                    f"{p['assist_rate']:.1%}",
                    f"{p['turnover_rate']:.1%}",
                    f"{p['shot_rate']:.1%}"
                ],
                "League Avg": [
                    f"{league_avg_goal:.1%}",
                    f"{league_avg_assist:.1%}",
                    f"{league_avg_turnover:.1%}",
                    f"{league_avg_shot:.1%}"
                ]
            })
            st.dataframe(rates_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Top performers
        st.subheader("Top OIT Index")
        top_df = df.nlargest(10, "oit_index")[["player", "team", "oit_index", "touches"]]
        top_df.columns = ["Player", "Team", "OIT Index", "Touches"]
        st.dataframe(top_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    
    # Usage optimization
    st.subheader("Usage Optimization")
    
    underutilized = df[
        (df["oit_index"] > 110) & 
        (df["touches"] < avg_touches * 0.85)
    ].sort_values("oit_index", ascending=False)

    overutilized = df[
        (df["oit_index"] < 90) & 
        (df["touches"] > avg_touches * 1.15)
    ].sort_values("oit_index", ascending=True)

    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### Should Get More Touches")
        st.caption("High efficiency but below-average usage")
        if len(underutilized) > 0:
            under_df = underutilized[["player", "team", "oit_index", "touches"]].head(10)
            under_df.columns = ["Player", "Team", "OIT Index", "Touches"]
            st.dataframe(under_df, use_container_width=True, hide_index=True)
        else:
            st.info("No players meet criteria")

    with c2:
        st.markdown("#### Should Get Fewer Touches")
        st.caption("Low efficiency but above-average usage")
        if len(overutilized) > 0:
            over_df = overutilized[["player", "team", "oit_index", "touches"]].head(10)
            over_df.columns = ["Player", "Team", "OIT Index", "Touches"]
            st.dataframe(over_df, use_container_width=True, hide_index=True)
        else:
            st.info("No players meet criteria")

# TAB 2: Team Summary
with tab2:
    st.subheader("Team Performance")
    
    team_stats = df.groupby("team").agg({
        "oit_index": "mean",
        "touches": "sum",
        "goal_rate": "mean",
        "assist_rate": "mean",
        "turnover_rate": "mean",
        "player": "count"
    }).reset_index()
    
    team_stats.columns = ["Team", "Avg OIT Index", "Total Touches", "Avg Goal Rate", "Avg Assist Rate", "Avg TO Rate", "Players"]
    team_stats = team_stats.sort_values("Avg OIT Index", ascending=False)
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        fig_team = px.bar(
            team_stats,
            x="Team",
            y="Avg OIT Index",
            color="Avg OIT Index",
            color_continuous_scale="RdYlGn",
            text="Avg OIT Index"
        )
        fig_team.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5, annotation_text="League Avg")
        fig_team.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_team.update_layout(height=500, showlegend=False)
        
        # Dark mode support
        if st.session_state.dark_mode:
            fig_team.update_layout(
                paper_bgcolor='#0e1117',
                plot_bgcolor='#0e1117',
                font_color='#fafafa'
            )
            fig_team.update_xaxes(gridcolor='#2d3139')
            fig_team.update_yaxes(gridcolor='#2d3139')
        
        st.plotly_chart(fig_team, use_container_width=True)
    
    with col2:
        st.dataframe(
            team_stats[["Team", "Avg OIT Index", "Players", "Avg Goal Rate", "Avg Assist Rate"]]
                .style.format({
                    "Avg OIT Index": "{:.1f}",
                    "Avg Goal Rate": "{:.1%}",
                    "Avg Assist Rate": "{:.1%}"
                }),
            use_container_width=True,
            hide_index=True,
            height=500
        )
    
    st.markdown("---")
    
    # Role distribution
    st.subheader("Role Distribution by Team")
    role_dist = df.groupby(["team", "role"]).size().reset_index(name="count")
    
    fig_roles = px.bar(
        role_dist,
        x="team",
        y="count",
        color="role",
        color_discrete_map=ROLE_COLORS,
        labels={"team": "Team", "count": "Players", "role": "Role"},
        barmode="stack"
    )
    fig_roles.update_layout(height=400)
    
    # Dark mode support
    if st.session_state.dark_mode:
        fig_roles.update_layout(
            paper_bgcolor='#0e1117',
            plot_bgcolor='#0e1117',
            font_color='#fafafa'
        )
    
    st.plotly_chart(fig_roles, use_container_width=True)

# TAB 3: Percentile Rankings
with tab3:
    st.subheader("Percentile Rankings")
    
    percentile_player = st.selectbox(
        "Select player",
        sorted(df["player"].unique().tolist()),
        key="percentile_select"
    )
    
    if percentile_player:
        p_data = df[df["player"] == percentile_player].iloc[0]
        
        st.markdown(f"### {p_data['player']}")
        st.caption(f"{p_data['team']} • {p_data['role']}")
        
        # Percentile visualization
        percentile_metrics = {
            "OIT Index": p_data["oit_index_percentile"],
            "Touches": p_data["touches_percentile"],
            "Goal Rate": p_data["goal_rate_percentile"],
            "Assist Rate": p_data["assist_rate_percentile"],
            "Shot Rate": p_data["shot_rate_percentile"],
            "Turnover Rate": 100 - p_data["turnover_rate_percentile"]
        }
        
        perc_df = pd.DataFrame({
            "Metric": list(percentile_metrics.keys()),
            "Percentile": list(percentile_metrics.values())
        })
        
        fig_perc = px.bar(
            perc_df,
            y="Metric",
            x="Percentile",
            orientation="h",
            color="Percentile",
            color_continuous_scale="RdYlGn",
            text="Percentile"
        )
        fig_perc.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5, annotation_text="Median")
        fig_perc.add_vline(x=75, line_dash="dot", line_color="green", opacity=0.3, annotation_text="75th")
        fig_perc.add_vline(x=25, line_dash="dot", line_color="red", opacity=0.3, annotation_text="25th")
        fig_perc.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        fig_perc.update_layout(height=400, showlegend=False, xaxis_range=[0, 105])
        fig_perc.update_yaxes(categoryorder="total ascending")
        
        # Dark mode support
        if st.session_state.dark_mode:
            fig_perc.update_layout(
                paper_bgcolor='#0e1117',
                plot_bgcolor='#0e1117',
                font_color='#fafafa'
            )
            fig_perc.update_xaxes(gridcolor='#2d3139')
            fig_perc.update_yaxes(gridcolor='#2d3139')
        
        st.plotly_chart(fig_perc, use_container_width=True)
        
        st.caption("Note: Turnover Rate is inverted (higher percentile = lower turnover rate)")

# TAB 4: Year Comparison (if enabled)
if tab4 and comparison_mode and compare_data:
    with tab4:
        st.subheader(f"{selected_year} vs {compare_year} Comparison")
        
        df_compare = compare_data["df"]
        df_compare = df_compare[df_compare["touches"] >= min_touches].copy()
        
        players_both = set(df["player"]) & set(df_compare["player"])
        
        if len(players_both) == 0:
            st.warning(f"No players with {min_touches}+ touches in both seasons")
        else:
            st.info(f"Analyzing {len(players_both)} players who played both seasons")
            
            comparison_list = []
            for player in players_both:
                p1 = df[df["player"] == player].iloc[0]
                p2 = df_compare[df_compare["player"] == player].iloc[0]
                
                comparison_list.append({
                    "player": player,
                    "team": p1["team"],
                    "oit_current": p1["oit_index"],
                    "oit_previous": p2["oit_index"],
                    "oit_change": p1["oit_index"] - p2["oit_index"],
                    "role_current": p1["role"],
                    "role_previous": p2["role"]
                })
            
            comp_df = pd.DataFrame(comparison_list)
            comp_df = comp_df.sort_values("oit_change", ascending=False)
            
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("#### Biggest Improvers")
                improvers = comp_df.nlargest(10, "oit_change")
                imp_df = improvers[["player", "team", "oit_change", "oit_current"]]
                imp_df.columns = ["Player", "Team", "Change", f"{selected_year} OIT"]
                st.dataframe(
                    imp_df.style.format({"Change": "{:+.1f}", f"{selected_year} OIT": "{:.1f}"}),
                    use_container_width=True,
                    hide_index=True
                )
            
            with c2:
                st.markdown("#### Biggest Decliners")
                decliners = comp_df.nsmallest(10, "oit_change")
                dec_df = decliners[["player", "team", "oit_change", "oit_current"]]
                dec_df.columns = ["Player", "Team", "Change", f"{selected_year} OIT"]
                st.dataframe(
                    dec_df.style.format({"Change": "{:+.1f}", f"{selected_year} OIT": "{:.1f}"}),
                    use_container_width=True,
                    hide_index=True
                )
            
            st.markdown("---")
            
            # Scatter comparison
            fig_comp = px.scatter(
                comp_df,
                x="oit_previous",
                y="oit_current",
                size=abs(comp_df["oit_change"]),
                color="oit_change",
                color_continuous_scale="RdYlGn",
                hover_name="player",
                labels={
                    "oit_previous": f"{compare_year} OIT",
                    "oit_current": f"{selected_year} OIT"
                }
            )
            
            min_val = min(comp_df["oit_previous"].min(), comp_df["oit_current"].min())
            max_val = max(comp_df["oit_previous"].max(), comp_df["oit_current"].max())
            fig_comp.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='No Change',
                showlegend=True
            ))
            
            fig_comp.update_layout(height=600)
            
            # Dark mode support
            if st.session_state.dark_mode:
                fig_comp.update_layout(
                    paper_bgcolor='#0e1117',
                    plot_bgcolor='#0e1117',
                    font_color='#fafafa'
                )
                fig_comp.update_xaxes(gridcolor='#2d3139')
                fig_comp.update_yaxes(gridcolor='#2d3139')
            
            st.plotly_chart(fig_comp, use_container_width=True)
            st.caption("Players above the line improved; players below declined")

# TAB 5: Career Trajectories (NEW)
with tab5:
    st.subheader("Career Progression Analysis")
    
    # Load all years data
    all_years_data = load_all_years_data(assist_weight, turnover_weight)
    
    if len(all_years_data) < 2:
        st.warning("Need data from multiple years for career analysis")
    else:
        # Find players who appear in multiple years
        all_players = set()
        for year, data in all_years_data.items():
            all_players.update(data["df"]["player"].unique())
        
        # Filter to players with 2+ years of data
        players_with_history = []
        for player in all_players:
            years_played = sum(1 for data in all_years_data.values() if player in data["df"]["player"].values)
            if years_played >= 2:
                players_with_history.append(player)
        
        if not players_with_history:
            st.warning("No players found with multi-year history")
        else:
            trajectory_player = st.selectbox(
                "Select player to view career trajectory",
                sorted(players_with_history),
                key="trajectory_select"
            )
            
            if trajectory_player:
                # Build career data
                career_data = []
                for year in sorted(all_years_data.keys()):
                    year_df = all_years_data[year]["df"]
                    player_data = year_df[year_df["player"] == trajectory_player]
                    
                    if not player_data.empty:
                        p = player_data.iloc[0]
                        career_data.append({
                            "year": year,
                            "oit_index": p["oit_index"],
                            "touches": p["touches"],
                            "team": p["team"],
                            "role": p["role"]
                        })
                
                career_df = pd.DataFrame(career_data)
                
                st.markdown(f"### {trajectory_player}")
                
                # Career trajectory chart
                fig_career = go.Figure()
                
                fig_career.add_trace(go.Scatter(
                    x=career_df["year"],
                    y=career_df["oit_index"],
                    mode='lines+markers',
                    name='OIT Index',
                    line=dict(color='#4ECDC4', width=3),
                    marker=dict(size=12)
                ))
                
                fig_career.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5, annotation_text="League Avg")
                
                fig_career.update_layout(
                    title=f"{trajectory_player} - OIT Index Over Time",
                    xaxis_title="Season",
                    yaxis_title="OIT Index",
                    height=400,
                    hovermode='x unified'
                )
                
                # Dark mode support
                if st.session_state.dark_mode:
                    fig_career.update_layout(
                        paper_bgcolor='#0e1117',
                        plot_bgcolor='#0e1117',
                        font_color='#fafafa'
                    )
                    fig_career.update_xaxes(gridcolor='#2d3139')
                    fig_career.update_yaxes(gridcolor='#2d3139')
                
                st.plotly_chart(fig_career, use_container_width=True)
                
                # Career summary table
                st.markdown("#### Career Summary")
                summary_df = career_df[["year", "team", "role", "oit_index", "touches"]].copy()
                summary_df.columns = ["Year", "Team", "Role", "OIT Index", "Touches"]
                
                # Calculate year-over-year change
                summary_df["YoY Change"] = summary_df["OIT Index"].diff()
                
                st.dataframe(
                    summary_df.style.format({
                        "OIT Index": "{:.1f}",
                        "Touches": "{:.0f}",
                        "YoY Change": "{:+.1f}"
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Career insights
                st.markdown("#### Career Insights")
                peak_year = career_df.loc[career_df["oit_index"].idxmax()]
                current_year = career_df.iloc[-1]
                first_year = career_df.iloc[0]
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Peak OIT", f"{peak_year['oit_index']:.1f}", f"in {int(peak_year['year'])}")
                col2.metric("Career Change", f"{current_year['oit_index'] - first_year['oit_index']:+.1f}")
                col3.metric("Years Tracked", len(career_df))

# Export section
st.markdown("---")
col1, col2 = st.columns([3, 1])
with col1:
    st.caption(f"Data: {selected_year} PLL Season | OIT Dashboard v7.0")
with col2:
    st.download_button(
        "Export Data",
        data=df.sort_values("oit_index", ascending=False).to_csv(index=False).encode("utf-8"),
        file_name=f"pll_oit_{selected_year}.csv",
        mime="text/csv",
    )
