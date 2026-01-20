import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mplsoccer import Sbopen, Pitch
from scipy import stats  # for z-score

st.set_page_config(page_title="France 2018 WC Final – Paul Pogba Analysis", layout="wide")

st.title("Test Analysis on Paul Pogba's Performance - 2018 FIFA World Cup Final")

# ------------------------------------------------------------
# 1. Load data from StatsBomb using Sbopen
# ------------------------------------------------------------

parser = Sbopen()

# --- Filter to 2018 FIFA World Cup (International, male) ---
df_competition = parser.competition()
df_filtered = df_competition.loc[
    (df_competition['competition_gender'].isin(['male'])) &
    (df_competition['country_name'].isin(['International'])) &
    (df_competition['competition_name'].isin(['FIFA World Cup'])) &
    (df_competition['season_name'].isin(['2018']))
]

df_match = parser.match(competition_id=43, season_id=3)

# Only the Final (France vs Croatia)
france_final_df = df_match.loc[df_match['competition_stage_name'] == 'Final']

# Get competition name (single value from filtered competition)
competition_name = df_filtered.iloc[0]["competition_name"]

# Create a display label for Streamlit
france_final_df["display_label"] = (
    competition_name + " - " + france_final_df["competition_stage_name"]
)

match_id = st.selectbox(
    "Select Match",
    france_final_df.index,
    format_func=lambda x: f"{france_final_df.loc[x, 'display_label']}"
)

# Get the real selected match_id
selected_match_id = france_final_df.loc[match_id, "match_id"]

# Load events and lineup for the selected match
df_events, df_related, df_freeze, df_tactics = parser.event(selected_match_id)
df_lineup = parser.lineup(selected_match_id)

# Focus on France team
teamplay_name = "France"
df_lineup_fr = df_lineup.loc[df_lineup['team_name'] == teamplay_name].copy()
df_events_fr = df_events.loc[df_events['team_name'] == teamplay_name].copy()


# ---------------------------------------------------
# 3. Single-player analysis – shot map
# ---------------------------------------------------
st.subheader("Single Player Shot Map")

selected_player = st.selectbox(
    "Select Player to Analyze (France)",
    options=sorted(df_lineup_fr["player_name"].dropna().unique())
)

st.write("**Selected Player Info**")
#st.write(df_lineup_fr[df_lineup_fr["player_name"] == selected_player])

st.write(
    df_lineup_fr[df_lineup_fr["player_name"] == selected_player]
    .rename(columns=lambda x: x.replace('_', ' ').title())
)

df_france_event, df_related, df_freeze, df_tactics = parser.event(8658)


# ------------------------------------
# Recovery
# ------------------------------------

df_events = df_france_event.sort_values(['match_id', 'period', 'minute', 'second']).copy()


df_events['event_time'] = (
    (df_events['period'] - 1) * 45 * 60
    + df_events['minute'] * 60
    + df_events['second']
)

df_events['recovery_time'] = np.where(
    df_events['type_name'] == 'Ball Recovery',
    df_events['event_time'],
    np.nan
)
df_events['last_recovery_time'] = (
    df_events
    .groupby(['match_id', 'player_name'])['recovery_time']
    .ffill()
)
# 3. All shots
shots_with_recovery = df_events[df_events['type_name'] == 'Shot'].copy()

# Shot taken after a recovery? (at some earlier point)
shots_with_recovery['after_recovery'] = shots_with_recovery['last_recovery_time'].notna()

# Time since that recovery (in seconds)
shots_with_recovery['time_since_recovery'] = (
    shots_with_recovery['event_time'] - shots_with_recovery['last_recovery_time']
)


# --- Shot Map ---

# -------------------------------
# PLAYER SHOTS INFO
# -------------------------------
st.write(f"**Player Shots Info — {selected_player}**")

player_shots = df_events_fr.loc[
    (df_events_fr['type_name'] == 'Shot') &
    (df_events_fr['player_name'] == selected_player)
]

if player_shots.empty:
    st.info(f"No shots found for **{selected_player}** under the current filters.")

else:
#    st.write(player_shots.set_index('id'))
    st.write(player_shots.set_index('index').rename(columns=lambda x: x.replace('_', ' ').title())
)


# -------------------------------
# SHOTS WITH RECOVERY
# -------------------------------
st.write(f"**Shots With Recovery — {selected_player}**")

shots = shots_with_recovery.loc[
    shots_with_recovery['player_name'] == selected_player]

shots_with_recov = shots[shots['after_recovery']][
    ['minute', 'second', 'time_since_recovery', 'x', 'y', 'outcome_name']
]

if shots_with_recov.empty:
    st.info(f"No shots following a recovery were found for **{selected_player}**.")
else:
    st.write(shots_with_recov.set_index(shots.index).rename(columns=lambda x: x.replace('_', ' ').title())
)


# ✅ SAFETY CHECK: Only draw pitch if shots exist
if shots.empty:
    st.info(f"No shots to display for **{selected_player}**.")

else:
    pitch = Pitch(line_color="black")
    fig, ax = pitch.draw(figsize=(10, 7))

    pitchLengthX = 120
    pitchWidthY = 80

    team1, team2 = df_events.team_name.dropna().unique()[:2]

    for i, shot in shots.iterrows():
        x = shot['x']
        y = shot['y']
        goal = shot['outcome_name'] == 'Goal'
        team_name = shot['team_name']
        after_recovery = shot['after_recovery']

        circleSize = 2

        # ✅ Color logic
        if after_recovery:
            base_color = "gold"
        else:
            base_color = "red" if team_name == team1 else "blue"

        if team_name == team1:
            if goal:
                shotCircle = plt.Circle((x, y), circleSize, color=base_color)
                plt.text(x + 1, y - 2, selected_player)
            else:
                shotCircle = plt.Circle((x, y), circleSize, color=base_color)
                shotCircle.set_alpha(.2)
        else:
            shotCircle = plt.Circle((pitchLengthX - x, pitchWidthY - y), circleSize, color=base_color)
            shotCircle.set_alpha(.2)

        ax.add_patch(shotCircle)

    fig.suptitle(f"{selected_player} – Shots", fontsize=24)
    st.pyplot(fig)



# ---------------------------
# My Z-score analysis
# ---------------------------

player_metrics = (
    shots_with_recovery
    .groupby('player_name')
    .agg(
        total_shots = ('id', 'count'),
        goals = ('outcome_name', lambda x: (x == 'Goal').sum()),
        shots_after_recovery = ('after_recovery', 'sum'),
        avg_time_after_recovery = ('time_since_recovery', 'mean'),
        total_xg = ('shot_statsbomb_xg', 'sum')  # remove if you don't have xG
    )
    .reset_index()
)

player_metrics['pct_shots_after_recovery'] = (
    player_metrics['shots_after_recovery'] / player_metrics['total_shots']
)

metrics_for_z = [
    'total_shots',
    'goals',
    'shots_after_recovery',
    'pct_shots_after_recovery',
    'avg_time_after_recovery',
    'total_xg'
]

for col in metrics_for_z:
    player_metrics[f'z_{col}'] = (
        (player_metrics[col] - player_metrics[col].mean()) /
        player_metrics[col].std(ddof=0)
    )

selected_players = st.multiselect(
    "Select players to compare",
    player_metrics['player_name'].unique()
)

comparison_table = player_metrics[
    player_metrics['player_name'].isin(selected_players)
][
    ['player_name'] +
    [f'z_{m}' for m in metrics_for_z]
]

st.subheader("Player Comparison (Z-Score Analysis)")

if comparison_table.empty:
    st.warning("No comparison data available. Select at least two valid players with events.")
    st.stop()

st.write(
    comparison_table
        .set_index('player_name')
        .round(2).rename(columns=lambda x: x[2:].replace('_', ' ').title(),
        .rename_axis('player_name'.replace('_', ' ').title())  # dynamically rename header
        )


plot_df = comparison_table.set_index('player_name')
    
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(plot_df.columns))  # metrics
width = 0.8 / len(plot_df.index)     # bar width based on number of players

for i, player in enumerate(plot_df.index):
    ax.bar(
        x + i * width,
        plot_df.loc[player].values,
        width,
        label=player
    )

ax.axhline(0, linestyle='--')  # Z-score baseline

ax.set_xticks(x + width * (len(plot_df.index) - 1) / 2)
ax.set_xticklabels([x[2:].replace('_', ' ').title() for x in plot_df.columns], rotation=45, ha='right')

ax.set_ylabel("Z-Score")
ax.set_title("Player Comparison — Z-Score Analysis")
ax.legend()

st.pyplot(fig)


plot_df['overall_z_score'] = plot_df.mean(axis=1)

st.subheader("Overall Z-Score Ranking")
st.write(
    plot_df[['overall_z_score']]
    .sort_values('overall_z_score', ascending=False)
    .round(2).rename(columns=lambda x: x.replace('_', ' ').title())
    .rename_axis('player_name'.replace('_', ' ').title())  # dynamically rename header
        
    
)
