from __future__ import annotations

from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor


st.set_page_config(
    page_title="Parkline | Smart Parking",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)

LOTS = [
    {"name": "Civic Centre", "area": "CENTRAL DISTRICT", "spaces": 180, "walk": 3, "code": 0},
    {"name": "Market Street", "area": "RIVERFRONT", "spaces": 124, "walk": 5, "code": 1},
    {"name": "Union Square", "area": "MIDTOWN", "spaces": 210, "walk": 7, "code": 2},
    {"name": "Museum Quarter", "area": "CULTURAL MILE", "spaces": 96, "walk": 4, "code": 3},
    {"name": "East Station", "area": "TRANSIT HUB", "spaces": 156, "walk": 8, "code": 4},
]
FEATURES = ["hour", "weekday", "lot", "rain", "event", "weekend"]


@st.cache_resource
def train_model() -> RandomForestRegressor:
    """Train a demo demand model on reproducible, synthetic occupancy history."""
    rng = np.random.default_rng(42)
    rows: list[dict[str, float]] = []
    for day_offset in range(84):
        day = date(2026, 1, 1) + timedelta(days=day_offset)
        weekday = day.weekday()
        weekend = int(weekday >= 5)
        for lot in LOTS:
            for hour in range(24):
                rain = int(rng.random() < 0.22)
                event = int(rng.random() < (0.10 if weekend else 0.06))
                morning_peak = np.exp(-((hour - 9) / 2.8) ** 2)
                evening_peak = np.exp(-((hour - 17) / 3.0) ** 2)
                daytime = np.exp(-((hour - 13) / 6.5) ** 2)
                baseline = [0.27, 0.21, 0.18, 0.12, 0.16][lot["code"]]
                demand = baseline + 0.28 * morning_peak + 0.25 * evening_peak + 0.25 * daytime
                demand += 0.10 * weekend + 0.12 * event + 0.04 * rain
                demand += 0.05 * np.sin(day_offset / 9 + lot["code"])
                demand += rng.normal(0, 0.035)
                rows.append(
                    {
                        "hour": hour,
                        "weekday": weekday,
                        "lot": lot["code"],
                        "rain": rain,
                        "event": event,
                        "weekend": weekend,
                        "occupancy": float(np.clip(demand, 0.04, 0.97)),
                    }
                )

    history = pd.DataFrame(rows)
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=14,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(history[FEATURES], history["occupancy"])
    return model


def predict_rate(
    model: RandomForestRegressor,
    lot: dict[str, int | str],
    when: datetime,
    rain: int = 0,
    event: int = 0,
) -> float:
    features = pd.DataFrame(
        [
            {
                "hour": when.hour,
                "weekday": when.weekday(),
                "lot": lot["code"],
                "rain": rain,
                "event": event,
                "weekend": int(when.weekday() >= 5),
            }
        ]
    )
    return float(np.clip(model.predict(features)[0], 0.02, 0.99))


def available_spaces(
    model: RandomForestRegressor,
    lot: dict[str, int | str],
    when: datetime,
    reservations: int = 0,
    rain: int = 0,
    event: int = 0,
) -> tuple[int, int]:
    capacity = int(lot["spaces"])
    occupied = round(predict_rate(model, lot, when, rain, event) * capacity) + reservations
    occupied = min(capacity, occupied)
    return capacity - occupied, occupied


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --ink:#1d302c; --muted:#71807a; --paper:#f5f6f1; --green:#315d4c; --lime:#c5dc7d; --amber:#d98a36; --line:#e2e7df; }
    html, body, [class*="css"] { font-family:'DM Sans', sans-serif; color:var(--ink); }
    .stApp { background:var(--paper); }
    [data-testid="stSidebar"] { background:#e9eee6; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#51645b; }
    .block-container { max-width:1440px; padding-top:2rem; padding-bottom:3rem; }
    h1, h2, h3 { font-family:'Manrope', sans-serif !important; color:var(--ink) !important; letter-spacing:0 !important; }
    h1 { font-size:2.4rem !important; font-weight:700 !important; }
    h2 { font-size:1.22rem !important; font-weight:700 !important; }
    .eyebrow { color:var(--green); font:500 11px 'DM Mono',monospace; letter-spacing:0; }
    .page-intro { display:flex; justify-content:space-between; align-items:end; margin:0 0 1.35rem; }
    .page-intro p { margin:0; color:var(--muted); font-size:14px; }
    .live-pill { border:1px solid #c8d8c9; border-radius:5px; color:#315d4c; padding:8px 11px; font:500 11px 'DM Mono',monospace; white-space:nowrap; }
    .live-dot { display:inline-block; width:7px; height:7px; border-radius:50%; background:#5d9a70; margin-right:7px; }
    .metric { background:#fff; border:1px solid var(--line); border-radius:7px; padding:17px 18px; min-height:104px; }
    .metric-label { color:var(--muted); font:500 10px 'DM Mono',monospace; }
    .metric-value { color:var(--ink); font:700 27px 'Manrope',sans-serif; margin:5px 0 0; }
    .metric-note { color:var(--muted); font-size:11px; margin-top:1px; }
    .section-label { color:#71807a; font:500 10px 'DM Mono',monospace; margin:0 0 9px; }
    .lot-row { display:flex; align-items:center; gap:14px; background:#fff; border:1px solid var(--line); border-radius:6px; padding:13px 15px; margin:7px 0; }
    .lot-name { font:600 14px 'Manrope',sans-serif; min-width:135px; }
    .lot-area { font:400 9px 'DM Mono',monospace; color:#849089; margin-top:3px; }
    .lot-track { flex:1; height:6px; background:#edf0e9; border-radius:4px; overflow:hidden; }
    .lot-fill { height:100%; border-radius:4px; }
    .lot-count { font:500 12px 'DM Mono',monospace; min-width:82px; text-align:right; }
    .status-tag { font:500 9px 'DM Mono',monospace; min-width:76px; text-align:right; }
    .recommendation { background:#315d4c; border-radius:7px; color:#f6f7f0; padding:20px; margin:4px 0 12px; }
    .recommendation .eyebrow { color:#c5dc7d; }
    .recommendation h3 { color:#fff !important; font-size:19px !important; margin:7px 0 3px !important; }
    .recommendation p { color:#d5e0d7; font-size:12px; margin:0; }
    .recommendation strong { color:#c5dc7d; }
    .booking-row { background:#fff; border:1px solid var(--line); border-radius:6px; padding:11px 13px; margin:6px 0; font-size:13px; }
    .booking-id { color:#859189; font:10px 'DM Mono',monospace; float:right; }
    [data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:7px; padding:13px 16px; }
    [data-testid="stMetricLabel"] p { font:500 10px 'DM Mono',monospace; color:var(--muted); }
    [data-testid="stMetricValue"] { font:700 24px 'Manrope',sans-serif; color:var(--ink); }
    div[data-testid="stButton"] > button[kind="primary"] { background:#315d4c; border-color:#315d4c; color:white; }
    div[data-testid="stButton"] > button { border-radius:5px; }
    hr { border-color:var(--line); }
    footer { visibility:hidden; }
    @media(max-width:700px) { .block-container { padding:1rem 1rem 2rem; } h1 { font-size:1.8rem !important; } .lot-name { min-width:105px; } .status-tag { min-width:58px; } }
    </style>
    """,
    unsafe_allow_html=True,
)

model = train_model()
if "reservations" not in st.session_state:
    st.session_state.reservations = []

with st.sidebar:
    st.markdown("<div class='eyebrow'>PARKLINE / CONTROL DESK</div>", unsafe_allow_html=True)
    st.markdown("## Plan a visit")
    arrival_date = st.date_input("Arrival date", value=date.today(), min_value=date.today())
    arrival_time = st.time_input("Arrival time", value=time(17, 30), step=1800)
    weather = st.selectbox("Forecast", ["Clear", "Rain"])
    event_day = st.toggle("Major event nearby", value=False)
    st.divider()
    st.caption("The demand model uses historical patterns to estimate future space availability.")

arrival = datetime.combine(arrival_date, arrival_time)
rain = int(weather == "Rain")
event = int(event_day)
reservation_counts = {
    lot["name"]: sum(booking["lot"] == lot["name"] for booking in st.session_state.reservations)
    for lot in LOTS
}

now = datetime.now()
live_rows = []
for lot in LOTS:
    free, occupied = available_spaces(model, lot, now, reservation_counts[lot["name"]])
    live_rows.append({**lot, "free": free, "occupied": occupied, "rate": occupied / int(lot["spaces"])})

future_rows = []
for lot in LOTS:
    free, occupied = available_spaces(
        model, lot, arrival, reservation_counts[lot["name"]], rain=rain, event=event
    )
    future_rows.append({**lot, "free": free, "occupied": occupied, "rate": occupied / int(lot["spaces"])})
future_rows.sort(key=lambda row: (row["rate"], row["walk"]))
recommendation = future_rows[0]
total_spaces = sum(int(lot["spaces"]) for lot in LOTS)
free_now = sum(row["free"] for row in live_rows)
occupancy_now = 1 - free_now / total_spaces

st.markdown(
    f"""
    <div class="page-intro">
      <div><div class="eyebrow">CITY PARKING / CENTRAL ZONE</div>
      <h1 style="margin:5px 0 2px">Find your space.</h1>
      <p>Demand-aware parking, without the extra lap around the block.</p></div>
    <div class="live-pill"><span class="live-dot"></span>ESTIMATED LIVE | {now.strftime('%I:%M %p')}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)
metrics = [
    ("OPEN SPACES", f"{free_now:,}", f"across {len(LOTS)} city facilities"),
    ("NETWORK CAPACITY", f"{total_spaces:,}", "managed spaces total"),
    ("CURRENT DEMAND", f"{occupancy_now:.0%}", "model-estimated occupancy"),
    ("BEST ARRIVAL LOT", f"{recommendation['walk']} min", recommendation["name"]),
]
for column, (label, value, note) in zip(metric_cols, metrics):
    column.markdown(
        f"<div class='metric'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-note'>{note}</div></div>",
        unsafe_allow_html=True,
    )

st.write("")
left, right = st.columns([1.15, 0.85], gap="large")
with left:
    st.markdown("## Facility status")
    st.markdown("<div class='section-label'>ESTIMATED CURRENT AVAILABILITY | SORTED BY NAME</div>", unsafe_allow_html=True)
    for row in live_rows:
        level = row["rate"]
        color = "#6a9b72" if level < 0.65 else "#d2a346" if level < 0.82 else "#ca7251"
        label = "OPEN" if level < 0.65 else "FILLING" if level < 0.82 else "BUSY"
        st.markdown(
            f"""
            <div class="lot-row">
              <div style="min-width:135px"><div class="lot-name">{row['name']}</div><div class="lot-area">{row['area']}</div></div>
              <div class="lot-track"><div class="lot-fill" style="width:{level:.0%};background:{color}"></div></div>
              <div class="lot-count">{row['free']} / {row['spaces']}</div>
              <div class="status-tag" style="color:{color}">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption("Occupancy is AI/ML-estimated from generated demo history; connect parking sensors for live counts.")

    st.markdown("## Demand through the day")
    chart_rows = []
    for hour in range(6, 23):
        slot = datetime.combine(arrival_date, time(hour))
        average = np.mean(
            [predict_rate(model, lot, slot, rain=rain, event=event) for lot in LOTS]
        )
        chart_rows.append({"Hour": slot.strftime("%I %p"), "Predicted occupancy": average})
    chart = pd.DataFrame(chart_rows).set_index("Hour")
    st.line_chart(chart, color="#47785f", height=220)

with right:
    st.markdown("## Your arrival")
    st.markdown(
        f"""
        <div class="recommendation">
          <div class="eyebrow">TOP PREDICTION | {arrival.strftime('%a, %b %d - %I:%M %p').upper()}</div>
          <h3>{recommendation['name']}</h3>
          <p><strong>{recommendation['free']} spaces likely open</strong> | {recommendation['walk']} min walk | {recommendation['area'].title()}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='section-label'>OTHER NEARBY OPTIONS | PREDICTED FREE SPACES</div>", unsafe_allow_html=True)
    for row in future_rows[1:]:
        st.markdown(
            f"<div class='booking-row'><b>{row['name']}</b><span style='color:#71807a'> &nbsp;| {row['walk']} min walk</span><span style='float:right;color:#315d4c;font-family:DM Mono,monospace'>{row['free']} open</span></div>",
            unsafe_allow_html=True,
        )

    with st.form("reserve_space"):
        chosen_lot_name = st.selectbox("Reserve a space", [row["name"] for row in future_rows])
        submitted = st.form_submit_button("Reserve one space", type="primary", use_container_width=True)
    if submitted:
        chosen = next(row for row in future_rows if row["name"] == chosen_lot_name)
        if chosen["free"] > 0:
            st.session_state.reservations.append(
                {
                    "lot": chosen_lot_name,
                    "arrival": arrival.strftime("%a, %b %d - %I:%M %p"),
                    "id": f"PL-{len(st.session_state.reservations) + 2418:04d}",
                }
            )
            st.rerun()
        else:
            st.error("That lot is predicted to be full at your arrival time. Choose another facility.")

    if st.session_state.reservations:
        st.markdown("### Your reservations")
        for booking in st.session_state.reservations:
            booking_col, cancel_col = st.columns([4, 1])
            booking_col.markdown(
                f"<div class='booking-row'><b>{booking['lot']}</b><br><span style='color:#71807a'>{booking['arrival']}</span><span class='booking-id'>{booking['id']}</span></div>",
                unsafe_allow_html=True,
            )
            if cancel_col.button("Cancel", key=f"cancel_{booking['id']}"):
                st.session_state.reservations.remove(booking)
                st.rerun()

st.divider()
st.markdown(
    "<div class='section-label'>PARKLINE DEMO | RANDOM FOREST REGRESSOR | SYNTHETIC TRAINING DATA | AVAILABILITY IS AN ESTIMATE</div>",
    unsafe_allow_html=True,
)