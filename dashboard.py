import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Gas Pressure Monitor", layout="wide")

st.title("Gas Pressure Anomaly Monitor")
st.caption("Prototype: sensor data is synthetic. Earthquake data is real (USGS, India region, 2025).")


# ---------- Load data ----------
@st.cache_data
def load_sensor():
    return pd.read_csv("data/fake_sensor_data_residual.csv", parse_dates=["timestamp"])


@st.cache_data
def load_quakes():
    return pd.read_csv("data/quakes.csv", parse_dates=["time"])


sensor = load_sensor()
quakes = load_quakes()

# ---------- Sidebar controls ----------
st.sidebar.header("Settings")
contamination = st.sidebar.slider(
    "Contamination (fraction flagged)", 0.005, 0.10, 0.02, 0.005
)
min_mag = st.sidebar.slider("Minimum earthquake magnitude", 4.0, 7.5, 4.0, 0.1)

# ---------- Anomaly detection ----------
df = sensor.copy()
df["roll_mean"] = df["pressure_residual"].rolling(30).mean()
df["roll_std"] = df["pressure_residual"].rolling(30).std()
df["rate"] = df["pressure_residual"].diff()
df = df.dropna()

features = ["pressure_residual", "roll_mean", "roll_std", "rate"]
iso = IsolationForest(contamination=contamination, random_state=42)
df["anomaly"] = iso.fit_predict(df[features])

truth = (df.index >= 3500) & (df.index < 3600)
flagged = df["anomaly"] == -1
caught = int((truth & flagged).sum())
false_alarms = int((~truth & flagged).sum())

# ---------- Section 1: sensor ----------
st.header("1. Sensor analysis")

precision = caught / max(int(flagged.sum()), 1)
recall = caught / int(truth.sum())
f1 = 2 * precision * recall / max(precision + recall, 1e-9)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total flagged", int(flagged.sum()))
c2.metric("False alarms", false_alarms)
c3.metric("Precision", f"{precision:.2f}")
c4.metric("Recall", f"{recall:.2f}")
c5.metric("F1 score", f"{f1:.2f}")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df["timestamp"], df["pressure_residual"], color="green", label="residual")
ax.scatter(df["timestamp"][flagged], df["pressure_residual"][flagged],
           color="red", s=10, label="flagged anomaly")
ax.set_xlabel("Time")
ax.set_ylabel("Residual (hPa)")
ax.legend()
st.pyplot(fig)

with st.expander("Raw pressure and temperature"):
    st.line_chart(sensor.set_index("timestamp")[["pressure", "temperature"]])

# ---------- Section 2: earthquakes ----------
st.header("2. Earthquakes (India region, 2025)")

q = quakes[quakes["mag"] >= min_mag]
st.write(f"Showing {len(q)} earthquakes of magnitude {min_mag} or higher.")
st.map(q[["latitude", "longitude"]])
st.dataframe(q.sort_values("mag", ascending=False), use_container_width=True)

st.info("The sensor data (Jan 2026, synthetic) and the earthquakes (2025, real) "
        "are not linked yet. Linking them needs real sensor data.")