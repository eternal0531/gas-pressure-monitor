import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# 1. Load the residual data from Stage 3
df = pd.read_csv("data/fake_sensor_data_residual.csv", parse_dates=["timestamp"])

# 2. Create features from recent behaviour
df["roll_mean"] = df["pressure_residual"].rolling(30).mean()
df["roll_std"] = df["pressure_residual"].rolling(30).std()
df["rate"] = df["pressure_residual"].diff()
df = df.dropna()   # the first rows can't have a 30-minute window

# 3. Train the Isolation Forest and flag anomalies
features = ["pressure_residual", "roll_mean", "roll_std", "rate"]
iso = IsolationForest(contamination=0.02, random_state=42)
df["anomaly"] = iso.fit_predict(df[features])   # -1 = anomaly, 1 = normal

# 4. Check the result against the truth we planted
truth = (df.index >= 3500) & (df.index < 3600)
flagged = df["anomaly"] == -1
print("Total flagged:", flagged.sum())
print("Planted anomaly readings caught:", (truth & flagged).sum(), "out of", truth.sum())
print("False alarms (flagged but normal):", (~truth & flagged).sum())

# 5. Save and plot
df.to_csv("data/fake_sensor_data_anomalies.csv", index=False)

plt.figure(figsize=(12, 5))
plt.plot(df["timestamp"], df["pressure_residual"], color="green", label="residual")
plt.scatter(df["timestamp"][flagged], df["pressure_residual"][flagged],
            color="red", s=10, label="flagged anomaly")
plt.xlabel("Time")
plt.ylabel("Residual (hPa)")
plt.title("Anomalies found by Isolation Forest")
plt.legend()
plt.show()