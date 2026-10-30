import pandas as pd
import numpy as np

n = 5000  # number of readings

# 1. Timestamps: one reading per minute
time = pd.date_range("2026-01-01", periods=n, freq="min")

# 2. Temperature: a slow wave plus a little random noise
temperature = 28 + 4 * np.sin(np.arange(n) / 200) + np.random.normal(0, 0.3, n)

# 3. Pressure: depends a bit on temperature, plus noise
pressure = 1010 + 0.5 * (temperature - 28) + np.random.normal(0, 0.2, n)

# 4. Plant a fake anomaly: pressure jumps up for 100 readings
pressure[3500:3600] += np.random.normal(3, 0.5, 100)

# 5. Put everything into a table and save it
df = pd.DataFrame({"timestamp": time, "pressure": pressure, "temperature": temperature})
df.to_csv("data/fake_sensor_data.csv", index=False)

print("Saved!")
print(df.head())