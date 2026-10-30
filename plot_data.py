import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the CSV file into a table
df = pd.read_csv("data/fake_sensor_data.csv", parse_dates=["timestamp"])

# 2. Check that it loaded correctly
print(df.shape)
print(df.describe())

# 3. Draw the pressure graph
plt.figure(figsize=(12, 5))
plt.plot(df["timestamp"], df["pressure"])
plt.xlabel("Time")
plt.ylabel("Pressure (hPa)")
plt.title("Fake sensor pressure")
plt.show()