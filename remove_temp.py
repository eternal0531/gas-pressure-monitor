import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Load the data
df = pd.read_csv("data/fake_sensor_data.csv", parse_dates=["timestamp"])

# 2. Learn the temperature -> pressure relationship
X = df[["temperature"]]   # input (double brackets = keep it as a table)
y = df["pressure"]        # what we want to predict

model = LinearRegression()
model.fit(X, y)

print("slope:", model.coef_[0])
print("intercept:", model.intercept_)

# 3. Calculate expected pressure and the residual
df["pressure_expected"] = model.predict(X)
df["pressure_residual"] = y - df["pressure_expected"]

# 4. Save the result for the next stage
df.to_csv("data/fake_sensor_data_residual.csv", index=False)

# 5. Plot raw vs residual, one above the other
fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

top.plot(df["timestamp"], df["pressure"], label="actual")
top.plot(df["timestamp"], df["pressure_expected"], label="expected from temperature")
top.set_ylabel("Pressure (hPa)")
top.set_title("Before: raw pressure")
top.legend()

bottom.plot(df["timestamp"], df["pressure_residual"], color="green")
bottom.set_ylabel("Residual (hPa)")
bottom.set_title("After: pressure with temperature effect removed")
bottom.set_xlabel("Time")

plt.tight_layout()
plt.show()