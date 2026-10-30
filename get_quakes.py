import pandas as pd

url = (
    "https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv"
    "&starttime=2025-01-01&endtime=2025-12-31"
    "&minmagnitude=4"
    "&minlatitude=6&maxlatitude=37"
    "&minlongitude=68&maxlongitude=98"
)

quakes = pd.read_csv(url, parse_dates=["time"])
quakes = quakes[["time", "latitude", "longitude", "mag", "place"]]
quakes.to_csv("data/quakes.csv", index=False)

print(quakes.shape)
print(quakes.head(10))
print("Magnitude range:", quakes["mag"].min(), "to", quakes["mag"].max())