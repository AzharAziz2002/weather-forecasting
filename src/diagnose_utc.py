"""Diagnostic: at what UTC time were observations recorded, month by month?
Usage: python src/diagnose_utc.py data/GlobalWeatherRepository.csv
"""
import sys
import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1], usecols=["last_updated_epoch"])
u = pd.to_datetime(df["last_updated_epoch"], unit="s")
d = pd.DataFrame({"ym": u.dt.to_period("M"), "h": u.dt.hour + u.dt.minute / 60})
ref = 6.0
dist = np.minimum((d["h"] - ref).abs(), 24 - (d["h"] - ref).abs())
d["w1"], d["w2"], d["w3"] = dist <= 1, dist <= 2, dist <= 3
g = d.groupby("ym")
out = pd.DataFrame({"rows": g.size(), "median_utc_hour": g["h"].median().round(1),
                    "pct_within_1h_of_06": (g["w1"].mean() * 100).round(0),
                    "pct_within_2h": (g["w2"].mean() * 100).round(0),
                    "pct_within_3h": (g["w3"].mean() * 100).round(0)})
print(out.to_string())
