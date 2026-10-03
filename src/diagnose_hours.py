"""Diagnostic: did the time of day of the observations change over time?
Usage: python src/diagnose_hours.py data/GlobalWeatherRepository.csv Europe
"""
import sys
import pandas as pd

cols = ["location_name", "timezone", "last_updated", "temperature_celsius"]
df = pd.read_csv(sys.argv[1], usecols=cols)
cont = sys.argv[2] if len(sys.argv) > 2 else "Europe"
df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
df["ym"] = df["last_updated"].dt.to_period("M")
df["hour"] = df["last_updated"].dt.hour
df["continent"] = df["timezone"].astype(str).str.split("/").str[0]
d = df[df["continent"] == cont].copy()
d["band"] = pd.cut(d["hour"], [-1, 5, 11, 17, 23], labels=["00-05", "06-11", "12-17", "18-23"])

print(f"\n=== {cont}: share of rows by local hour band, per month (%) ===")
sh = d.groupby(["ym", "band"], observed=False).size().unstack(fill_value=0)
print((sh.div(sh.sum(axis=1), axis=0) * 100).round(0).astype(int).to_string())

print(f"\n=== {cont}: mean temperature by local hour band, per month ===")
print(d.pivot_table(index="ym", columns="band", values="temperature_celsius", aggfunc="mean", observed=False).round(1).to_string())
