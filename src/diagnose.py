"""Diagnostic: why do some regional temperature trends look implausible, and why is a country's mean odd?
Usage: python src/diagnose.py data/GlobalWeatherRepository.csv Europe
"""
import sys
import pandas as pd

cols = ["location_name", "country", "timezone", "last_updated", "temperature_celsius"]
df = pd.read_csv(sys.argv[1], usecols=cols)
cont = sys.argv[2] if len(sys.argv) > 2 else "Europe"
df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
df["date"] = df["last_updated"].dt.normalize()
df["ym"] = df["date"].dt.to_period("M")
df["year"], df["month"] = df["date"].dt.year, df["date"].dt.month
df["continent"] = df["timezone"].astype(str).str.split("/").str[0]
d = df[df["continent"] == cont].copy()
d["anom"] = d["temperature_celsius"] - d.groupby(["location_name", "month"])["temperature_celsius"].transform("mean")

print(f"\n=== {cont}: per month ===")
t = d.groupby("ym").agg(rows=("anom", "size"), locations=("location_name", "nunique"),
                        mean_temp=("temperature_celsius", "mean"), mean_anomaly=("anom", "mean"))
print(t.round(2).to_string())

print(f"\n=== {cont}: mean temperature, year x month (same months compared across years) ===")
print(d.pivot_table(index="year", columns="month", values="temperature_celsius", aggfunc="mean").round(1).to_string())

print("\n=== 6 hottest countries by mean temp: coverage ===")
c = df.groupby("country").agg(mean_temp=("temperature_celsius", "mean"), rows=("date", "size"), days=("date", "nunique"),
                              locations=("location_name", "nunique"), first=("date", "min"), last=("date", "max"))
print(c.sort_values("mean_temp", ascending=False).head(6).round(1).to_string())
