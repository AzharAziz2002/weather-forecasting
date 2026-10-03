"""Generate a SYNTHETIC dataset with the same schema as the Kaggle 'Global Weather Repository'.
Only for smoke-testing the pipeline. Real results require the real CSV."""
import numpy as np, pandas as pd, sys
rng = np.random.default_rng(0)
tzs = ["Asia/Kabul", "Europe/Paris", "America/New_York", "Africa/Cairo", "Australia/Sydney",
       "Asia/Karachi", "America/Sao_Paulo", "Europe/Moscow", "Africa/Nairobi", "Pacific/Auckland"]
n_loc, n_days = 120, 420
locs = []
for i in range(n_loc):
    tz = tzs[i % len(tzs)]
    lat = rng.uniform(-45, 60); lon = rng.uniform(-120, 150)
    locs.append((f"Country{i}", f"City{i}", tz, lat, lon, rng.normal(0, 3)))
days = pd.date_range("2024-05-16", periods=n_days, freq="D")
rows = []
for (c, city, tz, lat, lon, off) in locs:
    doy = days.dayofyear.values
    season = -np.sign(lat) * 12 * np.cos(2 * np.pi * (doy - 15) / 365.25) * (abs(lat) / 60 + .2)
    t = 28 - 0.45 * abs(lat) + season + off + rng.normal(0, 2.0, n_days)
    hum = np.clip(65 + rng.normal(0, 15, n_days) - 0.3 * (t - 20), 5, 100)
    precip = np.where(rng.random(n_days) < 0.25, rng.exponential(3, n_days), 0.0)
    wind = np.abs(rng.normal(12, 6, n_days))
    pm25 = np.exp(rng.normal(2.3, .8, n_days)) * (1 + 0.01 * (30 - wind)) / (1 + 0.1 * precip)
    for d, tt, h, p, w, pm in zip(days, t, hum, precip, wind, pm25):
        rows.append(dict(country=c, location_name=city, region="", latitude=lat, longitude=lon, timezone=tz,
            last_updated_epoch=int(d.timestamp()), last_updated=(d + pd.Timedelta(hours=int(rng.integers(0, 24)))).strftime("%Y-%m-%d %H:%M"),
            temperature_celsius=tt, temperature_fahrenheit=tt * 9 / 5 + 32, condition_text="Rain" if p > 0 else "Sunny",
            wind_mph=w / 1.609, wind_kph=w, wind_degree=int(rng.integers(0, 360)), wind_direction="N",
            pressure_mb=1013 + rng.normal(0, 6), pressure_in=29.9, precip_mm=p, precip_in=p / 25.4, humidity=h,
            cloud=int(np.clip(h + rng.normal(0, 20), 0, 100)), feels_like_celsius=tt - 0.05 * w, feels_like_fahrenheit=0,
            visibility_km=np.clip(10 - 0.05 * pm + rng.normal(0, 1), 0.5, 10), visibility_miles=6, uv_index=max(0, rng.normal(5, 3)),
            gust_mph=w, gust_kph=w * 1.5, **{"air_quality_Carbon_Monoxide": pm * 20 + 200, "air_quality_Ozone": rng.normal(60, 20),
            "air_quality_Nitrogen_dioxide": pm * .5, "air_quality_Sulphur_dioxide": pm * .2, "air_quality_PM2.5": pm,
            "air_quality_PM10": pm * 1.4, "air_quality_us-epa-index": int(np.clip(pm // 12 + 1, 1, 6)), "air_quality_gb-defra-index": int(np.clip(pm // 10 + 1, 1, 10))},
            moon_phase="Full Moon", moon_illumination=int(rng.integers(0, 100))))
df = pd.DataFrame(rows)
for col in ["humidity", "pressure_mb", "air_quality_Ozone"]:
    df.loc[df.sample(frac=0.01, random_state=1).index, col] = np.nan
df.loc[df.sample(n=40, random_state=2).index, "temperature_celsius"] += 25
df.loc[df.sample(n=40, random_state=3).index, "precip_mm"] = 300
out = sys.argv[1] if len(sys.argv) > 1 else "data/sample_weather.csv"
df.to_csv(out, index=False); print("wrote", out, df.shape)
