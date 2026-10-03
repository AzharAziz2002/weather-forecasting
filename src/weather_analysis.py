#!/usr/bin/env python3
"""
Weather Trend Forecasting - Advanced Assessment pipeline
Dataset: Kaggle "Global Weather Repository" (nelgiriyewithana/global-weather-repository)

Usage:
    python src/weather_analysis.py --data data/GlobalWeatherRepository.csv --out outputs

Produces outputs/report.md, outputs/figures/*.png and a few CSV tables.
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
SEED = 42
np.random.seed(SEED)

TEMP, PRECIP = "temperature_celsius", "precip_mm"
REDUNDANT = ["temperature_fahrenheit", "feels_like_fahrenheit", "wind_mph", "pressure_in",
             "precip_in", "visibility_miles", "gust_mph", "last_updated_epoch"]
# Some country names in the dataset are localised (e.g. "Saudi Arabien", "Letonia"); map known ones to English.
COUNTRY_FIX = {"Saudi Arabien": "Saudi Arabia", "Letonia": "Latvia", "Polônia": "Poland", "Turkménistan": "Turkmenistan",
               "Tsjaad": "Chad", "Marrocos": "Morocco", "Bélgica": "Belgium", "Malásia": "Malaysia", "Mexique": "Mexico",
               "Südkorea": "South Korea", "Kazajstán": "Kazakhstan", "Estonie": "Estonia"}
WEATHER_VARS = ["temperature_celsius", "humidity", "wind_kph", "pressure_mb", "precip_mm",
                "cloud", "visibility_km", "uv_index"]


# --------------------------------------------------------------------------- report helper
class Report:
    def __init__(self, out):
        self.out = Path(out)
        self.figdir = self.out / "figures"
        self.figdir.mkdir(parents=True, exist_ok=True)
        self.body, self.insights = [], []

    def h(self, text, level=2):
        self.body += ["", "#" * level + " " + text, ""]

    def p(self, text):
        self.body += [text, ""]

    def insight(self, text):
        self.insights.append(text)
        self.body += ["> **Insight:** " + text, ""]

    def fig(self, fig, name, caption):
        fig.savefig(self.figdir / f"{name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        self.body += [f"![{caption}](figures/{name}.png)", f"*{caption}*", ""]

    def table(self, df, fmt="{:.3f}", index=True):
        d = df.reset_index() if index else df.copy()
        cols = [str(c) for c in d.columns]
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        for _, r in d.iterrows():
            cells = [fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r.values]
            lines.append("| " + " | ".join(cells) + " |")
        self.body += lines + [""]

    def write(self, mission, n_rows, n_locs, span):
        head = ["# Global Weather Trend Forecasting - Advanced Analysis", "",
                "## PM Accelerator Mission", "", "> " + mission, "",
                "## Overview", "",
                f"- Dataset: Global Weather Repository ({n_rows:,} rows after cleaning, {n_locs:,} locations, {span})",
                "- Scope: data cleaning, EDA, anomaly detection, multi-model forecasting with an ensemble, "
                "climate / air-quality / feature-importance / spatial / geographic analyses.", "",
                "## Key insights (computed from the data)", ""]
        head += [f"{i}. {t}" for i, t in enumerate(self.insights, 1)] + [""]
        (self.out / "report.md").write_text("\n".join(head + self.body), encoding="utf-8")


# --------------------------------------------------------------------------- 1. cleaning
def continent_from_tz(tz):
    if not isinstance(tz, str) or "/" not in tz:
        return "Unknown"
    first = tz.split("/")[0]
    return {"America": "Americas", "Australia": "Oceania", "Pacific": "Oceania", "Etc": "Unknown"}.get(first, first)


def mark_consistent_time(df, R):
    """The UTC time at which observations were recorded drifted gradually (about 14:00 in 2024 -> 06:00 in 2026), which shifts
    temperatures by several degrees. Keep rows recorded within +/- tol hours of the most recent observation time, starting from the
    first month after which at least 90% of rows satisfy this."""
    h = df["utc_hour"]
    if h.isna().all():
        df["consistent_time"] = True
        R.tinfo = None
        return
    tol = getattr(R, "time_tol", 3.0)
    recent = h[df["date"] >= df["date"].max() - pd.Timedelta(days=60)].dropna()
    mode = float(recent.round().mode().iloc[0])
    d = (h.fillna(mode) - mode).abs()
    ok = np.minimum(d, 24 - d) <= tol
    ym = df["date"].dt.to_period("M")
    share = ok.groupby(ym).mean()
    bad = share[share < 0.9]
    start = share.index.min() if len(bad) == 0 else bad.index.max() + 1
    if start > share.index.max():  # fallback: no month is clean enough, rely on the hour rule only
        start = share.index.min()
    df["consistent_time"] = ok & (ym >= start)
    R.tinfo = (mode, tol, float(df["consistent_time"].mean()), str(start))


def time_check(df, R):
    R.h("1b. Data quality - time of day of the observations")
    info = getattr(R, "tinfo", None)
    if info is None:
        R.p("- No epoch column available; time-of-day consistency could not be checked.")
        return
    mode, tol, share, start = info
    ym = df["date"].dt.to_period("M")
    sh = df.groupby(ym)["consistent_time"].mean() * 100
    cont = df["continent"].value_counts().index[0]
    d = df[df["continent"] == cont]
    dym = d["date"].dt.to_period("M")
    allm, conm = d.groupby(dym)[TEMP].mean(), d[d["consistent_time"]].groupby(dym)[TEMP].mean()
    fig, ax = plt.subplots(1, 2, figsize=(16, 4.5))
    ax[0].bar(sh.index.to_timestamp(), sh.values, width=25, color="tab:purple"); ax[0].set_ylabel("% of rows"); ax[0].set_title("Rows recorded at the latest observation time (by month)")
    ax[1].plot(allm.index.to_timestamp(), allm.values, marker="o", label="all rows"); ax[1].plot(conm.index.to_timestamp(), conm.values, marker="o", label="consistent-time rows")
    ax[1].set_title(f"Monthly mean temperature, {cont} (C)"); ax[1].legend()
    R.fig(fig, "data_quality_time_of_day", "Observation time of day drifted during collection")
    mh = df.groupby(ym)["utc_hour"].median()
    R.insight(f"Observation time (UTC) drifted gradually from about {mh.iloc[0]:.0f}:00 in {mh.index[0]} to about {mh.iloc[-1]:.0f}:00 in {mh.index[-1]}. "
              f"Temperature depends strongly on time of day, so the forecasting target, seasonal/climate, country and map analyses use only rows recorded within "
              f"+/-{tol:g} h of {mode:02.0f}:00 UTC from {start} onward ({share * 100:.0f}% of all rows).")
    R.p("- **Caveat:** every location is observed at the same UTC time, i.e. at a different LOCAL time of day (e.g. 06:00 UTC is early afternoon in "
        "South-East Asia but night-time in the Americas). Country, continent and map comparisons therefore mix climate with local time of day.")


def load_and_clean(path, R):
    R.h("1. Data cleaning & preprocessing")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(axis=1, how="all")  # drop columns that are entirely empty
    n_raw = len(df)
    df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
    df = df.dropna(subset=["last_updated"])
    df = df.drop_duplicates(subset=["country", "location_name", "last_updated"]).copy()
    if "last_updated_epoch" in df.columns:  # true UTC time of each observation
        _u = pd.to_datetime(df["last_updated_epoch"], unit="s", errors="coerce")
        df["utc_hour"] = _u.dt.hour + _u.dt.minute / 60
    else:
        df["utc_hour"] = np.nan
    df = df.drop(columns=[c for c in REDUNDANT if c in df.columns])
    df["country"] = df["country"].replace(COUNTRY_FIX)
    pd.Series(sorted(df["country"].astype(str).unique()), name="country").to_csv(R.out / "country_names.csv", index=False)
    R.p(f"- Loaded **{n_raw:,}** rows, kept **{len(df):,}** after parsing `last_updated` and removing duplicates.")
    R.p("- Dropped redundant unit-duplicate columns (Fahrenheit, mph, inches, miles, epoch).")

    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if len(miss):
        R.p("**Missing values before imputation:**")
        R.table(miss.rename("missing").to_frame(), fmt="{:.0f}")
    else:
        R.p("- No missing values found.")

    num = [c for c in df.select_dtypes(include="number").columns if c != "utc_hour"]
    for c in num:
        if df[c].isna().any():
            df[c] = df[c].fillna(df.groupby("location_name")[c].transform("median")).fillna(df[c].median())
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].fillna("Unknown")
    R.p("- Numeric gaps filled with the **per-location median** (global median as fallback); categorical gaps with `Unknown`.")

    df["date"] = df["last_updated"].dt.normalize()
    df["month"] = df["last_updated"].dt.month
    df["hour"] = df["last_updated"].dt.hour
    df["continent"] = df["timezone"].map(continent_from_tz)
    df["hemisphere"] = np.where(df["latitude"] >= 0, "Northern", "Southern")
    mark_consistent_time(df, R)

    feats = [c for c in num if c in df.columns]
    q1, q3 = df[feats].quantile(.25), df[feats].quantile(.75)
    iqr = q3 - q1
    out_counts = ((df[feats] < q1 - 1.5 * iqr) | (df[feats] > q3 + 1.5 * iqr)).sum()
    out_tbl = pd.DataFrame({"outliers(IQR 1.5)": out_counts, "share_%": out_counts / len(df) * 100})
    out_tbl = out_tbl[out_tbl["outliers(IQR 1.5)"] > 0].sort_values("outliers(IQR 1.5)", ascending=False).head(12)
    R.p("**Outlier counts (1.5xIQR rule), top columns:**")
    R.table(out_tbl, fmt="{:.2f}")

    lo, hi = df[feats].quantile(.01), df[feats].quantile(.99)
    dfc = df.copy()
    dfc[feats] = df[feats].clip(lo, hi, axis=1)
    R.p("- Outliers are **kept in the raw frame** (needed for anomaly detection) and **winsorised at the 1st/99th percentile** "
        "in a second frame used for correlations and ML.")

    scaled = pd.DataFrame(StandardScaler().fit_transform(dfc[feats]), columns=feats, index=dfc.index)
    R.p(f"- Features standardised (z-score) for ML: mean ~ {scaled.mean().abs().max():.1e}, std ~ {scaled.std().mean():.2f}.")
    return df, dfc, scaled, feats


# --------------------------------------------------------------------------- 2. EDA
def eda(df, dfc, feats, R):
    R.h("2. Exploratory data analysis")
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
    sns.histplot(df[TEMP], bins=60, kde=True, ax=ax[0], color="tab:red"); ax[0].set_title("Temperature (C)")
    sns.histplot(np.log1p(df[PRECIP]), bins=60, ax=ax[1], color="tab:blue"); ax[1].set_title("Precipitation, log(1+mm)")
    sns.histplot(df["humidity"], bins=50, ax=ax[2], color="tab:green"); ax[2].set_title("Humidity (%)")
    R.fig(fig, "eda_distributions", "Distributions of temperature, precipitation and humidity")

    daily = df.groupby("date").agg(temp=(TEMP, "mean"), precip=(PRECIP, "mean"))
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax[0].plot(daily.index, daily["temp"], lw=.8, alpha=.6); ax[0].plot(daily.index, daily["temp"].rolling(7, min_periods=1).mean(), color="tab:red")
    ax[0].set_ylabel("Mean temp (C)"); ax[0].set_title("Global daily mean temperature (blue) and 7-day average (red)")
    ax[1].bar(daily.index, daily["precip"], color="tab:blue", width=1); ax[1].set_ylabel("Mean precip (mm)")
    ax[1].set_title("Global daily mean precipitation")
    R.fig(fig, "eda_time_series", "Global temperature and precipitation over time")

    fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
    sns.boxplot(data=dfc, x="month", y=TEMP, ax=ax[0], color="lightcoral", fliersize=1); ax[0].set_title("Temperature by month")
    pm = df.groupby("month")[PRECIP].mean()
    ax[1].bar(pm.index, pm.values, color="tab:blue"); ax[1].set_title("Mean precipitation by month (mm)"); ax[1].set_xlabel("month")
    R.fig(fig, "eda_seasonality", "Seasonal patterns of temperature and precipitation")

    corr = dfc[feats].corr()
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": .7}); ax.set_title("Correlation matrix (winsorised data)")
    R.fig(fig, "eda_correlation", "Correlation matrix of numeric features")

    ct = corr[TEMP].drop([c for c in [TEMP, "feels_like_celsius"] if c in corr]).sort_values(key=abs, ascending=False)
    top = ", ".join(f"{k} ({v:+.2f})" for k, v in ct.head(3).items())
    R.insight(f"Strongest linear correlates of temperature (excluding feels-like): {top}.")
    wet = df.groupby("month")[PRECIP].mean()
    R.insight(f"Global mean temperature is {df[TEMP].mean():.1f} C; the wettest month on average is {wet.idxmax()} "
              f"({wet.max():.2f} mm) and the driest is {wet.idxmin()} ({wet.min():.2f} mm). "
              f"{(df[PRECIP] == 0).mean() * 100:.0f}% of observations record zero precipitation.")


# --------------------------------------------------------------------------- 3. anomaly detection
def anomalies(df, feats, R):
    R.h("3. Advanced EDA - anomaly detection")
    X = RobustScaler().fit_transform(df[feats])
    iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=SEED, n_jobs=-1)
    df["iso_anomaly"] = iso.fit_predict(X) == -1
    df["iso_score"] = -iso.score_samples(X)

    g = df.groupby(["location_name", "month"])[TEMP]
    std = g.transform("std").replace(0, np.nan)
    df["temp_z"] = ((df[TEMP] - g.transform("mean")) / std).fillna(0)
    df["z_anomaly"] = df["temp_z"].abs() > 3
    R.p(f"- **Isolation Forest** (multivariate, 1% contamination): **{int(df.iso_anomaly.sum()):,}** anomalous records.")
    R.p(f"- **Local z-score** (temperature vs. same location & month, |z|>3): **{int(df.z_anomaly.sum()):,}** records.")

    pca = PCA(2, random_state=SEED).fit_transform(X[::max(1, len(X) // 40000)])
    flag = df["iso_anomaly"].values[::max(1, len(X) // 40000)]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].scatter(pca[~flag, 0], pca[~flag, 1], s=2, alpha=.3, label="normal")
    ax[0].scatter(pca[flag, 0], pca[flag, 1], s=8, color="red", label="anomaly"); ax[0].legend(); ax[0].set_title("Isolation Forest anomalies in PCA space")
    daily = df.groupby("date")["iso_anomaly"].sum()
    ax[1].plot(daily.index, daily.values); ax[1].set_title("Anomalous records per day"); ax[1].tick_params(axis="x", rotation=30)
    R.fig(fig, "anomaly_overview", "Anomaly overview (PCA projection and daily counts)")

    an = df[df.iso_anomaly]
    comp = pd.DataFrame({"anomalies_median": an[WEATHER_VARS].median(), "normal_median": df.loc[~df.iso_anomaly, WEATHER_VARS].median()})
    R.p("**Median weather conditions: anomalies vs. normal records**")
    R.table(comp)
    top_c = an["country"].value_counts().head(10)
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
    top_c[::-1].plot.barh(ax=ax[0], color="tab:red"); ax[0].set_title("Countries with most anomalies (Isolation Forest)")
    sns.histplot(df["temp_z"].clip(-8, 8), bins=80, ax=ax[1]); ax[1].set_title("Local temperature z-scores")
    R.fig(fig, "anomaly_countries", "Where anomalies occur and local z-score distribution")

    cols = ["country", "location_name", "last_updated", TEMP, PRECIP, "temp_z", "iso_score", "condition_text"]
    df.sort_values("iso_score", ascending=False).head(50)[cols].to_csv(R.out / "top_anomalies.csv", index=False)
    R.p("- The 50 most anomalous records are saved to `top_anomalies.csv`.")
    cond = an["condition_text"].value_counts(normalize=True).head(3)
    R.insight(f"{df.iso_anomaly.mean() * 100:.1f}% of records are multivariate anomalies; they cluster in {top_c.index[0]} "
              f"({top_c.iloc[0]} records). Their most common conditions: "
              + ", ".join(f"{k} ({v * 100:.0f}%)" for k, v in cond.items()) + ".")


# --------------------------------------------------------------------------- 4. forecasting
def core_panel(df):
    """Locations observed on >=90% of days, after dropping incomplete days (e.g. today's partial update).
    The repository keeps adding cities, so averaging over all rows mixes in composition changes."""
    per_day = df.groupby("date")["location_name"].nunique()
    full_days = per_day[per_day >= 0.5 * per_day.median()].index
    d = df[df["date"].isin(full_days)]
    cov = d.groupby("location_name")["date"].nunique() / max(1, len(full_days))
    core = cov[cov >= 0.9].index
    if len(core) < 20:  # fallback: not enough stable locations
        return d, d["location_name"].nunique(), len(per_day) - len(full_days)
    return d[d["location_name"].isin(core)], len(core), len(per_day) - len(full_days)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    mape = np.nanmean(np.abs((y - p) / np.where(y == 0, np.nan, y))) * 100
    smape = np.mean(2 * np.abs(y - p) / (np.abs(y) + np.abs(p) + 1e-9)) * 100
    return dict(MAE=mean_absolute_error(y, p), RMSE=np.sqrt(mean_squared_error(y, p)), MAPE=mape, sMAPE=smape, R2=r2_score(y, p))


def fourier(idx, K):
    doy = idx.dayofyear.values + idx.hour.values / 24.0
    cols = {}
    for k in range(1, K + 1):
        cols[f"s{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        cols[f"c{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    return pd.DataFrame(cols, index=idx)


def trend(idx):
    return (idx - pd.Timestamp("2000-01-01")).days.values / 365.25


def m_naive(y, fi):
    return np.repeat(y.iloc[-1], len(fi))


def m_snaive(y, fi):
    return np.resize(y.iloc[-7:].values, len(fi))


def m_ridge(y, fi):
    def X(i):
        d = fourier(i, 3); d["t"] = trend(i); return d
    return Ridge(alpha=1.0).fit(X(y.index), y.values).predict(X(fi))


def m_sarimax(y, fi):
    res = SARIMAX(y.values, exog=fourier(y.index, 2).values, order=(2, 0, 1), trend="c",
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200)
    return np.asarray(res.forecast(len(fi), exog=fourier(fi, 2).values))


def m_ets(y, fi):
    return np.asarray(ExponentialSmoothing(y.values, trend="add", damped_trend=True,
                                           initialization_method="estimated").fit().forecast(len(fi)))


LAGS = [1, 2, 3, 7, 14]


def _row(hist, frow):
    return list(hist[-l] for l in LAGS) + [np.mean(hist[-7:]), np.mean(hist[-14:])] + list(frow)


def make_lag_model(factory):
    def f(y, fi):
        v = y.values.astype(float)
        F, Ff = fourier(y.index, 2).values, fourier(fi, 2).values
        Xtr = np.array([_row(v[:i], F[i]) for i in range(14, len(v))])
        model = factory().fit(Xtr, v[14:])
        hist, preds = list(v), []
        for j in range(len(fi)):
            p = model.predict(np.array([_row(np.array(hist), Ff[j])]))[0]
            preds.append(p); hist.append(p)
        return np.array(preds)
    return f


MODELS = {
    "naive": m_naive,
    "seasonal_naive_7": m_snaive,
    "harmonic_ridge": m_ridge,
    "sarimax": m_sarimax,
    "ets_damped": m_ets,
    "random_forest": make_lag_model(lambda: RandomForestRegressor(300, min_samples_leaf=2, random_state=SEED, n_jobs=-1)),
    "gradient_boosting": make_lag_model(lambda: GradientBoostingRegressor(random_state=SEED)),
}
BASELINES = {"naive", "seasonal_naive_7"}


def forecasting(df, R, horizon=14):
    R.h("4. Forecasting with multiple models and an ensemble")
    panel, n_core, n_drop = core_panel(df)
    d0 = panel.groupby("date")[TEMP].mean()
    gaps = d0.index.to_series().diff().dt.days
    brk = gaps[gaps > 3].index
    if len(brk):
        d0 = d0.loc[brk[-1]:]  # keep only the last contiguous stretch of consistent-time data
    s = d0.asfreq("D").interpolate(limit_direction="both")
    R.p("- Only rows observed at a consistent time of day are used (see section 1b).")
    R.p(f"- Stable panel: the target averages **{n_core} locations** observed on >=90% of days; {n_drop} incomplete day(s) were dropped. "
        "This avoids artificial jumps caused by cities being added to the repository over time.")
    H = min(horizon, max(3, len(s) // 6))
    R.p(f"- Target: **global daily mean temperature** (mean over the stable panel, `last_updated` as time index), {len(s)} days "
        f"({s.index.min().date()} to {s.index.max().date()}).")
    R.p(f"- Split: last **{H} days = test**, the {H} days before that = **validation** (used only to weight the ensemble); "
        "all models are refit on train+validation before predicting the test window.")
    train_full, test = s.iloc[:-H], s.iloc[-H:]
    inner, val = train_full.iloc[:-H], train_full.iloc[-H:]

    val_pred, test_pred = {}, {}
    for name, f in MODELS.items():
        try:
            val_pred[name] = f(inner, val.index)
            test_pred[name] = f(train_full, test.index)
        except Exception as e:  # keep pipeline alive if a model fails
            R.p(f"- Model `{name}` failed and was skipped: {e}")
    members = [m for m in test_pred if m not in BASELINES]
    rm_val = {m: metrics(val.values, val_pred[m])["RMSE"] for m in members}
    w = {m: 1 / rm_val[m] for m in members}; tot = sum(w.values()); w = {m: v / tot for m, v in w.items()}
    top3 = sorted(members, key=rm_val.get)[:3]
    test_pred["ensemble_average"] = np.mean([test_pred[m] for m in members], axis=0)
    test_pred["ensemble_weighted"] = np.sum([w[m] * test_pred[m] for m in members], axis=0)
    test_pred["ensemble_top3"] = np.mean([test_pred[m] for m in top3], axis=0)
    val_pred["ensemble_weighted"] = np.sum([w[m] * val_pred[m] for m in members], axis=0)

    res = pd.DataFrame({k: metrics(test.values, v) for k, v in test_pred.items()}).T.sort_values("RMSE")
    res.to_csv(R.out / "model_metrics.csv")
    R.p("**Test-set metrics (lower is better, except R2):**")
    R.table(res)
    R.p("**Ensemble weights (inverse validation RMSE):** " + ", ".join(f"{m}={w[m]:.2f}" for m in members) +
        f". `ensemble_top3` averages: {', '.join(top3)}.")

    hist = s.iloc[-(H * 4):]
    fig, ax = plt.subplots(1, 2, figsize=(17, 5))
    ax[0].plot(hist.index, hist.values, color="black", lw=2, label="actual")
    for k, v in test_pred.items():
        if k.startswith("ensemble"):
            ax[0].plot(test.index, v, lw=2.5, ls="--", label=k)
        else:
            ax[0].plot(test.index, v, lw=1, alpha=.6, label=k)
    ax[0].axvline(test.index[0], color="grey", ls=":"); ax[0].set_title("Hold-out forecasts vs actual"); ax[0].legend(fontsize=7, ncol=2)
    ax[0].tick_params(axis="x", rotation=30)
    res["RMSE"].sort_values()[::-1].plot.barh(ax=ax[1], color=["tab:orange" if i.startswith("ensemble") else "tab:blue" for i in res["RMSE"].sort_values()[::-1].index])
    ax[1].set_title("RMSE by model (orange = ensembles)")
    R.fig(fig, "forecast_comparison", "Model comparison on the hold-out window")

    best = res.index[0]
    best_single = res.loc[[i for i in res.index if not i.startswith("ensemble")]].index[0]
    d = (1 - res.loc[best, "RMSE"] / res.loc[best_single, "RMSE"]) * 100
    R.insight(f"Best test model: **{best}** (RMSE {res.loc[best, 'RMSE']:.3f} C, MAE {res.loc[best, 'MAE']:.3f} C). "
              f"Best single model: {best_single} (RMSE {res.loc[best_single, 'RMSE']:.3f} C)"
              + (f"; the ensemble improves RMSE by {d:.1f}%." if best.startswith("ensemble") else "; the ensemble did not beat it on this window."))

    # genuine future forecast with the weighted ensemble
    fi = pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=H, freq="D")
    fut = np.sum([w[m] * MODELS[m](s, fi) for m in members], axis=0)
    sd = np.std(val.values - val_pred["ensemble_weighted"])
    pd.DataFrame({"date": fi, "forecast_c": fut, "lower": fut - 1.96 * sd, "upper": fut + 1.96 * sd}).to_csv(R.out / "future_forecast.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    h2 = s.iloc[-90:]
    ax.plot(h2.index, h2.values, label="history"); ax.plot(fi, fut, color="tab:red", label="ensemble forecast")
    ax.fill_between(fi, fut - 1.96 * sd, fut + 1.96 * sd, color="tab:red", alpha=.2, label="approx. 95% band (validation residuals)")
    ax.set_title(f"{H}-day-ahead forecast of global mean temperature"); ax.legend()
    R.fig(fig, "forecast_future", "Out-of-sample forecast beyond the last observed date")


# --------------------------------------------------------------------------- 5. unique analyses
def climate(df, R):
    R.h("5. Climate analysis - patterns by region")
    cm = df.groupby(["continent", "month"])[TEMP].mean().unstack()
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    sns.heatmap(cm, cmap="coolwarm", annot=True, fmt=".0f", ax=ax[0]); ax[0].set_title("Mean temperature by continent and month (C)")
    for c in cm.index:
        ax[1].plot(cm.columns, cm.loc[c], marker="o", label=c)
    ax[1].legend(fontsize=8); ax[1].set_title("Seasonal cycle by continent"); ax[1].set_xlabel("month")
    R.fig(fig, "climate_continent_month", "Monthly climatology by continent")

    big = df["country"].value_counts()
    mc = df[df["country"].isin(big[big >= 100].index)].groupby(["country", "month"])[TEMP].mean().unstack()
    full = mc[mc.notna().sum(axis=1) >= 10]  # prefer countries with (almost) a full seasonal cycle
    mc = full if len(full) >= 10 else mc
    rng_c = (mc.max(axis=1) - mc.min(axis=1)).sort_values(ascending=False)
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
    rng_c.head(10)[::-1].plot.barh(ax=ax[0], color="tab:red"); ax[0].set_title("Largest seasonal temperature range (C)")
    rng_c.tail(10).plot.barh(ax=ax[1], color="tab:blue"); ax[1].set_title("Smallest seasonal temperature range (C)")
    R.fig(fig, "climate_seasonal_range", "Seasonal range of monthly mean temperature by country")

    panel, _, _ = core_panel(df)
    panel = panel.copy()
    panel["ym"] = panel["date"].dt.to_period("M")
    span_m = panel["ym"].nunique()
    if span_m < 24:
        R.p(f"- **Long-term trends cannot be estimated:** the consistent-time data covers only {span_m} months (less than two full years), "
            "so there is no repeated calendar month to compare against. Climate analysis is therefore limited to the seasonal cycle and regional differences above.")
        R.insight(f"Seasonal cycles dominate: {rng_c.index[0]} has the largest annual temperature swing ({rng_c.iloc[0]:.1f} C). "
                  f"With only {span_m} months of consistent-time data, climate TRENDS cannot be estimated, so none are reported.")
        return
    # each location is compared with ITSELF in the same calendar month -> immune to cities being added over time
    panel["t_anom"] = panel[TEMP] - panel.groupby(["location_name", "month"])[TEMP].transform("mean")
    R.p(f"- Trend analysis uses the stable panel ({panel['location_name'].nunique()} locations) and anomalies relative to each location's own monthly mean.")
    ma = panel.groupby(["ym", "continent"])["t_anom"].mean().unstack()
    ma = ma.iloc[1:-1] if len(ma) > 3 else ma  # drop partial first/last months
    fig, ax = plt.subplots(figsize=(13, 4.5))
    slopes = {}
    for c in ma.columns:
        y = ma[c].dropna()
        if len(y) >= 4:
            slopes[c] = np.polyfit(np.arange(len(y)), y.values, 1)[0] * 12
        ax.plot(y.index.to_timestamp(), y.values, label=c)
    ax.axhline(0, color="k", lw=.5); ax.legend(fontsize=8, ncol=3); ax.set_title("Monthly temperature anomaly vs. each location's own monthly mean (C)")
    R.fig(fig, "climate_anomaly_trend", "Deseasonalised monthly temperature anomaly by continent")
    if slopes:
        sl = pd.Series(slopes).sort_values(ascending=False)
        R.p("**Linear trend of the deseasonalised anomaly (C per year):**")
        R.table(sl.rename("C_per_year").to_frame())
        R.insight(f"Seasonal cycles dominate: {rng_c.index[0]} has the largest annual temperature swing ({rng_c.iloc[0]:.1f} C). "
                  f"Fitted anomaly trends over {span_m} months ({sl.min():+.2f} to {sl.max():+.2f} C/yr) are indicative only, not evidence of long-term climate change.")


def air_quality(df, dfc, feats, R):
    R.h("6. Environmental impact - air quality vs weather")
    aq = [c for c in feats if c.startswith("air_quality_")]
    wv = [c for c in WEATHER_VARS if c in dfc.columns]
    cor = dfc[aq + wv].corr(method="spearman").loc[aq, wv]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.heatmap(cor, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax); ax.set_title("Spearman correlation: air quality vs weather")
    R.fig(fig, "aq_correlation", "Air-quality pollutants vs weather variables")

    pm = "air_quality_PM2.5"
    s = dfc.sample(min(20000, len(dfc)), random_state=SEED)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for a, v in zip(ax[:2], ["humidity", "wind_kph"]):
        a.scatter(s[v], np.log1p(s[pm]), s=3, alpha=.25); a.set_xlabel(v); a.set_ylabel("log(1+PM2.5)"); a.set_title(f"PM2.5 vs {v}")
    sns.boxplot(data=df.assign(l=np.log1p(df[pm])), x="continent", y="l", ax=ax[2], fliersize=1); ax[2].tick_params(axis="x", rotation=40); ax[2].set_title("log(1+PM2.5) by continent")
    R.fig(fig, "aq_scatter", "PM2.5 relationships and continental distribution")

    cnt = df["country"].value_counts()
    top = df[df["country"].isin(cnt[cnt >= 100].index)].groupby("country")[pm].median().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5)); top.head(12)[::-1].plot.barh(ax=ax, color="tab:brown"); ax.set_title("Median PM2.5 by country (top 12)")
    R.fig(fig, "aq_top_countries", "Most polluted countries by median PM2.5")

    rainy = df.loc[df[PRECIP] > 0, pm].median(); dry = df.loc[df[PRECIP] == 0, pm].median()
    best = cor.loc[pm].drop("temperature_celsius", errors="ignore").abs().idxmax()
    R.insight(f"PM2.5 is most associated with **{best}** (Spearman {cor.loc[pm, best]:+.2f}) among weather variables; "
              f"median PM2.5 is {rainy:.1f} on rainy records vs {dry:.1f} on dry ones. Most polluted country by median PM2.5: {top.index[0]} ({top.iloc[0]:.1f}).")


def feature_importance(dfc, feats, R):
    R.h("7. Feature importance (target: temperature)")
    cand = [f for f in feats if f not in (TEMP, "feels_like_celsius")] + ["month", "hour"]
    d = dfc.sample(min(40000, len(dfc)), random_state=SEED)
    X, y = d[cand], d[TEMP]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED)
    rf = RandomForestRegressor(200, min_samples_leaf=3, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    r2 = r2_score(yte, rf.predict(Xte))
    sub = Xte.iloc[:8000]
    perm = permutation_importance(rf, sub, yte.loc[sub.index], n_repeats=5, random_state=SEED, n_jobs=-1).importances_mean
    mi = mutual_info_regression(Xtr.iloc[:15000], ytr.iloc[:15000], random_state=SEED)
    tbl = pd.DataFrame({
        "pearson": X.corrwith(y).abs(), "spearman": X.corrwith(y, method="spearman").abs(),
        "mutual_info": pd.Series(mi, index=cand), "rf_impurity": pd.Series(rf.feature_importances_, index=cand),
        "rf_permutation": pd.Series(perm, index=cand)})
    norm = tbl / tbl.max()
    norm["consensus_rank"] = tbl.rank(ascending=False).mean(axis=1)
    norm = norm.sort_values("consensus_rank")
    norm.to_csv(R.out / "feature_importance.csv")
    R.p(f"Random-forest hold-out R2 = **{r2:.3f}** (random split; location identity is partly encoded by latitude/longitude).")
    R.table(norm.head(10).round(3))
    fig, ax = plt.subplots(figsize=(12, 7))
    norm.drop(columns="consensus_rank").head(15)[::-1].plot.barh(ax=ax, width=.8); ax.set_title("Top 15 features, five methods (scaled to 1)")
    R.fig(fig, "feature_importance", "Feature importance by method")
    R.insight(f"Top features for temperature by consensus rank: {', '.join(norm.index[:3])} (RF hold-out R2 {r2:.2f}).")


def spatial(df, R):
    R.h("8. Spatial analysis")
    pm = "air_quality_PM2.5"
    loc = df.groupby("location_name").agg(lat=("latitude", "first"), lon=("longitude", "first"), temp=(TEMP, "mean"),
                                          precip=(PRECIP, "mean"), pm25=(pm, "median"), anom=("iso_anomaly", "sum"))
    fig, ax = plt.subplots(2, 2, figsize=(17, 9.5))
    for a, (col, cmap, title, tf) in zip(ax.ravel(), [("temp", "coolwarm", "Mean temperature (C)", None), ("precip", "Blues", "Mean precipitation, log(1+mm)", np.log1p),
                                                      ("pm25", "magma_r", "Median PM2.5, log(1+ug/m3)", np.log1p), ("anom", "Reds", "Anomaly count (Isolation Forest)", None)]):
        v = loc[col] if tf is None else tf(loc[col])
        sc = a.scatter(loc["lon"], loc["lat"], c=v, cmap=cmap, s=22, edgecolor="k", linewidth=.2)
        a.set_title(title); a.set_xlabel("longitude"); a.set_ylabel("latitude"); a.set_xlim(-180, 180); a.set_ylim(-60, 80); plt.colorbar(sc, ax=a)
    R.fig(fig, "spatial_maps", "Geographic distribution of temperature, precipitation, PM2.5 and anomalies (lon/lat scatter)")

    r = np.corrcoef(loc["lat"].abs(), loc["temp"])[0, 1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(loc["lat"], loc["temp"], s=14, alpha=.6); ax.set_xlabel("latitude"); ax.set_ylabel("mean temp (C)"); ax.set_title("Temperature vs latitude")
    R.fig(fig, "spatial_lat_temp", "Mean temperature versus latitude")
    R.insight(f"Mean temperature falls with distance from the equator (corr with |latitude| = {r:+.2f}).")


def geography(df, R):
    R.h("9. Geographical patterns - countries and continents")
    fig, ax = plt.subplots(2, 2, figsize=(16, 9))
    for a, v in zip(ax.ravel(), [TEMP, "humidity", "wind_kph", "cloud"]):
        sns.boxplot(data=df, x="continent", y=v, ax=a, fliersize=1); a.tick_params(axis="x", rotation=30); a.set_title(f"{v} by continent")
    R.fig(fig, "geo_continent_boxplots", "Weather variable distributions by continent")

    cs = df.groupby("country").agg(temp=(TEMP, "mean"), humidity=("humidity", "mean"), precip=(PRECIP, "mean"), wind=("wind_kph", "mean"), n=(TEMP, "size"))
    cs = cs[cs["n"] >= 100].drop(columns="n")  # ignore countries with only a handful of stray rows
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for a, (col, color) in zip(ax, [("temp", "tab:red"), ("humidity", "tab:green"), ("precip", "tab:blue")]):
        s = cs[col].sort_values(); pd.concat([s.head(8), s.tail(8)]).plot.barh(ax=a, color=color); a.set_title(f"Lowest / highest mean {col}")
    R.fig(fig, "geo_country_extremes", "Country extremes in temperature, humidity and precipitation")
    cs.sort_values("temp").to_csv(R.out / "country_summary.csv")

    hem = df.groupby(["hemisphere", "month"])[TEMP].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(9, 4.5)); hem.plot(marker="o", ax=ax); ax.set_title("Seasonal cycle by hemisphere")
    R.fig(fig, "geo_hemispheres", "Northern vs southern hemisphere seasonal cycle")
    wetc = cs["precip"].idxmax(); hot = cs["temp"].idxmax(); cold = cs["temp"].idxmin()
    R.insight(f"Hottest country on average: {hot} ({cs.loc[hot, 'temp']:.1f} C); coldest: {cold} ({cs.loc[cold, 'temp']:.1f} C); "
              f"highest mean precipitation: {wetc} ({cs.loc[wetc, 'precip']:.2f} mm). Northern and southern hemisphere seasonal cycles are in opposite phase.")


# --------------------------------------------------------------------------- main
def read_mission(path):
    if Path(path).exists():
        lines = [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
        if lines:
            return " ".join(lines)
    return "PM Accelerator mission - see https://www.pmaccelerator.io/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to GlobalWeatherRepository.csv")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--mission", default="MISSION.txt")
    ap.add_argument("--horizon", type=int, default=14)
    ap.add_argument("--time-tol", type=float, default=3.0, help="hours of tolerance around the latest UTC observation time")
    a = ap.parse_args()
    R = Report(a.out)
    R.time_tol = a.time_tol
    t0 = time.time()
    df, dfc, scaled, feats = load_and_clean(a.data, R); print("cleaned", round(time.time() - t0), "s")
    time_check(df, R)
    eda(df, dfc, feats, R); print("eda")
    anomalies(df, feats, R); print("anomalies")
    dfs = df[df["consistent_time"]].copy()  # rows observed at a consistent time of day
    forecasting(dfs, R, a.horizon); print("forecasting")
    climate(dfs, R); print("climate")
    air_quality(df, dfc, feats, R); print("air quality")
    feature_importance(dfc, feats, R); print("feature importance")
    spatial(dfs, R); print("spatial")
    geography(dfs, R); print("geography")
    span = f"{df['date'].min().date()} to {df['date'].max().date()}"
    R.write(read_mission(a.mission), len(df), df["location_name"].nunique(), span)
    print(f"Done in {time.time() - t0:.0f}s -> {Path(a.out) / 'report.md'}")


if __name__ == "__main__":
    main()
