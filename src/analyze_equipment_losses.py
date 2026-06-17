from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "russia_losses_equipment.json"
OUT = ROOT / "outputs"
FIG = OUT / "figures"


PRIMARY_COLUMNS = [
    "aircraft",
    "helicopter",
    "tank",
    "APC",
    "field artillery",
    "MRL",
    "anti-aircraft warfare",
    "drone",
    "naval ship",
    "cruise missiles",
    "special equipment",
    "ground robotic systems",
    "submarines",
]

HEAVY_COLUMNS = [
    "aircraft",
    "helicopter",
    "tank",
    "APC",
    "field artillery",
    "MRL",
    "anti-aircraft warfare",
    "naval ship",
    "special equipment",
]


def ensure_dirs() -> None:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def load_equipment() -> tuple[pd.DataFrame, dict]:
    raw = pd.DataFrame(json.loads(DATA_PATH.read_text(encoding="utf-8")))
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values("date").reset_index(drop=True)

    # Schema changed in 2022: military auto + fuel tank were merged into vehicles and fuel tanks.
    raw["vehicles_fuel_combined"] = raw["vehicles and fuel tanks"]
    early = raw["vehicles_fuel_combined"].isna()
    raw.loc[early, "vehicles_fuel_combined"] = (
        raw.loc[early, "military auto"].fillna(0) + raw.loc[early, "fuel tank"].fillna(0)
    )

    model_cols = PRIMARY_COLUMNS + ["vehicles_fuel_combined"]
    heavy_cols = HEAVY_COLUMNS + ["vehicles_fuel_combined"]
    cumulative = raw.set_index("date")[model_cols].ffill().fillna(0)
    daily = cumulative.diff().fillna(cumulative)
    neg_count = int((daily < 0).sum().sum())
    min_delta = float(daily.min().min())
    daily = daily.clip(lower=0)

    daily["broad_equipment"] = daily[model_cols].sum(axis=1)
    daily["heavy_equipment"] = daily[heavy_cols].sum(axis=1)
    daily["drone_only"] = daily["drone"]
    daily = daily.reset_index()

    metadata = {
        "start_date": str(raw["date"].min().date()),
        "end_date": str(raw["date"].max().date()),
        "n_days": int(raw.shape[0]),
        "negative_corrections": neg_count,
        "largest_negative_delta": min_delta,
        "final_cumulative_broad": int(cumulative.iloc[-1].sum()),
        "final_cumulative_heavy": int(cumulative[heavy_cols].iloc[-1].sum()),
    }
    return daily, metadata


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(y)
    return s.rolling(window, center=True, min_periods=max(3, window // 3)).mean().bfill().ffill().to_numpy()


def decompose_weekly(dates: pd.Series, y: np.ndarray) -> dict:
    trend = moving_average(y, 31)
    residual = y - trend
    weekdays = pd.to_datetime(dates).dt.weekday.to_numpy()
    seasonal = np.zeros_like(y, dtype=float)
    means = {}
    for wd in range(7):
        means[wd] = float(np.nanmean(residual[weekdays == wd]))
    mean_center = float(np.mean(list(means.values())))
    for wd in range(7):
        seasonal[weekdays == wd] = means[wd] - mean_center
    remainder = y - trend - seasonal
    q1, q3 = np.percentile(remainder, [25, 75])
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr
    shock = remainder > threshold
    return {
        "trend": trend,
        "seasonal": seasonal,
        "remainder": remainder,
        "shock": shock,
        "shock_threshold": float(threshold),
    }


@dataclass
class FitResult:
    name: str
    params: dict
    fitted: np.ndarray
    forecast: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    aic: float
    bic: float
    waic: float | None = None


def gaussian_ic(resid: np.ndarray, k: int) -> tuple[float, float, float]:
    resid = resid[np.isfinite(resid)]
    n = max(len(resid), 1)
    sse = max(float(np.sum(resid**2)), 1e-9)
    sigma2 = sse / n
    loglik = -0.5 * n * (math.log(2 * math.pi * sigma2) + 1)
    return -2 * loglik + 2 * k, -2 * loglik + math.log(n) * k, sigma2


def fit_ets(y_log: np.ndarray, horizon: int) -> FitResult:
    best = None
    for alpha in np.linspace(0.2, 0.9, 8):
        for beta in np.linspace(0.05, 0.5, 10):
            level = y_log[0]
            slope = y_log[1] - y_log[0]
            fitted = np.zeros_like(y_log)
            fitted[0] = y_log[0]
            for t in range(1, len(y_log)):
                fitted[t] = level + slope
                old_level = level
                level = alpha * y_log[t] + (1 - alpha) * (level + slope)
                slope = beta * (level - old_level) + (1 - beta) * slope
            resid = y_log[1:] - fitted[1:]
            aic, bic, sigma2 = gaussian_ic(resid, 4)
            if best is None or aic < best[0]:
                future = np.array([level + (i + 1) * slope for i in range(horizon)])
                se = math.sqrt(sigma2)
                best = (aic, bic, alpha, beta, fitted, future, se)
    assert best is not None
    aic, bic, alpha, beta, fitted, future, se = best
    return FitResult(
        name="ETS(A,A,N)",
        params={"alpha": round(float(alpha), 3), "beta": round(float(beta), 3)},
        fitted=np.expm1(fitted).clip(0),
        forecast=np.expm1(future).clip(0),
        lower=np.expm1(future - 1.96 * se).clip(0),
        upper=np.expm1(future + 1.96 * se).clip(0),
        aic=float(aic),
        bic=float(bic),
    )


def make_lag_matrix(series: np.ndarray, lags: list[int]) -> tuple[np.ndarray, np.ndarray]:
    max_lag = max(lags) if lags else 0
    rows = []
    target = []
    for t in range(max_lag, len(series)):
        rows.append([1.0] + [series[t - lag] for lag in lags])
        target.append(series[t])
    return np.asarray(rows), np.asarray(target)


def recursive_forecast_diff(diff: np.ndarray, lags: list[int], coef: np.ndarray, horizon: int) -> np.ndarray:
    hist = list(diff.astype(float))
    out = []
    for _ in range(horizon):
        x = [1.0] + [hist[-lag] for lag in lags]
        pred = float(np.dot(x, coef))
        hist.append(pred)
        out.append(pred)
    return np.asarray(out)


def fit_arima_family(y_log: np.ndarray, horizon: int, seasonal: bool) -> FitResult:
    diff = np.diff(y_log)
    best = None
    max_p = 5 if seasonal else 7
    for p in range(1, max_p + 1):
        lag_sets = [list(range(1, p + 1))]
        if seasonal:
            lag_sets = [sorted(set(list(range(1, p + 1)) + [7, 14]))]
        for lags in lag_sets:
            x, target = make_lag_matrix(diff, lags)
            coef = np.linalg.lstsq(x, target, rcond=None)[0]
            pred = x @ coef
            resid = target - pred
            aic, bic, sigma2 = gaussian_ic(resid, len(coef) + 1)
            if best is None or aic < best[0]:
                best = (aic, bic, sigma2, lags, coef)
    assert best is not None
    aic, bic, sigma2, lags, coef = best
    x, target = make_lag_matrix(diff, lags)
    diff_fitted = x @ coef
    fitted_log = np.full_like(y_log, np.nan)
    start = 1 + max(lags)
    fitted_log[start:] = y_log[start - 1 : -1] + diff_fitted
    fitted_log[:start] = y_log[:start]
    diff_future = recursive_forecast_diff(diff, lags, coef, horizon)
    future_log = y_log[-1] + np.cumsum(diff_future)
    se = math.sqrt(float(sigma2))
    scale = se * np.sqrt(np.arange(1, horizon + 1))
    name = "SARIMA-AR(weekly)" if seasonal else "ARIMA"
    return FitResult(
        name=name,
        params={"d": 1, "lags": lags},
        fitted=np.expm1(fitted_log).clip(0),
        forecast=np.expm1(future_log).clip(0),
        lower=np.expm1(future_log - 1.96 * scale).clip(0),
        upper=np.expm1(future_log + 1.96 * scale).clip(0),
        aic=float(aic),
        bic=float(bic),
    )


def kernel_matrix(x1: np.ndarray, x2: np.ndarray, kind: str, length: float, period: float, amp: float) -> np.ndarray:
    d = np.abs(x1[:, None] - x2[None, :])
    if kind == "RBF":
        return amp**2 * np.exp(-0.5 * (d / length) ** 2)
    if kind == "Periodic":
        return amp**2 * np.exp(-2 * (np.sin(np.pi * d / period) ** 2) / (length**2))
    if kind == "Matern32":
        z = math.sqrt(3) * d / length
        return amp**2 * (1 + z) * np.exp(-z)
    raise ValueError(kind)


def gp_log_marginal(y: np.ndarray, k: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    try:
        l = np.linalg.cholesky(k)
        alpha = np.linalg.solve(l.T, np.linalg.solve(l, y))
        logdet = 2 * np.sum(np.log(np.diag(l)))
        ll = -0.5 * y @ alpha - 0.5 * logdet - 0.5 * len(y) * math.log(2 * math.pi)
        return float(ll), l, alpha
    except np.linalg.LinAlgError:
        return -np.inf, np.empty((0, 0)), np.empty(0)


def fit_gp(y_log: np.ndarray, horizon: int) -> list[FitResult]:
    n_train = min(500, len(y_log))
    train_y_raw = y_log[-n_train:]
    y_mean = float(train_y_raw.mean())
    train_y = train_y_raw - y_mean
    x = np.arange(n_train, dtype=float)
    x_future = np.arange(n_train, n_train + horizon, dtype=float)
    results = []
    for kind in ["RBF", "Periodic", "Matern32"]:
        best = None
        for length in ([14, 30, 60, 120] if kind != "Periodic" else [0.5, 1.0, 2.0]):
            for amp in [0.5, 1.0, 1.5]:
                for noise in [0.08, 0.15, 0.3]:
                    period = 7.0
                    k = kernel_matrix(x, x, kind, length, period, amp)
                    k += (noise**2 + 1e-6) * np.eye(n_train)
                    ll, l, alpha = gp_log_marginal(train_y, k)
                    if best is None or ll > best[0]:
                        best = (ll, l, alpha, length, amp, noise)
        assert best is not None
        ll, l, alpha, length, amp, noise = best
        k_star = kernel_matrix(x, x_future, kind, float(length), 7.0, float(amp))
        pred_mean_centered = k_star.T @ alpha
        v = np.linalg.solve(l, k_star)
        k_ss_diag = np.diag(kernel_matrix(x_future, x_future, kind, float(length), 7.0, float(amp)))
        pred_var = np.maximum(k_ss_diag - np.sum(v * v, axis=0) + noise**2, 1e-9)
        future_log = pred_mean_centered + y_mean
        train_pred_centered = kernel_matrix(x, x, kind, float(length), 7.0, float(amp)) @ alpha
        train_pred = train_pred_centered + y_mean
        resid = train_y_raw - train_pred
        aic, bic, _ = gaussian_ic(resid, 3)
        waic = approximate_gp_waic(train_y_raw, train_pred, math.sqrt(float(noise**2)))
        se = np.sqrt(pred_var)
        results.append(
            FitResult(
                name=f"GP-{kind}",
                params={"length": round(float(length), 3), "amplitude": float(amp), "noise": float(noise)},
                fitted=np.concatenate([np.full(len(y_log) - n_train, np.nan), np.expm1(train_pred).clip(0)]),
                forecast=np.expm1(future_log).clip(0),
                lower=np.expm1(future_log - 1.96 * se).clip(0),
                upper=np.expm1(future_log + 1.96 * se).clip(0),
                aic=float(aic),
                bic=float(bic),
                waic=float(waic),
            )
        )
    return results


def approximate_gp_waic(y: np.ndarray, mu: np.ndarray, sigma: float) -> float:
    rng = np.random.default_rng(20260602)
    draws = rng.normal(mu[:, None], sigma, size=(len(y), 300))
    logp = -0.5 * ((y[:, None] - draws) / sigma) ** 2 - math.log(sigma) - 0.5 * math.log(2 * math.pi)
    lppd = np.sum(np.log(np.mean(np.exp(logp - logp.max(axis=1, keepdims=True)), axis=1)) + logp.max(axis=1))
    p_waic = np.sum(np.var(logp, axis=1, ddof=1))
    return float(-2 * (lppd - p_waic))


def line_svg(
    path: Path,
    series: list[tuple[str, np.ndarray, str]],
    title: str,
    y_label: str,
    dates: pd.Series | None = None,
    points: np.ndarray | None = None,
    band: tuple[np.ndarray, np.ndarray, str] | None = None,
    width: int = 960,
    height: int = 420,
) -> None:
    margin = dict(left=70, right=30, top=50, bottom=55)
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    all_y = np.concatenate([s[1][np.isfinite(s[1])] for s in series if np.isfinite(s[1]).any()])
    if band is not None:
        all_y = np.concatenate([all_y, band[0][np.isfinite(band[0])], band[1][np.isfinite(band[1])]])
    ymin, ymax = float(np.nanmin(all_y)), float(np.nanmax(all_y))
    pad = (ymax - ymin) * 0.08 + 1e-9
    ymin, ymax = max(0, ymin - pad), ymax + pad
    n = max(len(s[1]) for s in series)

    def xp(i: int) -> float:
        return margin["left"] + (i / max(n - 1, 1)) * plot_w

    def yp(v: float) -> float:
        return margin["top"] + (1 - (v - ymin) / (ymax - ymin)) * plot_h

    def poly(values: np.ndarray) -> str:
        pts = []
        for i, v in enumerate(values):
            if np.isfinite(v):
                pts.append(f"{xp(i):.1f},{yp(float(v)):.1f}")
        return " ".join(pts)

    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin["left"]}" y="28" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#111827">{title}</text>',
        f'<text x="20" y="{height/2}" transform="rotate(-90 20 {height/2})" font-family="Arial, sans-serif" font-size="12" fill="#374151">{y_label}</text>',
    ]
    for frac in np.linspace(0, 1, 5):
        y = margin["top"] + frac * plot_h
        val = ymax - frac * (ymax - ymin)
        elems.append(f'<line x1="{margin["left"]}" y1="{y:.1f}" x2="{width-margin["right"]}" y2="{y:.1f}" stroke="#E5E7EB"/>')
        elems.append(f'<text x="{margin["left"]-8}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#6B7280">{val:.0f}</text>')
    elems.append(f'<line x1="{margin["left"]}" y1="{height-margin["bottom"]}" x2="{width-margin["right"]}" y2="{height-margin["bottom"]}" stroke="#9CA3AF"/>')
    elems.append(f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{height-margin["bottom"]}" stroke="#9CA3AF"/>')
    if band is not None:
        lower, upper, color = band
        upper_pts = [(xp(i), yp(float(v))) for i, v in enumerate(upper) if np.isfinite(v)]
        lower_pts = [(xp(i), yp(float(v))) for i, v in reversed(list(enumerate(lower))) if np.isfinite(v)]
        pts = " ".join([f"{x:.1f},{y:.1f}" for x, y in upper_pts + lower_pts])
        elems.append(f'<polygon points="{pts}" fill="{color}" opacity="0.18"/>')
    for label, values, color in series:
        elems.append(f'<polyline points="{poly(values)}" fill="none" stroke="{color}" stroke-width="2.2"/>')
    if points is not None:
        for i, mask in enumerate(points):
            if mask:
                elems.append(f'<circle cx="{xp(i):.1f}" cy="{yp(float(series[0][1][i])):.1f}" r="3" fill="#DC2626" opacity="0.75"/>')
    xlabels = []
    if dates is not None:
        date_list = pd.to_datetime(dates).dt.strftime("%Y-%m-%d").tolist()
        for idx in np.linspace(0, len(date_list) - 1, 5, dtype=int):
            xlabels.append((idx, date_list[idx]))
    else:
        for idx in np.linspace(0, n - 1, 5, dtype=int):
            xlabels.append((idx, str(idx)))
    for idx, label in xlabels:
        elems.append(f'<text x="{xp(int(idx)):.1f}" y="{height-20}" text-anchor="middle" font-family="Arial" font-size="11" fill="#6B7280">{label}</text>')
    lx = width - margin["right"] - 170
    ly = margin["top"] + 8
    for j, (label, _, color) in enumerate(series):
        elems.append(f'<line x1="{lx}" y1="{ly+j*20}" x2="{lx+18}" y2="{ly+j*20}" stroke="{color}" stroke-width="3"/>')
        elems.append(f'<text x="{lx+25}" y="{ly+4+j*20}" font-family="Arial" font-size="12" fill="#374151">{label}</text>')
    elems.append("</svg>")
    path.write_text("\n".join(elems), encoding="utf-8")


def write_outputs(df: pd.DataFrame, meta: dict, decomp: dict, models: list[FitResult], horizon: int) -> None:
    dates = df["date"]
    y = df["broad_equipment"].to_numpy(float)
    line_svg(
        FIG / "01_series_trend_shocks.svg",
        [("每日装备损耗", y, "#2563EB"), ("31日趋势", decomp["trend"], "#111827")],
        "每日装备损耗与趋势",
        "daily losses",
        dates,
        points=decomp["shock"],
    )
    line_svg(
        FIG / "02_decomposition.svg",
        [
            ("趋势", decomp["trend"], "#111827"),
            ("周季节", decomp["seasonal"] + np.nanmean(decomp["trend"]), "#059669"),
            ("残差(平移)", decomp["remainder"] + np.nanmean(decomp["trend"]), "#D97706"),
        ],
        "STL 类分解：趋势、周季节、突发残差",
        "component scale",
        dates,
    )
    best_gp = min([m for m in models if m.name.startswith("GP")], key=lambda m: m.waic if m.waic is not None else np.inf)
    best_classical = min([m for m in models if not m.name.startswith("GP")], key=lambda m: m.aic)
    future_dates = pd.date_range(dates.iloc[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    combined_dates = pd.concat([dates, pd.Series(future_dates)], ignore_index=True)
    hist_pad = np.concatenate([y, np.full(horizon, np.nan)])
    gp_line = np.concatenate([np.full(len(y), np.nan), best_gp.forecast])
    cls_line = np.concatenate([np.full(len(y), np.nan), best_classical.forecast])
    lower = np.concatenate([np.full(len(y), np.nan), best_gp.lower])
    upper = np.concatenate([np.full(len(y), np.nan), best_gp.upper])
    line_svg(
        FIG / "03_forecast.svg",
        [("历史", hist_pad, "#2563EB"), (best_gp.name, gp_line, "#DC2626"), (best_classical.name, cls_line, "#111827")],
        "60日预测与不确定性带",
        "daily losses",
        combined_dates,
        band=(lower, upper, "#DC2626"),
    )
    metrics = pd.DataFrame(
        [
            {
                "model": m.name,
                "params": json.dumps(m.params, ensure_ascii=False),
                "AIC": m.aic,
                "BIC": m.bic,
                "WAIC": m.waic,
                "forecast_mean_60d": float(np.mean(m.forecast)),
                "forecast_sum_60d": float(np.sum(m.forecast)),
            }
            for m in models
        ]
    )
    metrics.to_csv(OUT / "model_metrics.csv", index=False, encoding="utf-8-sig")
    forecast = pd.DataFrame({"date": future_dates})
    for m in models:
        safe = m.name.replace("(", "").replace(")", "").replace(",", "").replace(" ", "_")
        forecast[f"{safe}_mean"] = m.forecast
        forecast[f"{safe}_lower"] = m.lower
        forecast[f"{safe}_upper"] = m.upper
    forecast.to_csv(OUT / "forecast_60d.csv", index=False, encoding="utf-8-sig")
    (OUT / "analysis_summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def fmt_num(x: float) -> str:
    return f"{x:,.0f}"

def main() -> None:
    ensure_dirs()
    df, meta = load_equipment()
    horizon = 60
    y = df["broad_equipment"].to_numpy(float)
    y_log = np.log1p(y)
    decomp = decompose_weekly(df["date"], y)
    models = [
        fit_ets(y_log, horizon),
        fit_arima_family(y_log, horizon, seasonal=False),
        fit_arima_family(y_log, horizon, seasonal=True),
    ]
    models.extend(fit_gp(y_log, horizon))
    write_outputs(df, meta, decomp, models, horizon)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print("Wrote", OUT / "model_metrics.csv")
    print("Wrote", OUT / "forecast_60d.csv")


if __name__ == "__main__":
    main()
