"""
Feature pipeline for the Hb benchmark.

This is a vectorised re-implementation of the preprocessing in
``modelling_regression-MUW_only_measured.ipynb`` (cells 2-90). The semantics
are identical to the notebook (verified in ``tests/test_equivalence.py``):

1. sort by encounter/time, flag real Hb measurements (``Hb_not_NaN``) and set
   the *first* measurement of each encounter to 0 (it is never a target),
2. map sex (M=0, F=1, other=2),
3. keep encounters with >=1 flagged Hb and trim each encounter to
   [first flagged Hb - 18 h, last flagged Hb],
4. patient-level 70/30 split (``train_test_split`` on ``patientId``, seed 42),
5. forward fill (and, as in the published notebook, backward fill) within
   encounter, then median imputation with training medians,
6. hourly lags 1..18 for all time-varying variables (first-row fallback),
7. outcome: minimum *measured* Hb in (t, t+6 h] < 8 g/dL; samples without any
   measured Hb in the window are dropped,
8. engineered features (differences to 1/6/12/18 h earlier; 6 h and 18 h
   cumulative sums) -> 51 features in alphabetical order,
9. training weights: class balancing and 5x boost of 0->1 transitions.

The only intentional addition is the ``bfill`` switch (default True = as
published) that allows a sensitivity analysis without backward filling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ENC = "encounterId"
PAT = "patientId"
TIME = "utcChartTime"
HB = "hemoglobin_g/dl"
FLAG = "Hb_not_NaN"

SEQUENCE_LEN = 18
PRED_OFFSET = 6
HB_THRESHOLD = 8
SAMPLING_INTERVAL = "1h"

TIME_VARIABLES = [
    "heart_rate", "respiratory_rate", "blood_pressure_systolic_mmHg", "spo2",
    "blood_pressure_mean_mmHg", "blood_pressure_diastolic_mmHg",
    "combined_vaso",
    "fluids_ml", "colloids_ml", "lactate_mmol/l", "base_excess_mmol/l",
    HB, "fibrinogen_mg/dl", "platelet_count_G/l", "harnk_ml",
    "drain_sum", "blood_input",
]
STATIC_VARIABLES = ["age", "sex_or_gender"]

_SUM6 = {"lags": [1, 2, 3, 4, 5, 6], "mode": "sum"}
LAGS_CONFIG = {
    "heart_rate": {"lags": [1, 6, 18], "mode": "diff"},
    "respiratory_rate": {"lags": [6, 18], "mode": "diff"},
    "blood_pressure_systolic_mmHg": {"lags": [6], "mode": "diff"},
    "spo2": {"lags": [6], "mode": "diff"},
    "blood_pressure_mean_mmHg": {"lags": [6, 18], "mode": "diff"},
    "blood_pressure_diastolic_mmHg": {"lags": [6], "mode": "diff"},
    "combined_vaso": {"lags": [18], "mode": "diff"},
    "lactate_mmol/l": {"lags": [6, 18], "mode": "diff"},
    "base_excess_mmol/l": {"lags": [6, 18], "mode": "diff"},
    HB: {"lags": [6, 12, 18], "mode": "diff"},
    "fibrinogen_mg/dl": {"lags": [6, 18], "mode": "diff"},
    "platelet_count_G/l": {"lags": [6, 18], "mode": "diff"},
    "harnk_ml": dict(_SUM6), "drain_sum": dict(_SUM6), "blood_input": dict(_SUM6),
    "fluids_ml": dict(_SUM6), "colloids_ml": dict(_SUM6),
}
_SUM18 = {"lags": list(range(1, 19)), "mode": "sum"}
SUM_CONFIG = {v: dict(_SUM18) for v in ["harnk_ml", "drain_sum", "blood_input", "fluids_ml", "colloids_ml"]}

NON_FEATURES = {FLAG, ENC, PAT, TIME, "y", "y_binary", "y_target"}


# --------------------------------------------------------------------------- #
def flag_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values([ENC, TIME]).reset_index(drop=True)
    df[FLAG] = df[HB].notna().astype(int)
    first = (df[FLAG] == 1) & (df.groupby(ENC)[FLAG].cumsum() == 1)
    df.loc[first, FLAG] = 0
    df["sex_or_gender"] = df["sex_or_gender"].map({"M": 0, "F": 1}).fillna(2).astype(int)
    return df


def trim_sequences(df: pd.DataFrame, sequence_len=SEQUENCE_LEN, sampling_interval=SAMPLING_INTERVAL) -> pd.DataFrame:
    flagged = df.loc[df[FLAG] == 1].groupby(ENC)[TIME].agg(["min", "max"])
    df = df[df[ENC].isin(flagged.index)]
    lo = df[ENC].map(flagged["min"]) - sequence_len * pd.Timedelta(sampling_interval)
    hi = df[ENC].map(flagged["max"])
    return df[(df[TIME] >= lo) & (df[TIME] <= hi)].reset_index(drop=True)


def split_by_patient(df: pd.DataFrame, cis: pd.DataFrame, train_size=0.7, seed=42):
    df = df.merge(cis[[ENC, PAT]], on=ENC)
    ids = df[PAT].unique()
    train_ids, test_ids = train_test_split(ids, train_size=train_size, random_state=seed)
    return df[df[PAT].isin(train_ids)].copy(), df[df[PAT].isin(test_ids)].copy()


def base_feature_cols(df: pd.DataFrame):
    cols = [c for c in df.columns if c not in NON_FEATURES]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def impute(train: pd.DataFrame, others: list[pd.DataFrame], feature_cols, bfill=True, medians=None):
    """ffill (+ optional bfill) within encounter, then fill with training medians."""
    out = []
    for d in [train] + list(others):
        d = d.sort_values([ENC, TIME]).copy()
        d[feature_cols] = d[feature_cols].astype(np.float32)
        g = d.groupby(ENC, sort=False)
        d[feature_cols] = g[feature_cols].ffill()
        if bfill:
            d[feature_cols] = d.groupby(ENC, sort=False)[feature_cols].bfill()
        out.append(d.reset_index(drop=True))
    if medians is None:
        medians = out[0][feature_cols].median()
    out = [d.fillna({c: medians[c] for c in feature_cols}) for d in out]
    for d in out:
        assert not d[feature_cols].isnull().values.any(), "NaNs remain after imputation"
    return out, medians


def add_lags(df: pd.DataFrame, sequence_len=SEQUENCE_LEN) -> pd.DataFrame:
    g = df.groupby(ENC, sort=False)
    new = {}
    for var in TIME_VARIABLES:
        first = g[var].transform("first")
        for i in range(1, sequence_len + 1):
            new[f"{var}_lag_{i}"] = g[var].shift(i).fillna(first).astype(np.float32)
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def add_outcome(df: pd.DataFrame, pred_offset=PRED_OFFSET, hb_threshold=HB_THRESHOLD) -> pd.DataFrame:
    real = df[HB].where(df[FLAG] == 1)
    g = real.groupby(df[ENC], sort=False)
    fut = pd.concat([g.shift(-k) for k in range(1, pred_offset + 1)], axis=1).min(axis=1, skipna=True)
    df = df.assign(future_min_hb=fut)
    df = df.dropna(subset=["future_min_hb"])
    df["y_binary"] = (df["future_min_hb"] < hb_threshold).astype(int)
    return df.drop(columns=["future_min_hb"]).reset_index(drop=True)


def _apply_config(df, config, out):
    for var, p in config.items():
        lags = [l for l in p["lags"] if f"{var}_lag_{l}" in df.columns]
        if not lags:
            continue
        if p["mode"] == "diff":
            for l in lags:
                out[f"{var}_lag_{l}_diff"] = (df[f"{var}_lag_{l}"] - df[var]).astype(np.float32)
        elif p["mode"] == "sum":
            out[f"{var}_lags_{max(p['lags'])}_sum"] = df[[f"{var}_lag_{l}" for l in lags]].sum(axis=1).astype(np.float32)
        elif p["mode"] == "normal":
            for l in lags:
                out[f"{var}_lag_{l}"] = df[f"{var}_lag_{l}"]


def engineer_features(df: pd.DataFrame):
    """Returns (table with ids/outcome + 51 engineered features, sorted feature list)."""
    lag_cols = [c for c in df.columns if "_lag_" in c]
    keep = df[[c for c in df.columns if c not in lag_cols]].copy()
    new = {}
    _apply_config(df, LAGS_CONFIG, new)
    _apply_config(df, SUM_CONFIG, new)
    out = pd.concat([keep, pd.DataFrame(new, index=df.index)], axis=1)
    feats = sorted(c for c in out.columns if c not in NON_FEATURES and c != "y")
    return out, feats


def sequence_array(df: pd.DataFrame, sequence_len=SEQUENCE_LEN, dtype=np.float32):
    """(n, sequence_len+1, 17 time-varying + 2 static) array, oldest step first."""
    steps = []
    for i in range(sequence_len, 0, -1):
        steps.append(np.stack([df[f"{v}_lag_{i}"].to_numpy(dtype) for v in TIME_VARIABLES], axis=1))
    steps.append(np.stack([df[v].to_numpy(dtype) for v in TIME_VARIABLES], axis=1))
    x = np.stack(steps, axis=1)
    static = np.stack([df[v].to_numpy(dtype) for v in STATIC_VARIABLES], axis=1)
    static = np.repeat(static[:, None, :], x.shape[1], axis=1)
    return np.concatenate([x, static], axis=2)


def transition_weights(enc_ids, y, balance_classes=True, switch_boost_factor=5.0):
    """Class balancing + boost of 0->1 transitions (identical to the notebook)."""
    y = np.asarray(y)
    if balance_classes:
        n = len(y); n0 = (y == 0).sum(); n1 = (y == 1).sum()
        sw = np.where(y == 1, n / (2.0 * n1), n / (2.0 * n0)).astype(float)
    else:
        sw = np.ones(len(y), dtype=float)
    enc = pd.Series(np.asarray(enc_ids))
    prev = pd.Series(y).groupby(enc.values, sort=False).shift(1)
    boost = ((prev == 0) & (pd.Series(y) == 1)).to_numpy()
    sw[boost] *= switch_boost_factor
    return sw


def build(df_raw: pd.DataFrame, cis: pd.DataFrame | None, bfill=True, keep_lags=False,
          medians=None, external=False):
    """Full pipeline. Returns dict(train=..., test=..., features=..., medians=...).

    external=True: no split (all encounters go to 'test'); ``medians`` from the
    development training set must then be supplied.
    """
    df = flag_and_encode(df_raw)
    df = trim_sequences(df)
    if external:
        df[PAT] = df[ENC]
        parts = {"test": df}
    else:
        tr, te = split_by_patient(df, cis)
        parts = {"train": tr, "test": te}
    fcols = base_feature_cols(next(iter(parts.values())))
    names = list(parts)
    if external:
        (imputed,), medians = impute(parts["test"], [], fcols, bfill=bfill, medians=medians)
        imputed = [imputed]
    else:
        imputed, medians = impute(parts["train"], [parts["test"]], fcols, bfill=bfill)
    res = {"medians": medians}
    for name, d in zip(names, imputed):
        d = add_lags(d)
        d = add_outcome(d)
        seq = sequence_array(d) if keep_lags else None
        tab, feats = engineer_features(d)
        res[name] = tab
        res[name + "_seq"] = seq
        res["features"] = feats
    return res
