"""Synthetic stand-in for muw_df.parquet / CIS.parquet (same schema, random
values) so the pipeline can be tested without patient data."""
import numpy as np
import pandas as pd

COLS = ["age", "sex_or_gender", "blood_pressure_diastolic_mmHg", "blood_pressure_mean_mmHg",
        "blood_pressure_systolic_mmHg", "combined_vaso", "colloids_ml", "fluids_ml", "fibrinogen_mg/dl",
        "lactate_mmol/l", "platelet_count_G/l", "hemoglobin_g/dl", "base_excess_mmol/l", "harnk_ml",
        "drain_sum", "heart_rate", "respiratory_rate", "spo2", "blood_input"]


def make(n_enc=120, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_enc):
        L = int(rng.integers(8, 90))
        t = pd.date_range("2020-01-01", periods=L, freq="1h") + pd.Timedelta(days=int(e))
        hb = 10 + np.cumsum(rng.normal(-0.05, 0.3, L))
        d = pd.DataFrame({"utcChartTime": t, "encounterId": 1000 + e})
        d["age"] = float(rng.integers(20, 90))
        d["sex_or_gender"] = rng.choice(["M", "F", "W"], p=[0.55, 0.4, 0.05])
        for c in COLS[2:]:
            d[c] = rng.normal(50, 10, L)
            d.loc[rng.random(L) < 0.6, c] = np.nan
        d["hemoglobin_g/dl"] = np.where(rng.random(L) < 0.2, hb, np.nan)
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    cis = pd.DataFrame({"encounterId": 1000 + np.arange(n_enc), "patientId": 5000 + np.arange(n_enc) // 2})
    return df, cis


if __name__ == "__main__":
    import sys
    out = sys.argv[1]
    df, cis = make(n_enc=300)
    df.to_parquet(f"{out}/muw_df.parquet")
    cis.to_parquet(f"{out}/CIS.parquet")
    m, _ = make(n_enc=60, seed=1)
    m.to_parquet(f"{out}/mimic_df.parquet")
