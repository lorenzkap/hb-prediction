#!/usr/bin/env python
"""Build the benchmark feature tables from the processed ViennaAIdb data.

Example
-------
python benchmark/prepare_features.py \
    --muw /home/lkapral/hb/data/muw_df.parquet \
    --cis /home/lkapral/hb/data/CIS.parquet \
    --out ~/hb_benchmark/features_published

# sensitivity analysis without backward filling
python benchmark/prepare_features.py ... --no-bfill --out ~/hb_benchmark/features_nobfill

Outputs (patient-level data - keep them on the server, never commit them):
    train.parquet, test.parquet   ids, outcome and the 51 engineered features
    train_seq.npy, test_seq.npy   hourly sequences for the LSTM (unless --no-seq)
    [external.parquet, external_seq.npy  if --mimic is given]
    meta.json                     feature list, counts, settings (aggregated only)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hbbench import pipeline as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--muw", required=True, help="muw_df.parquet (hourly ViennaAIdb data)")
    ap.add_argument("--cis", required=True, help="CIS.parquet (encounterId -> patientId)")
    ap.add_argument("--mimic", default=None, help="optional mimic_df.parquet for external evaluation")
    ap.add_argument("--mimic-ids", default=None, help="optional mimic_ids.csv to restrict MIMIC stays")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-bfill", action="store_true", help="sensitivity analysis: no backward fill")
    ap.add_argument("--no-seq", action="store_true", help="skip LSTM sequence arrays")
    ap.add_argument("--subsample-encounters", type=float, default=None,
                    help="smoke test: keep this fraction of encounters")
    a = ap.parse_args()

    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet(a.muw)
    cis = pd.read_parquet(a.cis)
    if a.subsample_encounters:
        keep = pd.Series(df[P.ENC].unique()).sample(frac=a.subsample_encounters, random_state=0)
        df = df[df[P.ENC].isin(keep)]
    df[P.TIME] = pd.to_datetime(df[P.TIME])
    print(f"loaded {len(df):,} rows / {df[P.ENC].nunique():,} encounters", flush=True)

    res = P.build(df, cis, bfill=not a.no_bfill, keep_lags=not a.no_seq)
    feats = res["features"]
    meta = {"bfill": not a.no_bfill, "n_features": len(feats), "features": feats,
            "medians": {k: float(v) for k, v in res["medians"].items()}}
    for name in ["train", "test"]:
        tab = res[name]
        cols = [P.ENC, P.PAT, P.TIME, "y_binary"] + feats
        tab[cols].to_parquet(os.path.join(out, f"{name}.parquet"), index=False)
        if res[name + "_seq"] is not None:
            np.save(os.path.join(out, f"{name}_seq.npy"), res[name + "_seq"].astype(np.float32))
        meta[name] = {"samples": int(len(tab)), "encounters": int(tab[P.ENC].nunique()),
                      "patients": int(tab[P.PAT].nunique()), "prevalence": float(tab["y_binary"].mean())}
        print(name, meta[name], flush=True)

    if a.mimic:
        m = pd.read_parquet(a.mimic)
        m[P.TIME] = pd.to_datetime(m[P.TIME])
        if a.mimic_ids:
            ids = pd.read_csv(a.mimic_ids)[P.ENC]
            m = m[m[P.ENC].isin(ids)]
        rx = P.build(m, None, bfill=not a.no_bfill, keep_lags=not a.no_seq,
                     medians=res["medians"], external=True)
        tab = rx["test"]
        missing = [f for f in feats if f not in tab.columns]
        assert not missing, f"MIMIC is missing features: {missing}"
        tab[[P.ENC, P.PAT, P.TIME, "y_binary"] + feats].to_parquet(os.path.join(out, "external.parquet"), index=False)
        if rx["test_seq"] is not None:
            np.save(os.path.join(out, "external_seq.npy"), rx["test_seq"].astype(np.float32))
        meta["external"] = {"samples": int(len(tab)), "encounters": int(tab[P.ENC].nunique()),
                            "prevalence": float(tab["y_binary"].mean()),
                            "note": "imputed with ViennaAIdb training medians"}
        print("external", meta["external"], flush=True)

    meta["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("done ->", out)


if __name__ == "__main__":
    main()
