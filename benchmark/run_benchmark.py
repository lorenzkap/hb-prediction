#!/usr/bin/env python
"""Benchmark candidate model classes on identical data.

1. Patient-grouped 5-fold cross-validation inside the training set
   (GroupKFold on patientId) -> mean +/- SD of AUROC, AUPRC, Brier.
2. Refit on the full training set -> internal test set (and MIMIC-IV if
   prepared) with patient-level bootstrap 95% CIs.

Only aggregated metrics are written (safe to share):
    cv_folds.csv, cv_summary.csv, test_metrics.csv, summary.md, run_info.json

Example
-------
python benchmark/run_benchmark.py --features ~/hb_benchmark/features_published \
       --out ~/hb_benchmark/results_published --device cuda
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hbbench import models as M  # noqa: E402
from hbbench import pipeline as P  # noqa: E402
from hbbench.metrics import bootstrap_ci, point_metrics  # noqa: E402

UNCALIBRATED = {"logreg", "rf", "xgb", "lstm"}  # trained with class/transition weights


def load(features_dir, name, need_seq):
    tab = pd.read_parquet(os.path.join(features_dir, f"{name}.parquet"))
    seq = None
    p = os.path.join(features_dir, f"{name}_seq.npy")
    if need_seq:
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing - rerun prepare_features.py without --no-seq")
        seq = np.load(p, mmap_mode="r")
    return tab, seq


def predict(name, model, X, S):
    if name == "lstm":
        return model.predict_proba(S)[:, 1]
    return model.predict_proba(X)[:, 1]


def fit(name, X, y, w, groups, S, a):
    if name == "lstm":
        return M.LSTMModel(epochs=a.lstm_epochs, batch_size=a.lstm_batch, verbose=2 if a.verbose else 0).fit(
            np.asarray(S), y, w, groups)
    model = M.make_tabular(name, device=a.device, n_jobs=a.n_jobs)
    return M.fit_tabular(name, model, X, y, w)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)).decode().strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=",".join(M.ALL), help=f"comma list from {M.ALL}")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--cv-calibrated", action="store_true",
                    help="also cross-validate xgb_calibrated_published (5x5 fits, slow)")
    ap.add_argument("--device", default="cpu", help="xgboost device: cpu or cuda")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--lstm-epochs", type=int, default=20)
    ap.add_argument("--lstm-batch", type=int, default=2048)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    names = [m.strip() for m in a.models.split(",") if m.strip()]
    for n in names:
        assert n in M.ALL, n
    out = os.path.expanduser(a.out)
    fdir = os.path.expanduser(a.features)
    os.makedirs(out, exist_ok=True)
    meta = json.load(open(os.path.join(fdir, "meta.json")))
    feats = meta["features"]
    need_seq = "lstm" in names

    tr, Str = load(fdir, "train", need_seq)
    te, Ste = load(fdir, "test", need_seq)
    ext = None
    if os.path.exists(os.path.join(fdir, "external.parquet")):
        ext, Sext = load(fdir, "external", need_seq)

    Xtr, ytr = tr[feats].to_numpy(np.float32), tr["y_binary"].to_numpy()
    Xte, yte = te[feats].to_numpy(np.float32), te["y_binary"].to_numpy()
    gtr = tr[P.PAT].to_numpy()
    info = {"started": time.strftime("%Y-%m-%d %H:%M"), "git_commit": git_commit(), "python": platform.python_version(),
            "features_dir_meta": {k: meta[k] for k in ("bfill", "n_features", "train", "test") if k in meta},
            "models": names, "folds": a.folds, "device": a.device}
    try:
        import sklearn, xgboost
        info["sklearn"], info["xgboost"] = sklearn.__version__, xgboost.__version__
    except Exception:
        pass

    # ------------------------- cross-validation --------------------------- #
    fold_rows = []
    if not a.no_cv:
        gkf = GroupKFold(n_splits=a.folds)
        for k, (i_tr, i_va) in enumerate(gkf.split(Xtr, ytr, gtr)):
            w = P.transition_weights(tr[P.ENC].to_numpy()[i_tr], ytr[i_tr])
            for n in names:
                if n == "xgb_calibrated_published" and not a.cv_calibrated:
                    continue
                t0 = time.time()
                if n in M.BASELINES:
                    p = M.baseline_scores(n, tr[P.HB].to_numpy()[i_va])
                else:
                    S_tr = Str[i_tr] if n == "lstm" else None
                    model = fit(n, Xtr[i_tr], ytr[i_tr], w, gtr[i_tr], S_tr, a)
                    p = predict(n, model, Xtr[i_va], Str[i_va] if n == "lstm" else None)
                r = point_metrics(ytr[i_va], p, calibrated_probability=n not in M.BASELINES)
                r.update(model=n, fold=k, seconds=round(time.time() - t0, 1))
                fold_rows.append(r)
                print(f"[cv fold {k}] {n:28s} AUROC {r['auroc']:.4f} AUPRC {r['auprc']:.4f} ({r['seconds']} s)", flush=True)
                pd.DataFrame(fold_rows).to_csv(os.path.join(out, "cv_folds.csv"), index=False)
        cv = pd.DataFrame(fold_rows)
        summ = cv.groupby("model").agg(auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
                                       auprc_mean=("auprc", "mean"), auprc_sd=("auprc", "std"),
                                       brier_mean=("brier", "mean"), brier_sd=("brier", "std")).reset_index()
        summ.to_csv(os.path.join(out, "cv_summary.csv"), index=False)

    # --------------------- refit + held-out evaluation -------------------- #
    w_full = P.transition_weights(tr[P.ENC].to_numpy(), ytr)
    test_rows = []
    for n in names:
        t0 = time.time()
        model = None if n in M.BASELINES else fit(n, Xtr, ytr, w_full, gtr, Str if n == "lstm" else None, a)
        for split, tab, X, y, S in [("internal_test", te, Xte, yte, Ste)] + (
                [("external_mimic", ext, ext[feats].to_numpy(np.float32), ext["y_binary"].to_numpy(), Sext)] if ext is not None else []):
            p = M.baseline_scores(n, tab[P.HB].to_numpy()) if model is None else predict(n, model, X, S)
            r = point_metrics(y, p, calibrated_probability=n not in M.BASELINES)
            r.update(bootstrap_ci(y, p, groups=tab[P.PAT].to_numpy(), n_boot=a.n_boot))
            r.update(model=n, split=split, seconds=round(time.time() - t0, 1))
            test_rows.append(r)
            print(f"[{split}] {n:28s} AUROC {r['auroc']:.4f} ({r['auroc_lo']:.4f}-{r['auroc_hi']:.4f})", flush=True)
            pd.DataFrame(test_rows).to_csv(os.path.join(out, "test_metrics.csv"), index=False)
    test = pd.DataFrame(test_rows)

    # ------------------------------ summary ------------------------------- #
    lines = [f"# Model benchmark ({'published preprocessing' if meta.get('bfill', True) else 'no backward fill'})", "",
             f"Features: {len(feats)} engineered (tabular models); LSTM: raw hourly sequence (19 x 19).",
             f"Train: {meta['train']}  Test: {meta['test']}", "",
             "| Model | CV AUROC (mean ± SD) | CV AUPRC (mean ± SD) | CV Brier | Test AUROC (95% CI) | Test AUPRC (95% CI) | Test Brier |"
             + (" External AUROC (95% CI) |" if ext is not None else ""),
             "|---|---|---|---|---|---|---|" + ("---|" if ext is not None else "")]
    for n in names:
        row = [M.LABELS[n]]
        if fold_rows and n in set(cv["model"]):
            s = summ.set_index("model").loc[n]
            row += [f"{s.auroc_mean:.3f} ± {s.auroc_sd:.3f}", f"{s.auprc_mean:.3f} ± {s.auprc_sd:.3f}",
                    "–" if np.isnan(s.brier_mean) else f"{s.brier_mean:.3f}"]
        else:
            row += ["–", "–", "–"]
        t = test[(test.model == n) & (test.split == "internal_test")].iloc[0]
        row += [f"{t.auroc:.3f} ({t.auroc_lo:.3f}–{t.auroc_hi:.3f})", f"{t.auprc:.3f} ({t.auprc_lo:.3f}–{t.auprc_hi:.3f})",
                "–" if np.isnan(t.brier) else f"{t.brier:.3f}"]
        if ext is not None:
            e = test[(test.model == n) & (test.split == "external_mimic")].iloc[0]
            row += [f"{e.auroc:.3f} ({e.auroc_lo:.3f}–{e.auroc_hi:.3f})"]
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "Brier scores of models trained with class/transition weights (" + ", ".join(sorted(UNCALIBRATED)) +
              ") reflect weighted training and are not comparable with the calibrated model; compare discrimination."]
    open(os.path.join(out, "summary.md"), "w").write("\n".join(lines) + "\n")
    info["finished"] = time.strftime("%Y-%m-%d %H:%M")
    json.dump(info, open(os.path.join(out, "run_info.json"), "w"), indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
