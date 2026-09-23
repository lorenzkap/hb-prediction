from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def point_metrics(y, p, calibrated_probability=True):
    out = {"n": int(len(y)), "prevalence": float(np.mean(y)),
           "auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p))}
    # Brier score only meaningful for scores in [0, 1]
    out["brier"] = float(brier_score_loss(y, p)) if calibrated_probability and p.min() >= 0 and p.max() <= 1 else np.nan
    return out


def bootstrap_ci(y, p, groups=None, n_boot=200, seed=42, alpha=0.05):
    """95% CI for AUROC/AUPRC. groups -> cluster (patient-level) bootstrap."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p)
    if groups is not None:
        groups = np.asarray(groups)
        order = np.argsort(groups, kind="stable")
        g_sorted = groups[order]
        uniq, start = np.unique(g_sorted, return_index=True)
        ends = np.r_[start[1:], len(g_sorted)]
        idx_by_g = [order[s:e] for s, e in zip(start, ends)]
    aurocs, auprcs = [], []
    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, len(y), len(y))
        else:
            pick = rng.integers(0, len(idx_by_g), len(idx_by_g))
            idx = np.concatenate([idx_by_g[i] for i in pick])
        if y[idx].min() == y[idx].max():
            continue
        aurocs.append(roc_auc_score(y[idx], p[idx]))
        auprcs.append(average_precision_score(y[idx], p[idx]))
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    a = np.percentile(aurocs, q); b = np.percentile(auprcs, q)
    return {"auroc_lo": float(a[0]), "auroc_hi": float(a[1]), "auprc_lo": float(b[0]), "auprc_hi": float(b[1])}
