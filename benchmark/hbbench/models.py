"""Candidate models for the benchmark. All tabular models see the same 51
engineered features; the LSTM sees the raw hourly sequence (19 steps x 19
channels) from which those features were derived."""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Hyperparameters of the published model (xgb_models_best/best_params.json),
# found by RandomizedSearchCV (30 draws, 5-fold CV, AUROC) in the notebook.
PUBLISHED_XGB_PARAMS = {
    "colsample_bytree": 0.905269907953647,
    "gamma": 0.17606099749584053,
    "learning_rate": 0.03440764696895577,
    "max_depth": 7,
    "min_child_weight": 3,
    "n_estimators": 405,
    "reg_alpha": 0.3910606075732408,
    "reg_lambda": 1.8223608778806233,
    "subsample": 0.9266084230952957,
}

TABULAR = ["logreg", "rf", "xgb", "xgb_calibrated_published"]
BASELINES = ["last_hb_binary", "last_hb_continuous"]
SEQUENCE = ["lstm"]
ALL = BASELINES + TABULAR + SEQUENCE

LABELS = {
    "last_hb_binary": "Most recent Hb <8 g/dL (reference)",
    "last_hb_continuous": "Most recent Hb, continuous (reference)",
    "logreg": "Logistic regression (L2)",
    "rf": "Random forest",
    "xgb": "XGBoost (published hyperparameters, weighted)",
    "xgb_calibrated_published": "XGBoost + isotonic calibration (published final model)",
    "lstm": "LSTM on hourly sequences",
}


def _xgb(device, n_jobs):
    import xgboost as xgb
    return xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc", tree_method="hist",
                             device=device, n_jobs=n_jobs, random_state=42, **PUBLISHED_XGB_PARAMS)


def make_tabular(name, device="cpu", n_jobs=-1):
    if name == "logreg":
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(C=1.0, max_iter=2000))])
    if name == "rf":
        return Pipeline([("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                                        max_features="sqrt", max_samples=0.3,
                                                        n_jobs=n_jobs, random_state=42))])
    if name == "xgb":
        return Pipeline([("scaler", StandardScaler()), ("clf", _xgb(device, n_jobs))])
    if name == "xgb_calibrated_published":
        # exactly as in the notebook: CalibratedClassifierCV(best_pipeline, isotonic, cv=5).fit(X, y)
        # -> refit WITHOUT sample weights
        base = Pipeline([("scaler", StandardScaler()), ("classifier", _xgb(device, n_jobs))])
        return CalibratedClassifierCV(base, method="isotonic", cv=5)
    raise ValueError(name)


def fit_tabular(name, model, X, y, w):
    if name == "xgb_calibrated_published":
        model.fit(X, y)  # no weights, as published
    else:
        model.fit(X, y, clf__sample_weight=w)
    return model


def baseline_scores(name, hb_current):
    if name == "last_hb_binary":
        return (hb_current < 8).astype(float)
    if name == "last_hb_continuous":
        return -hb_current.astype(float)
    raise ValueError(name)


# ------------------------------- LSTM ------------------------------------- #
def _batches(X, y, w, idx, batch_size, mu, sd, shuffle, seed):
    """Keras data source that normalises and ships one batch at a time, so the
    (multi-GB) sequence array never has to be copied to the GPU as a whole."""
    import tensorflow as tf

    class _Seq(tf.keras.utils.Sequence):
        def __init__(self):
            try:
                super().__init__(workers=1, use_multiprocessing=False, max_queue_size=10)
            except TypeError:  # Keras 2
                super().__init__()
            self.idx = np.array(idx)
            self.rng = np.random.default_rng(seed)
            if shuffle:
                self.rng.shuffle(self.idx)

        def __len__(self):
            return int(np.ceil(len(self.idx) / batch_size))

        def __getitem__(self, i):
            b = np.sort(self.idx[i * batch_size:(i + 1) * batch_size])
            xb = ((np.asarray(X[b], dtype=np.float32) - mu) / sd).astype(np.float32)
            if w is None:
                return xb, y[b].astype(np.float32)
            return xb, y[b].astype(np.float32), w[b].astype(np.float32)

        def on_epoch_end(self):
            if shuffle:
                self.rng.shuffle(self.idx)

    return _Seq()


def _gpu_memory_growth():
    import tensorflow as tf
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass


class LSTMModel:
    """2-layer LSTM (64/32) + dense 32, dropout 0.2 - the architecture of
    modelling_neural.ipynb - with Adam(1e-3), early stopping on the AUROC of a
    patient-level 10 % validation split of the training data."""

    def __init__(self, epochs=20, batch_size=2048, patience=3, seed=42, verbose=2):
        self.epochs, self.batch_size, self.patience, self.seed, self.verbose = epochs, batch_size, patience, seed, verbose

    def fit(self, X, y, w, groups):
        _gpu_memory_growth()
        import tensorflow as tf
        from sklearn.model_selection import GroupShuffleSplit
        tf.keras.utils.set_random_seed(self.seed)
        n, t, c = X.shape
        # channel mean/sd in chunks (float64 accumulation, no full copy)
        s1 = np.zeros(c); s2 = np.zeros(c); cnt = 0
        for i in range(0, n, 100_000):
            ch = np.asarray(X[i:i + 100_000], dtype=np.float64).reshape(-1, c)
            s1 += ch.sum(0); s2 += (ch ** 2).sum(0); cnt += len(ch)
        self.mu = (s1 / cnt).astype(np.float32)
        self.sd = (np.sqrt(np.maximum(s2 / cnt - (s1 / cnt) ** 2, 0)) + 1e-6).astype(np.float32)
        tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=self.seed).split(np.zeros(n), y, groups))
        m = tf.keras.Sequential([
            tf.keras.Input((t, c)),
            tf.keras.layers.LSTM(64, return_sequences=True), tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32), tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"), tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc")])
        es = tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=self.patience,
                                              restore_best_weights=True)
        train_data = _batches(X, y, w, tr, self.batch_size, self.mu, self.sd, True, self.seed)
        val_data = _batches(X, y, None, va, 8192, self.mu, self.sd, False, self.seed)
        m.fit(train_data, validation_data=val_data, epochs=self.epochs, callbacks=[es], verbose=self.verbose)
        self.model = m
        return self

    def predict_proba(self, X):
        out = []
        for i in range(0, len(X), 100_000):
            xb = ((np.asarray(X[i:i + 100_000], dtype=np.float32) - self.mu) / self.sd).astype(np.float32)
            out.append(self.model.predict(xb, batch_size=8192, verbose=0).ravel())
        p = np.concatenate(out)
        return np.c_[1 - p, p]
