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
class LSTMModel:
    """2-layer LSTM (64/32) + dense 32, dropout 0.2 - the architecture of
    modelling_neural.ipynb - with Adam(1e-3), early stopping on the AUROC of a
    patient-level 10 % validation split of the training data."""

    def __init__(self, epochs=20, batch_size=2048, patience=3, seed=42, verbose=2):
        self.epochs, self.batch_size, self.patience, self.seed, self.verbose = epochs, batch_size, patience, seed, verbose

    def fit(self, X, y, w, groups):
        import tensorflow as tf
        from sklearn.model_selection import GroupShuffleSplit
        tf.keras.utils.set_random_seed(self.seed)
        n, t, c = X.shape
        self.mu = X.reshape(-1, c).mean(0)
        self.sd = X.reshape(-1, c).std(0) + 1e-6
        tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=self.seed).split(X, y, groups))
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
        m.fit(self._norm(X[tr]), y[tr], sample_weight=w[tr],
              validation_data=(self._norm(X[va]), y[va]),
              epochs=self.epochs, batch_size=self.batch_size, callbacks=[es], verbose=self.verbose)
        self.model = m
        return self

    def _norm(self, X):
        return ((X - self.mu) / self.sd).astype(np.float32)

    def predict_proba(self, X):
        out = []
        for i in range(0, len(X), 200_000):
            out.append(self.model.predict(self._norm(X[i:i + 200_000]), batch_size=8192, verbose=0).ravel())
        p = np.concatenate(out)
        return np.c_[1 - p, p]
