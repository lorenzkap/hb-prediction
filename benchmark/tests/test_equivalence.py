"""Checks that hbbench.pipeline reproduces the notebook preprocessing exactly.

The reference implementation below is copied from
modelling_regression-MUW_only_measured.ipynb (cells 16-90, unchanged logic,
file paths replaced by in-memory data). Requires pandas 2.x (the notebook's
groupby.apply semantics).

    python benchmark/tests/test_equivalence.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from hbbench import pipeline as P  # noqa: E402
import synthetic  # noqa: E402

sampling_interval, sequence_len, pred_offset, hb_threshold = '1h', 18, 6, 8
time_variables = P.TIME_VARIABLES


def reference(df, demo):
    df = df.copy()
    df.sort_values(by=['encounterId', 'utcChartTime'], inplace=True)
    df = df.reset_index(drop=True)
    df = df.sort_values(by=['encounterId', 'utcChartTime'])
    df['Hb_not_NaN'] = df['hemoglobin_g/dl'].notna().astype(int)
    first_occurrence_mask = (df['Hb_not_NaN'] == 1) & (df.groupby('encounterId')['Hb_not_NaN'].cumsum() == 1)
    df.loc[first_occurrence_mask, 'Hb_not_NaN'] = 0
    df['sex_or_gender_numeric'] = df['sex_or_gender'].map({'M': 0, 'F': 1})
    df['sex_or_gender'] = df['sex_or_gender_numeric'].fillna(2).astype(int)
    df = df.drop(columns=['sex_or_gender_numeric'])

    def trim_sequences(df, group_col, time_col, hb_col, sequence_len, sampling_interval):
        df_filtered = df.groupby(group_col).filter(lambda g: (g[hb_col] == 1).any())
        df_filtered = df_filtered.sort_values([group_col, time_col])
        sampling_interval = pd.Timedelta(sampling_interval)

        def _trim(group):
            first_time = group.loc[group[hb_col] == 1, time_col].min()
            last_time = group.loc[group[hb_col] == 1, time_col].max()
            mask = (group[time_col] >= first_time - sequence_len * sampling_interval) & (group[time_col] <= last_time)
            return group.loc[mask]
        return df_filtered.groupby(group_col, group_keys=False).apply(_trim).reset_index(drop=True)

    df = trim_sequences(df, 'encounterId', 'utcChartTime', 'Hb_not_NaN', sequence_len, sampling_interval)
    df['y'] = 0
    df.loc[df['Hb_not_NaN'] == 1, 'y'] = df.loc[df['Hb_not_NaN'] == 1, 'hemoglobin_g/dl'].le(hb_threshold).astype(int)
    df = df.merge(demo[['encounterId', 'patientId']], on='encounterId')
    encounter_ids = df['patientId'].unique()
    train_ids, test_ids = train_test_split(encounter_ids, train_size=0.7, random_state=42)
    df_train = df[df['patientId'].isin(train_ids)].copy()
    df_test = df[df['patientId'].isin(test_ids)].copy()
    pat_train, pat_test = df_train[['encounterId', 'patientId']], df_test[['encounterId', 'patientId']]
    df.drop(columns='patientId', inplace=True)
    df_train.drop(columns='patientId', inplace=True)
    df_test.drop(columns='patientId', inplace=True)
    feature_cols = df.columns.difference(['Hb_not_NaN', 'encounterId', 'utcChartTime', 'y'])
    non_numeric_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c not in non_numeric_cols]
    feature_cols_ids = ['encounterId'] + list(feature_cols)
    df_train[feature_cols] = df_train[feature_cols].astype(np.float32)
    df_test[feature_cols] = df_test[feature_cols].astype(np.float32)
    df_train.sort_values(['encounterId', 'utcChartTime'], inplace=True)
    df_test.sort_values(['encounterId', 'utcChartTime'], inplace=True)
    df_train[feature_cols_ids] = df_train.groupby('encounterId')[feature_cols_ids].ffill()
    df_test[feature_cols_ids] = df_test.groupby('encounterId')[feature_cols_ids].ffill()
    df_train[feature_cols_ids] = df_train.groupby('encounterId')[feature_cols_ids].bfill()
    df_test[feature_cols_ids] = df_test.groupby('encounterId')[feature_cols_ids].bfill()
    median_values = df_train[feature_cols].median()
    df_train[feature_cols] = df_train[feature_cols].fillna(median_values)
    df_test[feature_cols] = df_test[feature_cols].fillna(median_values)
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    def add_lags(df):
        df_sorted = df.sort_values(['encounterId', 'utcChartTime']).copy()

        def _create(subdf):
            last_values = subdf[time_variables].iloc[0]
            lag_dfs = []
            for var in time_variables:
                shifts = {f'{var}_lag_{i}': subdf[var].shift(i).fillna(last_values[var]) for i in range(1, sequence_len + 1)}
                lag_dfs.append(pd.DataFrame(shifts, index=subdf.index))
            return pd.concat([subdf, pd.concat(lag_dfs, axis=1)], axis=1)
        return df_sorted.groupby('encounterId', group_keys=False).apply(_create).reset_index(drop=True)

    df_train, df_test = add_lags(df_train), add_lags(df_test)

    def get_future_min_measured(series, mask, window):
        real_values = series.where(mask == 1)
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window)
        return real_values.shift(-1).rolling(window=indexer, min_periods=1).min()

    for d in (df_train, df_test):
        d['future_min_hb'] = d.groupby('encounterId').apply(
            lambda x: get_future_min_measured(x['hemoglobin_g/dl'], x['Hb_not_NaN'], pred_offset)).reset_index(level=0, drop=True)
    df_train = df_train.dropna(subset=['future_min_hb'])
    df_test = df_test.dropna(subset=['future_min_hb'])
    df_train['y_binary'] = (df_train['future_min_hb'] < hb_threshold).astype(int)
    df_test['y_binary'] = (df_test['future_min_hb'] < hb_threshold).astype(int)
    df_train = df_train.drop(columns=['future_min_hb'])
    df_test = df_test.drop(columns=['future_min_hb'])

    def select_specific_lags(df, config):
        time_vars = list(config.keys())
        non_lag_cols = [c for c in df.columns if not any(c.startswith(f"{v}_lag_") for v in time_vars)]
        out = df[non_lag_cols].copy()
        for var, p in config.items():
            existing = [(l, f"{var}_lag_{l}") for l in p['lags'] if f"{var}_lag_{l}" in df.columns]
            if p['mode'] == 'diff':
                for l, c in existing:
                    out[f"{var}_lag_{l}_diff"] = df[c] - df[var]
            elif p['mode'] == 'sum':
                out[f"{var}_lags_{max(p['lags'])}_sum"] = df[[c for _, c in existing]].sum(axis=1)
        return out

    tl, tel = df_train.copy(), df_test.copy()
    df_train, df_test = select_specific_lags(tl, P.LAGS_CONFIG), select_specific_lags(tel, P.LAGS_CONFIG)
    t18, te18 = select_specific_lags(tl, P.SUM_CONFIG), select_specific_lags(tel, P.SUM_CONFIG)
    add = t18.columns.difference(tl.columns)
    df_train[add] = t18[add]
    df_test[add] = te18[add]
    feature_cols = df_train.columns.difference(['Hb_not_NaN', 'encounterId', 'utcChartTime', 'y_binary', 'y_target', 'y'])
    return df_train, df_test, list(feature_cols)


def ref_weights(enc_ids, y, switch_boost_factor=5.0):
    n = len(y); n0 = np.sum(y == 0); n1 = np.sum(y == 1)
    sw = np.where(y == 1, n / (2.0 * n1), n / (2.0 * n0)).astype(float)
    df_sw = pd.DataFrame({'enc': enc_ids, 'y': y})
    idx = []
    for _, grp in df_sw.groupby('enc', sort=False):
        arr = grp['y'].values
        f = np.where((arr[:-1] == 0) & (arr[1:] == 1))[0] + 1
        idx.extend(grp.index[f])
    if idx:
        idx = np.unique(idx)
        sw[idx] *= switch_boost_factor
    return sw


def main():
    df, cis = synthetic.make()
    rtr, rte, rfeats = reference(df, cis)
    res = P.build(df, cis, bfill=True, keep_lags=True)
    assert res["features"] == rfeats, (res["features"], rfeats)
    assert len(rfeats) == 51, len(rfeats)
    for name, ref in [("train", rtr), ("test", rte)]:
        new = res[name]
        assert len(new) == len(ref), (name, len(new), len(ref))
        np.testing.assert_array_equal(new["encounterId"].to_numpy(), ref["encounterId"].to_numpy())
        np.testing.assert_array_equal(new["y_binary"].to_numpy(), ref["y_binary"].to_numpy())
        np.testing.assert_allclose(new[rfeats].to_numpy(np.float64), ref[rfeats].to_numpy(np.float64), rtol=1e-5, atol=1e-4)
        assert res[name + "_seq"].shape == (len(ref), 19, 19)
    w_new = P.transition_weights(res["train"]["encounterId"].to_numpy(), res["train"]["y_binary"].to_numpy())
    w_ref = ref_weights(rtr["encounterId"].to_numpy(), rtr["y_binary"].to_numpy())
    np.testing.assert_allclose(w_new, w_ref)
    print(f"OK: pipeline identical to notebook (train {len(rtr)}, test {len(rte)} samples, {len(rfeats)} features, "
          f"prevalence {rtr['y_binary'].mean():.3f})")


if __name__ == "__main__":
    main()
