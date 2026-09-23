# Model benchmark (reviewer request: algorithm justification, cross-validation, feature handling)

This folder compares candidate model classes **on identical data**. It uses the same outcome definition, preprocessing, patient-level split and patient-grouped cross-validation for every model. It re-implements the preprocessing of `modelling_regression-MUW_only_measured.ipynb` as a script (vectorised, so it is faster). `tests/test_equivalence.py` shows that it produces exactly the same samples, labels, 51 features and training weights as the notebook.

| Model | Input | Training |
|---|---|---|
| Most recent Hb < 8 g/dL | last Hb value | none (the reference reported in the paper) |
| Most recent Hb, continuous | last Hb value | none (stronger reference) |
| Logistic regression (L2) | 51 engineered features | class/transition weights |
| Random forest | 51 engineered features | class/transition weights |
| XGBoost | 51 engineered features | published hyperparameters, class/transition weights |
| XGBoost + isotonic calibration | 51 engineered features | **exactly the published final model** (`CalibratedClassifierCV(..., cv=5)`, unweighted refit) |
| LSTM (64/32) | raw hourly sequence, 19 h × 19 variables | class/transition weights, early stopping on a 10 % patient-level validation split |

The run produces:

1. **Patient-grouped 5-fold CV** within the training set (`GroupKFold` on `patientId`): AUROC, AUPRC and Brier score as mean ± SD.
2. **Refit on the full training set**, then evaluation on the internal test set, and on MIMIC-IV if `mimic_df.parquet` is present, with patient-level bootstrap 95 % CIs.
3. **Sensitivity analysis without backward filling.** The published notebook back-fills predictors within an encounter (`groupby().bfill()`), so values measured *after* a prediction time can reach earlier samples. `--no-bfill` removes this, and the benchmark reports the effect.

## Run on the research server

```bash
cd ~/hb-prediction            # your clone of this repository
git pull
git checkout model-benchmark  # skip once the branch is merged into main
pip install -r benchmark/requirements.txt   # inside the env that already has xgboost/tensorflow

# optional, 2-5 min, synthetic data only
bash benchmark/tests/smoke_test.sh

# full run (a few hours; use tmux/screen or nohup)
DATA=/home/lkapral/hb/data OUT=~/hb_benchmark DEVICE=cuda nohup bash benchmark/run_all.sh > ~/hb_benchmark.log 2>&1 &
tail -f ~/hb_benchmark.log
```

`DATA` must contain `muw_df.parquet` and `CIS.parquet`. `mimic_df.parquet` is optional; it adds the external evaluation. For MIMIC, this script imputes with ViennaAIdb training medians, so its XGBoost result can differ slightly from the published 0.91.

Single steps and options:

```bash
python benchmark/prepare_features.py --muw $DATA/muw_df.parquet --cis $DATA/CIS.parquet --out ~/hb_benchmark/features_published
python benchmark/run_benchmark.py --features ~/hb_benchmark/features_published --out ~/hb_benchmark/results_published \
       --device cuda --models logreg,xgb,lstm      # subset of models
       # --no-cv  (test set only) | --folds 5 | --n-boot 200 | --lstm-epochs 20 | --cv-calibrated
python benchmark/make_table.py --results ~/hb_benchmark/results_published --out ~/hb_benchmark/Table_S1.docx
```

Sanity check: `xgb_calibrated_published` should reproduce the published internal test AUROC (0.9406) when run on `features_published`.

## What to share

`features_*/train.parquet`, `test.parquet` and `*.npy` contain **patient-level data**. They are written outside the repository (`~/hb_benchmark`), are covered by `.gitignore`, and must not be copied off the server.

Only these aggregated files can be shared:

- `Table_S1_model_comparison.docx` / `.csv`
- `results_*/cv_summary.csv`, `cv_folds.csv`, `test_metrics.csv`, `summary.md`, `run_info.json`
- `features_*/meta.json` (counts, prevalence, feature list, training medians)
