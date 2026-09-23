#!/usr/bin/env bash
# Full benchmark on the research server.
#   DATA=/home/lkapral/hb/data OUT=~/hb_benchmark DEVICE=cuda bash benchmark/run_all.sh
# Run inside tmux/screen or with nohup - takes a few hours (LSTM + random forest dominate).
set -euo pipefail
DATA="${DATA:-/home/lkapral/hb/data}"
OUT="${OUT:-$HOME/hb_benchmark}"
DEVICE="${DEVICE:-cuda}"
HERE="$(cd "$(dirname "$0")" && pwd)"
MIMIC_ARGS=()
if [[ -f "$DATA/mimic_df.parquet" ]]; then
  MIMIC_ARGS=(--mimic "$DATA/mimic_df.parquet" --mimic-ids "$HERE/../mimic_ids.csv")
fi
mkdir -p "$OUT"

# 1) main analysis: preprocessing exactly as published
python "$HERE/prepare_features.py" --muw "$DATA/muw_df.parquet" --cis "$DATA/CIS.parquet" \
       "${MIMIC_ARGS[@]}" --out "$OUT/features_published" 2>&1 | tee "$OUT/prepare_published.log"
python "$HERE/run_benchmark.py" --features "$OUT/features_published" --out "$OUT/results_published" \
       --device "$DEVICE" 2>&1 | tee "$OUT/benchmark_published.log"

# 2) sensitivity analysis: no backward filling of predictors (tabular models only)
python "$HERE/prepare_features.py" --muw "$DATA/muw_df.parquet" --cis "$DATA/CIS.parquet" \
       "${MIMIC_ARGS[@]}" --no-bfill --no-seq --out "$OUT/features_nobfill" 2>&1 | tee "$OUT/prepare_nobfill.log"
python "$HERE/run_benchmark.py" --features "$OUT/features_nobfill" --out "$OUT/results_nobfill" \
       --device "$DEVICE" --models last_hb_binary,last_hb_continuous,logreg,xgb,xgb_calibrated_published \
       2>&1 | tee "$OUT/benchmark_nobfill.log"

# 3) table for the manuscript
python "$HERE/make_table.py" --results "$OUT/results_published" --sensitivity "$OUT/results_nobfill" \
       --out "$OUT/Table_S1_model_comparison.docx"

echo
echo "Done. Send back ONLY these aggregated files (no patient data):"
echo "  $OUT/Table_S1_model_comparison.docx / .csv"
echo "  $OUT/results_*/{cv_summary.csv,cv_folds.csv,test_metrics.csv,summary.md,run_info.json}"
echo "  $OUT/features_*/meta.json"
