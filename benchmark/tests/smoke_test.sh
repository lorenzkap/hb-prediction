#!/usr/bin/env bash
# End-to-end run on synthetic data (no patient data needed), ~2-5 min on CPU.
#   bash benchmark/tests/smoke_test.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TMP="$(mktemp -d)"
python "$HERE/synthetic.py" "$TMP"
python "$HERE/../prepare_features.py" --muw "$TMP/muw_df.parquet" --cis "$TMP/CIS.parquet" \
       --mimic "$TMP/mimic_df.parquet" --out "$TMP/features"
python "$HERE/../run_benchmark.py" --features "$TMP/features" --out "$TMP/results" \
       --folds 3 --n-boot 20 --lstm-epochs 2 --lstm-batch 256
python "$HERE/../make_table.py" --results "$TMP/results" --out "$TMP/results/table_S1.docx"
echo "SMOKE TEST OK -> $TMP/results"
