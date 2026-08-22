#!/bin/bash
# Run the full dataset publication pipeline for all 20 drivers
# Can be interrupted and resumed - pipeline skips already-completed sessions
PYTHON="${PYTHON:-python}"
SCRIPT="$(dirname "$0")/prepare_dataset.py"
LOGFILE="$(dirname "$0")/pipeline_run_$(date +%Y%m%d_%H%M%S).log"

echo "Starting full pipeline run at $(date)" | tee "$LOGFILE"
echo "Log: $LOGFILE"

PYTHONIOENCODING=utf-8 "$PYTHON" -u "$SCRIPT" 2>&1 | tee -a "$LOGFILE"

echo "Pipeline completed at $(date)" | tee -a "$LOGFILE"
