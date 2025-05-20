#!/usr/bin/env bash
# command: ./experiment.sh EXP_NAME [EPOCHS] [LR] [SIGMA]

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 EXP_NAME [LR] [FILENAME] [SCHEDULE]"
  exit 1
fi

# exp. parameters 
EXP_NAME=$1
# EPOCHS=${2:-5000}
LR=${2:-0.005}
# SIGMA=${4:-0.01}
# PARAMETER=$3
FILEMANE=$3
SCHEDULE=$4

# dir. setting
LOG_DIR="log"
CKPT_DIR="ckpt"
OUT_DIR="output"
SCHE_DIR="schedule"

mkdir -p "$LOG_DIR" "$CKPT_DIR" "$OUT_DIR"

# 1) training
python3 train.py \
  --lr "$LR" \
  --desc "$EXP_NAME" \
  --log_path "$LOG_DIR/sdf_model_${EXP_NAME}_${FILEMANE}.txt" \
  --ckpt_path "$CKPT_DIR/sdf_model_${EXP_NAME}_${FILEMANE}.pt" \
  --file_name "$FILEMANE" \
  --schedule_path "$SCHE_DIR/${SCHEDULE}.json"
echo "[1/3] Fininsh Training"

# 2) inference
python3 inference.py \
  --res 300 \
  --ckpt_path "$CKPT_DIR/sdf_model_${EXP_NAME}_${FILEMANE}.pt" \
  --output_mesh "$OUT_DIR/sdf_model_${EXP_NAME}_${FILEMANE}.ply" \
  --file_name "$FILEMANE"
echo "[2/3] Fininsh Inferencing"

# 3) plotting loss courve
# python3 tools/plot_loss.py --log "$LOG_DIR/sdf_model_${EPOCHS}_${EXP_NAME}.txt"
# echo "[3/3] Fininsh Plotting Loss Curve"



# ./experiment.sh NeuS_pe6_beta100_4.2_.5_.01to.05_[]_.05_1_adamw_cosAnn_narrow_band_msk0.05 2500 0.005 0.01
