#!/usr/bin/env bash
# command: ./experiment.sh EXP_NAME [EPOCHS] [LR] [SIGMA]

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 EXP_NAME [LR] [FILENAME] [SCHEDULE] [IS_A100] [PARAMETER]"
  exit 1
fi

# exp. parameters 
EXP_NAME=$1
LR=${2:-0.005}
FILENAME=$3
SCHEDULE=$4
IS_A100=$5
# PARAMETER=$6

# dir. setting
LOG_DIR="log"
CKPT_DIR="ckpt"
OUT_DIR="output"
SCHE_DIR="schedule"

mkdir -p "$LOG_DIR/$EXP_NAME/$FILENAME" "$CKPT_DIR/$EXP_NAME/$FILENAME" "$OUT_DIR/$EXP_NAME/$FILENAME"

# 1) training
python3 train.py \
  --lr "$LR" \
  --desc "$EXP_NAME" \
  --log_path "$LOG_DIR/$EXP_NAME/$FILENAME/sdf_model_${EXP_NAME}_${FILENAME}.txt" \
  --ckpt_path "$CKPT_DIR/$EXP_NAME/$FILENAME/sdf_model_${EXP_NAME}_${FILENAME}" \
  --file_name "$FILENAME" \
  --schedule_path "$SCHE_DIR/${SCHEDULE}.json" \
  --is_a100 "$IS_A100" 
  # --para "$PARAMETER"
echo -e "\033[32m[1/2] Finish Training\033[0m"

# 2) inference
epochs=("epoch1500" "epoch2000" "final")

for epoch in "${epochs[@]}"; do
  python3 inference.py \
    --res 256 \
    --ckpt_path "$CKPT_DIR/$EXP_NAME/$FILENAME/sdf_model_${EXP_NAME}_${FILENAME}_${epoch}.pt" \
    --output_mesh "$OUT_DIR/$EXP_NAME/$FILENAME/sdf_model_${EXP_NAME}_${FILENAME}_${epoch}.ply" \
    --file_name "$FILENAME"
  echo -e "\033[32m[2/2] Finish Inferencing ${FILENAME} ${epoch}\033[0m"
done