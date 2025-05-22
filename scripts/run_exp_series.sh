#!/bin/bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# EXP_NAME [LR] [FILENAME] [SCHEDULE]"

# for file in arc dtu church mtPres rock shoes; do
#   for para in {6..1}; do
#     name="NeuS_pe${para}_beta100_4.2_.5_.01to.05_[]_.05_adamw_cosAnn_narrow_band_msk0.05"
#     ./experiment.sh "$name" 2500 0.005 0.01 "$para" "$file"
#   done
# done

# for file in arc dtu church rock shoes; do
#   for para in {6..12}; do
#     name="NeuS_pe0to${para}_beta100_4.2_.5_.01to.05_[]_.05_1_adamw_cosAnn_narrow_band_msk0.05"
#     ./experiment.sh "$name" 2500 0.005 0.01 "$para" "$file"
#   done
# done

# for file in arc dtu church rock shoes; do
#     for lr in 0.0045 0.004 0.0035 0.003; do
#         ./scripts/experiment.sh "two_stage_test_6_$lr" "$lr" "$file" training_schedule
#     done
# done

for file in arc; do
    for lr in 0.0025; do
        ./scripts/experiment.sh "two_stage_test_hash_$lr" "$lr" "$file" schedule
    done
done

# lr_list=(0.0025 0.0035 0.0045)
# consistency_list=(0.001 0.005 0.01)
# files=(arc rock shoes dtu church)

# for file in "${files[@]}"; do
#   for lr in "${lr_list[@]}"; do
#     for cons in "${consistency_list[@]}"; do
#       tag="exp_${file}_lr${lr}_cons${cons}"
#       sched_path="cons_${cons}"
#       ./scripts/experiment.sh "$tag" "$lr" "$file" "$sched_path"
#     done
#   done
# done