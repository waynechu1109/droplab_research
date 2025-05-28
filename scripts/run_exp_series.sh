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

for file in arc shoes; do
    for lr in 0.0025; do
        for sparse in 0.0001 0.0002 0.0003 0.0004 0.0005; do
            echo -e "\033[34m$file: hash_normalized_test_sparse${sparse}_${lr}\033[0m"
            ./scripts/experiment.sh "hash_normalized_test_sparse${sparse}_${lr}" "$lr" "$file" schedule $sparse
        done
    done
done

# lr_list=(0.0025 0.0035 0.0045)
# fo_list=(0.05 0.075 0.1 0.125 0.15)
# files=(arc rock shoes dtu church)

# for file in "${files[@]}"; do
#   for lr in "${lr_list[@]}"; do
#     for fo in "${fo_list[@]}"; do
#       tag="exp4_hash_${file}_lr${lr}_fo${fo}"
#       ./scripts/experiment.sh "$tag" "$lr" "$file" schedule "$fo"
#     done
#   done
# done