#!/bin/bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# EXP_NAME [LR] [FILENAME] [SCHEDULE]"

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

# for file in arc; do
#     for lr in 0.0055; do
#         for sparse in 0.009; do
#             echo -e "\033[34m$file: pe(exp2)_normalized_test_sparse${sparse}_${lr}\033[0m"
#             ./scripts/experiment.sh "pe(exp2)_normalized_test_sparse${sparse}_${lr}" "$lr" "$file" schedule $sparse
#         done
#     done
# done

# dtu_65_47 dtu_73_42 dtu_82_14 dtu_90_41 dtu_110_43 dtu_114_32 dtu shoes


for file in dtu_65_47 dtu_73_42 dtu_82_14 dtu_90_41 dtu_110_43 dtu_114_32 dtu shoes; do
    for lr in 1e-3; do
        for loss_render in 50.0; do
            echo -e "\033[34m$file: pe_inNormalized_a100_3rgb_render10.0~${loss_render}_${lr}\033[0m"
            ./scripts/experiment.sh "pe_inNormalized_a100_3rgb_render10.0~${loss_render}_${lr}" "$lr" "$file" schedule true $loss_render
        done
    done
done