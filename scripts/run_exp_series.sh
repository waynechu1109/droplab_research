#!/bin/bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# EXP_NAME [LR] [FILENAME] [SCHEDULE]"

# for file in arc shoes; do
#     for lr in 0.0025; do
#         for sparse in 0.0082 0.0084 0.0086 0.0088; do
#             echo -e "\033[34m$file: pe_normalized_test_sparse${sparse}_${lr}\033[0m"
#             ./scripts/experiment.sh "pe_normalized_test_sparse${sparse}_${lr}" "$lr" "$file" schedule $sparse
#         done
#     done
# done

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

for file in shoes arc dtu; do
    for lr in 0.0025 0.0035 0.0045 0.0055; do
        for sparse in 0.008 0.0082 0.0084 0.0086 0.0088 0.009; do
            echo -e "\033[34m$file: pe_normalized_test_sparse${sparse}_${lr}\033[0m"
            ./scripts/experiment.sh "pe_normalized_test_sparse${sparse}_${lr}" "$lr" "$file" schedule $sparse
        done
    done
done