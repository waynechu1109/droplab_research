#!/bin/bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# dtu_65_47 dtu_73_42 dtu_82_14 dtu_90_41 dtu_110_43 dtu_114_32 dtu shoes


# for file in dtu_65_47 dtu_73_42 dtu_82_14 dtu_90_41 dtu_110_43 dtu_114_32 dtu shoes; do
#     for lr in 1e-3; do
#         for loss_render in 5.0; do
#             echo -e "\033[34m$file: pe_normalized_a100_8192rays_render10.0~${loss_render}_${lr}\033[0m"
#             ./scripts/experiment.sh "pe_normalized_a100_8192rays_render10.0~${loss_render}_${lr}" "$lr" "$file" schedule true $loss_render
#         done
#     done
# done

# for file in dtu_65_47 dtu_73_42 dtu_82_14 dtu_90_41 dtu_110_43 dtu_114_32 dtu shoes; do
#     for lr in 1e-3; do
#         for loss_render in 5.0; do
#             echo -e "\033[34m$file: new_pe_normalized_render10.0~${loss_render}_${lr}\033[0m"
#             ./scripts/experiment.sh "new_pe_normalized_render10.0~${loss_render}_${lr}" "$lr" "$file" schedule false $loss_render
#         done
#     done
# done

# dtu_90_41 dtu_110_43 dtu_65_47 dtu_73_42 dtu_82_14 dtu_114_32 dtu shoes

for file in my_shelf my_room; do
    for lr in 10e-3; do
        echo -e "\033[34m$file: 6d_tunemodel_sdftarget1_colorGeo_pe_normalized_${lr}\033[0m"
        ./scripts/experiment.sh "6d_tunemodel_sdftarget1_colorGeo_pe_normalized_${lr}" "$lr" "$file" schedule false
    done
done

# for file in dtu_65_47 dtu_73_42 dtu_82_14 dtu_90_41 dtu_110_43 dtu_114_32 dtu shoes; do
#     for lr in 1e-3; do
#         echo -e "\033[34m$file: 3stage_pe_normalized_${lr}\033[0m"
#         ./scripts/experiment.sh "3stage_pe_normalized_${lr}" "$lr" "$file" schedule true
#     done
# done