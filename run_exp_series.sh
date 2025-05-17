#!/bin/bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

./experiment.sh NeuS_pe6_beta100_4.2_.5_.01to.05_[]_.05_1_adamw_cosAnn_narrow_band_msk0.05 2500 0.005 0.01 1.0
# ./experiment.sh NeuS_pe6_beta100_4.2_.5_.01to.05_[]_.05_.5_adamw_cosAnn_narrow_band_msk0.05 2500 0.005 0.01 0.5
# ./experiment.sh NeuS_pe6_beta100_4.2_.5_.01to.05_[]_.05_.1_adamw_cosAnn_narrow_band_msk0.05 2500 0.005 0.01 0.1
# ./experiment.sh NeuS_pe6_beta100_4.2_.5_.01to.05_[]_.05_.08_adamw_cosAnn_narrow_band_msk0.05 2500 0.005 0.01 0.08
# ./experiment.sh NeuS_pe6_beta100_4.2_.5_.01to.05_[]_.05_.05_adamw_cosAnn_narrow_band_msk0.05 2500 0.005 0.01 0.05
# ./experiment.sh NeuS_pe6_beta100_4.2_.5_.01to.05_[]_.05_.03_adamw_cosAnn_narrow_band_msk0.05 2500 0.005 0.01 0.03
# ./experiment.sh NeuS_pe6_beta100_4.2_.5_.01to.05_[]_.05_.01_adamw_cosAnn_narrow_band_msk0.05 2500 0.005 0.01 0.01