#!/bin/bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

./experiment.sh NeuS_pe6_beta100_4.2_.5_.01to.05_[]_.05_1_adamw_cosAnn_narrow_band_msk0.05 2500 0.005 0.01 6 arc
./experiment.sh NeuS_pe5_beta100_4.2_.5_.01to.05_[]_.05_1_adamw_cosAnn_narrow_band_msk0.05 2500 0.005 0.01 5 arc
./experiment.sh NeuS_pe4_beta100_4.2_.5_.01to.05_[]_.05_1_adamw_cosAnn_narrow_band_msk0.05 2500 0.005 0.01 4 arc
./experiment.sh NeuS_pe3_beta100_4.2_.5_.01to.05_[]_.05_1_adamw_cosAnn_narrow_band_msk0.05 2500 0.005 0.01 3 arc
./experiment.sh NeuS_pe2_beta100_4.2_.5_.01to.05_[]_.05_1_adamw_cosAnn_narrow_band_msk0.05 2500 0.005 0.01 2 arc
./experiment.sh NeuS_pe1_beta100_4.2_.5_.01to.05_[]_.05_1_adamw_cosAnn_narrow_band_msk0.05 2500 0.005 0.01 1 arc

./experiment.sh NeuS_pe6_beta100_4.2_.5_.01to.05_[]_.05_1_adamw_cosAnn_narrow_band_msk0.05 2500 0.005 0.01 6 dtu
./experiment.sh NeuS_pe5_beta100_4.2_.5_.01to.05_[]_.05_1_adamw_cosAnn_narrow_band_msk0.05 2500 0.005 0.01 5 dtu
./experiment.sh NeuS_pe4_beta100_4.2_.5_.01to.05_[]_.05_1_adamw_cosAnn_narrow_band_msk0.05 2500 0.005 0.01 4 dtu
./experiment.sh NeuS_pe3_beta100_4.2_.5_.01to.05_[]_.05_1_adamw_cosAnn_narrow_band_msk0.05 2500 0.005 0.01 3 dtu
./experiment.sh NeuS_pe2_beta100_4.2_.5_.01to.05_[]_.05_1_adamw_cosAnn_narrow_band_msk0.05 2500 0.005 0.01 2 dtu
./experiment.sh NeuS_pe1_beta100_4.2_.5_.01to.05_[]_.05_1_adamw_cosAnn_narrow_band_msk0.05 2500 0.005 0.01 1 dtu