#!/bin/bash

# nohup bash D.1.sh > D.1.log 2>&1 &


anchors_sets=( "Standard.csv" "Joint_eR_recal.csv" "PLRC_all_eR_recal.csv" "PLRC_nZ_eR_recal.csv" "PLRC_nZ_eT_recal.csv" "PLRC_Z_eR_recal.csv" "PLRC_Z_eT_recal.csv" "TRGB_eR_recal.csv" )

cd /home/hollman/HIIGalaxies/Bayesian_samplers/Multinest/MAIN|| exit 1

# Limita los hilos internos de las bibliotecas numéricas.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1



parallel --jobs 9 --tag --halt soon,fail=1 \
    python3 D.py \
    --dataset Compilation2026_main.csv \
    --z_cut 0.16 \
    --sub_id 1 \
    --anchor_data {} \
    ::: "${anchors_sets[@]}" || exit 1

python3 h_look.py --SET D --sub_id 1