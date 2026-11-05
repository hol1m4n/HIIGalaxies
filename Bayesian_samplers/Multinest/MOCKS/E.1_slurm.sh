#!/bin/bash
#SBATCH --job-name=MockE
#SBATCH --output=/home/holman/bash_outs/Mocks/MockE_%A_%a.out
#SBATCH --error=/home/holman/bash_outs/Mocks/MockE_%A_%a.err
#SBATCH --array=0-100%12
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=23:00:00
#SBATCH --account=default
#SBATCH --partition=N1Mitad1
#SBATCH --begin=now+16hours

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate nest


cd "$HOME/HIIGalaxies/Bayesian_samplers/Multinest/MOCKS" || exit 1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

#anchors_sets=( "Standard.csv" "Joint_eR_recal.csv" "PLRC_all_eR_recal.csv" "PLRC_nZ_eR_recal.csv" "PLRC_nZ_eT_recal.csv" "PLRC_Z_eR_recal.csv" "PLRC_Z_eT_recal.csv" "TRGB_eR_recal.csv" )

#anchors_sets=( "Joint_eR_recal.csv" "PLRC_all_eR_recal.csv"  "TRGB_eR_recal.csv" )

#current_anchor="${anchors_sets[$SLURM_ARRAY_TASK_ID]}"

for anchor_set in \
    Joint_eR_recal.csv \
    PLRC_all_eR_recal.csv \
    TRGB_eR_recal.csv
do
    echo "=================================="
    echo "Procesando grupo ancla: $anchor_set"

    echo "Ejecutando para el anchor: $anchor_set (Índice: $SLURM_ARRAY_TASK_ID)"

    python3 E.py \
        --dataset Compilation2026_whole.csv \
        --z_cut 0.16 \
        --sub_id $SLURM_ARRAY_TASK_ID \
        --anchor_data "$anchor_set" || exit 1

    python3 reader_clean.py --SET E --sub_id $SLURM_ARRAY_TASK_ID --anchor_group "$anchor_set" || exit 1
done


