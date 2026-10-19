#!/bin/bash
#SBATCH --job-name=E.2
#SBATCH --output=/home/holman/bash_outs/ABCDEFG/E.2_%A_%a.out
#SBATCH --error=/home/holman/bash_outs/ABCDEFG/E.2_%A_%a.err
#SBATCH --array=0-7%9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=20:00:00
#SBATCH --account=default
#SBATCH --partition=N1Mitad1

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate nest


cd "$HOME/HIIGalaxies/Bayesian_samplers/Multinest/MAIN" || exit 1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

anchors_sets=( "Standard.csv" "Joint_eR_recal.csv" "PLRC_all_eR_recal.csv" "PLRC_nZ_eR_recal.csv" "PLRC_nZ_eT_recal.csv" "PLRC_Z_eR_recal.csv" "PLRC_Z_eT_recal.csv" "TRGB_eR_recal.csv" )

current_anchor="${anchors_sets[$SLURM_ARRAY_TASK_ID]}"

echo "Ejecutando para el anchor: $current_anchor (Índice: $SLURM_ARRAY_TASK_ID)"

python3 E.py \
    --dataset Compilation2026_main.csv \
    --z_cut 16 \
    --sub_id 2 \
    --anchor_data "$current_anchor" || exit 1
