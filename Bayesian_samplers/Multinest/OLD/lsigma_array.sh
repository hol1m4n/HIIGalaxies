#!/bin/bash
#SBATCH --job-name=LsHo
#SBATCH --output=/home/holman/HIIGalaxies/Bayesian_samplers/Multinest/logs_Ho/LsHo_%A_%a.out
#SBATCH --error=/home/holman/HIIGalaxies/Bayesian_samplers/Multinest/logs_Ho/LsHo_%A_%a.err
#SBATCH --array=0-14%10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --account=default
#SBATCH --partition=N1Mitad1

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on node: $(hostname)"
echo "Starting at: $(date)"

#mkdir -p logs_Ho

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate nest

echo "Python/conda activo:"
which python || true
conda info --envs || true

workplace="/home/holman/HIIGalaxies/Bayesian_samplers/Multinest"

cd "$workplace" || { echo "ERROR: no pude entrar a $workplace"; exit 1; }

echo "Corriendo el codigo(array):"

python3 run_lsigma_job.py ${SLURM_ARRAY_TASK_ID}

echo "Finished at: $(date)"