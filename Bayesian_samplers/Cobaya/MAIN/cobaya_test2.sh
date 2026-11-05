#!/bin/bash
#SBATCH --job-name=TRGB
#SBATCH --output=slurm_cobaya2.out
#SBATCH --error=slurm_cobaya2.err
#SBATCH --cpus-per-task=5
#SBATCH --mem=10G
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --account=default
#SBATCH --partition=N1Mitad1


source /opt/anaconda3/etc/profile.d/conda.sh
conda activate class

echo "Python/conda activo:"
which python || true
conda info --envs || true

workplace="/home/holman/HIIGalaxies/Bayesian_samplers/Cobaya/Lsigma/cobaya_lsigma_package"

cd "$workplace" || { echo "ERROR: no pude entrar a $workplace"; exit 1; }

echo "Corriendo el codigo:"

#mpirun -np 4 cobaya-run yaml/wCDM/planck_hiig_trgb.yaml --resume

mpirun -np 4 cobaya-run yaml/w0waCDM/planck_hiig_trgb.yaml

echo "Finished at: $(date)"