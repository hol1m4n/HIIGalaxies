#!/bin/bash
#SBATCH --job-name=LCDM_Ta
#SBATCH --output=/home/holman/bash_outs/Cobaya/LCDM_Ta.out
#SBATCH --error=/home/holman/bash_outs/Cobaya/LCDM_Ta.err
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=7-00:00:00
#SBATCH --account=default
#SBATCH --partition=N1Mitad1


source /opt/anaconda3/etc/profile.d/conda.sh
conda activate class

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Muy útil si vuelve a ocurrir un SIGSEGV
export PYTHONFAULTHANDLER=1


echo "Python:"
which python
python --version

echo "MPI:"
which mpirun
mpirun --version

python - <<'PY'
import cobaya
import mpi4py
from mpi4py import MPI

print("Cobaya:", cobaya.__version__)
print("mpi4py:", mpi4py.__version__)
print("MPI library:")
print(MPI.Get_library_version())
PY

export PYTHONPATH="/home/holman/HIIGalaxies/Bayesian_samplers/Cobaya/MAIN/hiig_lensing_cobaya:$PYTHONPATH"

echo "Python/conda activo:"
which python || true
conda info --envs || true

workplace="/home/holman/HIIGalaxies/Bayesian_samplers/Cobaya/MAIN/hiig_lensing_cobaya"

cd "$workplace" || { echo "ERROR: no pude entrar a $workplace"; exit 1; }

echo "Corriendo el codigo:"

mpirun -np 3 cobaya-run LCDM/cmb_bao.yaml --resume

echo "Finished at: $(date)"