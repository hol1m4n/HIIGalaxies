#!/bin/bash


#source /opt/anaconda3/etc/profile.d/conda.sh
source activate class

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

echo "Python/conda activo:"
which python || true
conda info --envs || true

workplace="/home/hollman/HIIGalaxies/Bayesian_samplers/Cobaya/MAIN/pruebas_lsigma_cobaya"

cd "$workplace" || { echo "ERROR: no pude entrar a $workplace"; exit 1; }

echo "Corriendo el codigo:"

mpirun -np 4 cobaya-run replica_astropy_model1.yaml # --resume

echo "Finished at: $(date)"



