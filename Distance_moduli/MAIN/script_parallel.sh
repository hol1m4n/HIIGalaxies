#!/bin/bash

galaxias=(
    "IC0010" "IC2574" "M33" "M81" "M101" "MRK116"
    "NGC0925" "NGC2366" "NGC2403" "NGC2541" "NGC3198"
    "NGC3319" "NGC4214" "NGC4236" "NGC4258" "NGC4395"
    "NGC6822" "NGC1073" "NGC2500" "NGC3184" "M96"
    "NGC3370" "M66" "NGC4414" "NGC4496A" "NGC4535"
    "NGC4536" "NGC4725" "UGC08091" "NGC5204" "UGC09128"
    "NGC5584" "NGC7331"
)

cd /home/hollman/HIIGalaxies/Distance_moduli/MAIN/ || exit 1

# Limita los hilos internos de las bibliotecas numéricas.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

for Subgrupo in \
    Joint_eR.csv \
    PLRC_all_eR.csv \
    PLRC_nZ_eR.csv \
    PLRC_nZ_eT.csv \
    PLRC_Z_eR.csv \
    PLRC_Z_eT.csv \
    TRGB_eR.csv
do
    echo "=================================="
    echo "Procesando subgrupo: $Subgrupo"

    tipo_de_error="random_error"

    if [[ "$Subgrupo" == "PLRC_nZ_eT.csv" ||
          "$Subgrupo" == "PLRC_Z_eT.csv" ]]; then
        tipo_de_error="total_error"
    fi

    parallel --jobs 12 --tag --halt soon,fail=1 \
        python3 stats_calculator.py \
        --DATASET "$Subgrupo" \
        --TIPO_E "$tipo_de_error" \
        --GALAXIA {} \
        ::: "${galaxias[@]}" || exit 1

    echo "Subgrupo terminado: $Subgrupo"
    echo "=================================="
    echo -e "\n"
done