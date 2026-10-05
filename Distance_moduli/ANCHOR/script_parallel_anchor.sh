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


anchors=("LMC" "GC" "LMC+GC" "N4258" "LMC+GC+N4258" "TM")


cd /home/hollman/HIIGalaxies/Distance_moduli/ANCHOR/ || exit 1

# Limita los hilos internos de las bibliotecas numéricas.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

for Subgrupo in \
    nZ_eR_ \
    nZ_eT_ \
    Z_eR_ \
    Z_eT_
do

    tipo_de_error="random_error"

    if [[ "$Subgrupo" == "nZ_eT_" ||
          "$Subgrupo" == "Z_eT_" ]]; then
        tipo_de_error="total_error"
    fi

    for ANCHOR in "${anchors[@]}"; do

        file_name="${Subgrupo}${ANCHOR}.csv"

        if [ -f "$file_name" ]; then
            echo "=================================="
            echo "Procesando subgrupo: $Subgrupo con $ANCHOR"

            parallel --jobs 12 --tag --halt soon,fail=1 \
                python3 stats_calculator.py \
                --DATASET "$file_name" \
                --TIPO_E "$tipo_de_error" \
                --GALAXIA {} \
                ::: "${galaxias[@]}" || exit 1

            echo "$Subgrupo con $ANCHOR terminado"
            echo "=================================="
            echo -e "\n"
        else
            echo "Omitido: no existe $file_name" >&2
        fi
    done
done