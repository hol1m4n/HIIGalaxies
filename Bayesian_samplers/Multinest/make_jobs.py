import pandas as pd
from itertools import product


calibrator_sets = [
    {
        "name": "Cep-nZeR",
        "file": "sm_r.csv",
    },
    {
        "name": "Cep-cZeR",
        "file": "cm_r.csv",
    },
    {
        "name": "Cep-nZeT",
        "file": "sm_t.csv",
    },
    {
        "name": "Cep-cZeT",
        "file": "cm_t.csv",
    },
    {
        "name": "TRGB",
        "file": "t_r.csv",
    },
]



error_kinds = [
    "sigma_w"
]

redshift_regimes = [
    {
        "name": "Local",
        "analysis_mode": "Low",
        "zmax": 0.10,
        "label": "z <= 0.10",
    },
    {
        "name": "Intermediate",
        "analysis_mode": "Moderate",
        "zmax": 0.5,
        "label": "z <= 0.5",
    },
    {
        "name": "Full",
        "analysis_mode": "High",
        "zmax": None,
        "label": "all z",
    },
]


jobs = []

for calibrator, err, regime in product(calibrator_sets, error_kinds, redshift_regimes):

    if regime["zmax"] is None:
        ztag = "allz"
        zmax = "None"
    else:
        ztag = f"zle{str(regime['zmax']).replace('.', 'p')}"
        zmax = regime["zmax"]

    folder_name = (
        f"{regime['name']}_"
        f"{calibrator['name']}_"
        f"{err}"
    ).replace(" ", "_").replace("-", "")

    id_prefix = (
        f"{regime['analysis_mode'].lower()}_"
        f"{calibrator['name'].lower()}_"
        f"{err}_"
        f"{ztag}"
    ).replace("-", "").replace(" ", "_")

    main_title = (
        rf"L-sigma test: {regime['name']} Universe "
        rf"({regime['label']}), "
        rf"Set: {calibrator['name']}, "
        rf"error: {err}"
    )

    jobs.append(
        {
            "calibrator_name": calibrator["name"],
            "distance_estimator_set": calibrator["file"],
            "estimator_error_kind": err,
            "regime_name": regime["name"],
            "analysis_mode": regime["analysis_mode"],
            "zmax": zmax,
            "main_title": main_title,
            "folder_name": folder_name,
            "id_prefix": id_prefix,
        }
    )


df = pd.DataFrame(jobs)
df.to_csv("jobs.csv", index=False)

print(f"Created jobs.csv with {len(df)} jobs.")