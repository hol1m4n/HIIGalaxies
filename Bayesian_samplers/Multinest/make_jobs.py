import pandas as pd
from itertools import product

HIIG_dataset_file = [
    "Compilation2026_mockE.csv"
]


'''
HIIG_dataset_file = [
    "Compilation2026_mockD.csv"
]

HIIG_dataset_file = [
    "Compilation2026_mockC.csv"
]


HIIG_dataset_file = [
    "Compilation2026_mockB.csv"
]

HIIG_dataset_file = [
    "Compilation2026_mockA.csv"
]

HIIG_dataset_file = [
    "Compilation2026_Ch12data.csv"
]

HIIG_dataset_file = [
    "Compilation2026_main.csv"
]
'''


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
        "name": "MockE_Local",
        "analysis_mode": "High",
        "zmax": 0.20,
        "label": "z <= 0.20",
    },
    {
        "name": "MockE_Intermediate",
        "analysis_mode": "High",
        "zmax": 1.5,
        "label": "z <= 1.5",
    },
    {
        "name": "MockE_Full",
        "analysis_mode": "High",
        "zmax": None,
        "label": "all z",
    },
]



'''

redshift_regimes = [
    {
        "name": "MockD_Local",
        "analysis_mode": "High",
        "zmax": 0.20,
        "label": "z <= 0.20",
    },
    {
        "name": "MockD_Intermediate",
        "analysis_mode": "High",
        "zmax": 1.5,
        "label": "z <= 1.5",
    },
    {
        "name": "MockD_Full",
        "analysis_mode": "High",
        "zmax": None,
        "label": "all z",
    },
]

******************************************

redshift_regimes = [
    {
        "name": "MockC_Local",
        "analysis_mode": "High",
        "zmax": 0.20,
        "label": "z <= 0.20",
    },
    {
        "name": "MockC_Intermediate",
        "analysis_mode": "High",
        "zmax": 1.5,
        "label": "z <= 1.5",
    },
    {
        "name": "MockC_Full",
        "analysis_mode": "High",
        "zmax": None,
        "label": "all z",
    },
]

******************************************

redshift_regimes = [
    {
        "name": "MockB_Local",
        "analysis_mode": "High",
        "zmax": 0.20,
        "label": "z <= 0.20",
    },
    {
        "name": "MockB_Intermediate",
        "analysis_mode": "High",
        "zmax": 1.5,
        "label": "z <= 1.5",
    },
    {
        "name": "MockB_Full",
        "analysis_mode": "High",
        "zmax": None,
        "label": "all z",
    },
]


******************************************

redshift_regimes = [
    {
        "name": "MockA_Local",
        "analysis_mode": "High",
        "zmax": 0.20,
        "label": "z <= 0.20",
    },
    {
        "name": "MockA_Intermediate",
        "analysis_mode": "High",
        "zmax": 1.5,
        "label": "z <= 1.5",
    },
    {
        "name": "MockA_Full",
        "analysis_mode": "High",
        "zmax": None,
        "label": "all z",
    },
]

******************************************

redshift_regimes = [
    {
        "name": "Liter_Local",
        "analysis_mode": "High",
        "zmax": 0.20,
        "label": "z <= 0.20",
    },
    {
        "name": "Liter_Intermediate",
        "analysis_mode": "High",
        "zmax": 1.5,
        "label": "z <= 1.5",
    },
    {
        "name": "Liter_Full",
        "analysis_mode": "High",
        "zmax": None,
        "label": "all z",
    },
]

******************************************

redshift_regimes = [
    {
        "name": "Main_Local",
        "analysis_mode": "High",
        "zmax": 0.20,
        "label": "z <= 0.20",
    },
    {
        "name": "Main_Intermediate",
        "analysis_mode": "High",
        "zmax": 1.5,
        "label": "z <= 1.5",
    },
    {
        "name": "Main_Full",
        "analysis_mode": "High",
        "zmax": None,
        "label": "all z",
    },
]


******************************************



redshift_regimes = [
    {
        "name": "Local",
        "analysis_mode": "Low",
        "zmax": 0.20,
        "label": "z <= 0.20",
    },
    {
        "name": "Intermediate",
        "analysis_mode": "Moderate",
        "zmax": 1.5,
        "label": "z <= 1.5",
    },
    {
        "name": "Full",
        "analysis_mode": "High",
        "zmax": None,
        "label": "all z",
    },
]

'''







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
            "dataset": HIIG_dataset_file[0],
        }
    )


df = pd.DataFrame(jobs)
df.to_csv("jobs.csv", index=False)

print(f"Created jobs.csv with {len(df)} jobs.")