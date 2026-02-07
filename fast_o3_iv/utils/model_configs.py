"""Climate model configs"""

# Maps CESM2 variable names -> CMIP6 variable names
CESM2_TO_CMIP6_MAP = {
    "TS": "ts",
    "QREFHT": "huss",
    "PSL": "psl",
    "PRECT": "pr",
    "FSDS": "rsds",
    "U": "ua",
    "V": "va",
    "Z3_500": "zg500",
    "PS": "ps",
}

MODEL_CONFIG = {
    "CESM2-LE": {"institution_id": "NCAR", "variable_map": CESM2_TO_CMIP6_MAP},
    "CESM2-WACCM6": {"institution_id": "NCAR", "variable_map": CESM2_TO_CMIP6_MAP},
}