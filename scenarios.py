import numpy as np

SCENARIOS: list = [
    {
        "simulation": "task 1, controllers comparison",
        "common": {
            "ini_type": "down",
            "ref_type": "swing-up",
        },
        "specific": [
            {"mod_type_ocp": "non-linear", "Q": np.diag([10, 10, 0.1, 0.1]), "R": 0.01},
            {
                "mod_type_ocp": "extended",
                "Q": np.diag([10, 10, 0.1, 0.1, 0.01]),
                "R": 0.01,
            },
        ],
    },
    {
        "simulation": "task 1, default non-linear",
        "common": {
            "ini_type": "down",
            "ref_type": "swing-up",
            "mod_type_ocp": "non-linear",
        },
        "specific": [
            {"Q": np.diag([10, 10, 0.1, 0.1]), "R": 0.01},
            {"Q": np.diag([10, 10, 0.1, 0.1]), "R": 0.10},
            {"Q": np.diag([10, 10, 0.1, 0.1]), "R": 0.20},
        ],
    },
    {
        "simulation": "task 1, extended controlloer",
        "common": {
            "ini_type": "down",
            "ref_type": "swing-up",
            "mod_type_ocp": "extended",
        },
        "specific": [
            {"Q": np.diag([10, 10, 0.1, 0.1, 0.01]), "R": 0.01},
            {"Q": np.diag([10, 10, 0.1, 0.1, 0.10]), "R": 0.01},
            {"Q": np.diag([10, 10, 0.1, 0.1, 0.20]), "R": 0.01},
        ],
    },
    {
        "simulation": "task 2, (only) simulate default system",
        "common": {
            "ini_type": "off-balance",
            "ctrl_on": False,
            "mod_type_sim": "non-linear",
            "ref_type": "empty-2s",
            "Ts": 1,  # this allows to pass (unnecessary) the checks on controller sampling
        },
        "specific": [
            {"ts_sim": 1e-5, "integ_type_sim": "ERK"},
            {"ts_sim": 1e-4, "integ_type_sim": "ERK"},
            {"ts_sim": 1e-3, "integ_type_sim": "ERK"},
            {"ts_sim": 1e-2, "integ_type_sim": "ERK"},
            {"ts_sim": 1e-1, "integ_type_sim": "ERK"},
            {"ts_sim": 1e-5, "integ_type_sim": "IRK"},
            {"ts_sim": 1e-4, "integ_type_sim": "IRK"},
            {"ts_sim": 1e-3, "integ_type_sim": "IRK"},
            {"ts_sim": 1e-2, "integ_type_sim": "IRK"},
            {"ts_sim": 1e-1, "integ_type_sim": "IRK"},
        ],
    },
    {
        "simulation": "task 2, (only) simulate system with spring",
        "common": {
            "ini_type": "off-balance",
            "ctrl_on": False,
            "mod_type_sim": "spring",
            "ref_type": "empty-2s",
            "Ts": 1,  # this allows to pass (unnecessary) the checks on controller sampling
        },
        "specific": [
            {"ts_sim": 1e-5, "integ_type_sim": "ERK"},
            {"ts_sim": 1e-4, "integ_type_sim": "ERK"},
            {"ts_sim": 1e-3, "integ_type_sim": "ERK"},
            {"ts_sim": 1e-2, "integ_type_sim": "ERK"},
            {"ts_sim": 1e-1, "integ_type_sim": "ERK"},
            {"ts_sim": 1e-5, "integ_type_sim": "IRK"},
            {"ts_sim": 1e-4, "integ_type_sim": "IRK"},
            {"ts_sim": 1e-3, "integ_type_sim": "IRK"},
            {"ts_sim": 1e-2, "integ_type_sim": "IRK"},
            {"ts_sim": 1e-1, "integ_type_sim": "IRK"},
        ],
    },
    {
        "simulation": "task 2, default non-linear, higher sampling time",
        "common": {"ini_type": "down", "ref_type": "swing-up", "Ts": 0.1, "N": 20},
        "specific": [
            {"integ_type_ocp": "ERK"},
            {"integ_type_ocp": "IRK"},
        ],
    },
    {
        "simulation": "task 3, NMPC vs linear MPC",
        "common": {"ini_type": "up", "ref_type": "horizontal"},
        "specific": [
            {"mod_type_ocp": "linear"},
            {"mod_type_ocp": "non-linear"},
        ],
    },
    {
        "simulation": "task 3, swing-up maneuver with linear MPC",
        "common": {"ini_type": "down", "ref_type": "swing-up"},
        "specific": [
            {"mod_type_ocp": "linear"},
            {"mod_type_ocp": "non-linear"},
        ],
    },
]
