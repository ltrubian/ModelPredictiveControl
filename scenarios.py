import numpy as np

SCENARIOS: list = [
    # ################################################
    # #################### TASK 1 ####################
    # ################################################
    {
        "simulation": "task 1, controllers comparison",
        "common": {
            "ini_type": "down",
            "ref_type": "swing-up",
            "R": 0.01,
        },
        "specific": [
            {"mod_type_ocp": "non-linear", "Q": np.diag([10, 10, 0.1, 0.1])},
            {
                "mod_type_ocp": "extended",
                "Q": np.diag([10, 10, 0.1, 0.1, 0.01]),
            },
        ],
        "labels": ["Non Linear", "Extended"],
    },
    {
        "simulation": "task 1, default non-linear",
        "common": {
            "ini_type": "down",
            "ref_type": "swing-up",
            "mod_type_ocp": "non-linear",
            "Q": np.diag([10, 10, 0.1, 0.1]),
        },
        "specific": [
            {"R": 0.01},
            {"R": 0.10},
            {"R": 0.20},
        ],
        "labels": [str(x) for x in [0.01, 0.1, 0.2]],
    },
    {
        "simulation": "task 1, extended controlloer",
        "common": {
            "ini_type": "down",
            "ref_type": "swing-up",
            "mod_type_ocp": "extended",
            "R": 0.01,
        },
        "specific": [
            {"Q": np.diag([10, 10, 0.1, 0.1, 0.01])},
            {"Q": np.diag([10, 10, 0.1, 0.1, 0.10])},
            {"Q": np.diag([10, 10, 0.1, 0.1, 0.20])},
        ],
        "labels": [str(x) for x in [0.01, 0.1, 0.2]],
    },
    {
        "simulation": "task 1, extended controlloer",
        "common": {
            "ini_type": "down",
            "ref_type": "swing-up",
            "mod_type_ocp": "extended",
            "Q": np.diag([10, 10, 0.1, 0.1, 0.01]),
        },
        "specific": [
            {"R": 0.01},
            {"R": 0.005},
            {"R": 0.001},
        ],
        "labels": [str(x) for x in [0.01, 0.005, 0.001]],
    },
    # ################################################
    # #################### TASK 2 ####################
    # ################################################
    {
        "simulation": "task 2, simulate default system (ERK)",
        "common": {
            "ini_type": "off-balance",
            "ctrl_on": False,
            "mod_type_sim": "non-linear",
            "ref_type": "empty-2s",
            # this allows to pass the checks on controller sampling
            "Ts": 1,
            "integ_type_sim": "ERK",
        },
        "specific": [
            {"ts_sim": 1e-5},
            {"ts_sim": 1e-4},
            {"ts_sim": 1e-3},
            {"ts_sim": 1e-2},
        ],
        "labels": ["1e-" + str(x) for x in range(5, 1, -1)],
    },
    {
        "simulation": "task 2, simulate default system (IRK)",
        "common": {
            "ini_type": "off-balance",
            "ctrl_on": False,
            "mod_type_sim": "non-linear",
            "ref_type": "empty-2s",
            # this allows to pass the checks on controller sampling
            "Ts": 1,
            "integ_type_sim": "IRK",
        },
        "specific": [
            {"ts_sim": 1e-5},
            {"ts_sim": 1e-4},
            {"ts_sim": 1e-3},
            {"ts_sim": 1e-2},
        ],
        "labels": ["1e-" + str(x) for x in range(5, 1, -1)],
    },
    {
        "simulation": "task 2, simulate default system (ERK vs IRK)",
        "common": {
            "ini_type": "off-balance",
            "ctrl_on": False,
            "mod_type_sim": "non-linear",
            "ref_type": "empty-2s",
            # this allows to pass the checks on controller sampling
            "Ts": 1,
            "ts_sim": 1e-2,
        },
        "specific": [
            {"integ_type_sim": "ERK"},
            {"integ_type_sim": "IRK"},
        ],
        "labels": ["ERK", "IRK"],
    },
    {
        "simulation": "task 2, simulate default system (ERK vs IRK)",
        "common": {
            "ini_type": "off-balance",
            "ctrl_on": False,
            "mod_type_sim": "non-linear",
            "ref_type": "empty-2s",
            # this allows to pass the checks on controller sampling
            "Ts": 1,
            "ts_sim": 1e-1,
        },
        "specific": [
            {"integ_type_sim": "ERK"},
            {"integ_type_sim": "IRK"},
        ],
        "labels": ["ERK", "IRK"],
    },
    {
        "simulation": "task 2, simulate spring system (ERK)",
        "common": {
            "ini_type": "off-balance",
            "ctrl_on": False,
            "mod_type_sim": "spring",
            "ref_type": "empty-2s",
            # this allows to pass the checks on controller sampling
            "Ts": 1,
            "integ_type_sim": "ERK",
        },
        "specific": [
            {"ts_sim": 1e-5},
            {"ts_sim": 1e-4},
            {"ts_sim": 1e-3},
            {"ts_sim": 1e-2},
        ],
        "labels": ["1e-" + str(x) for x in range(5, 1, -1)],
    },
    {
        "simulation": "task 2, simulate spring system (IRK)",
        "common": {
            "ini_type": "off-balance",
            "ctrl_on": False,
            "mod_type_sim": "spring",
            "ref_type": "empty-2s",
            # this allows to pass the checks on controller sampling
            "Ts": 1,
            "integ_type_sim": "IRK",
        },
        "specific": [
            {"ts_sim": 1e-5},
            {"ts_sim": 1e-4},
            {"ts_sim": 1e-3},
            {"ts_sim": 1e-2},
        ],
        "labels": ["1e-" + str(x) for x in range(5, 1, -1)],
    },
    {
        "simulation": "task 2, simulate spring system (ERK vs IRK)",
        "common": {
            "ini_type": "off-balance",
            "ctrl_on": False,
            "mod_type_sim": "spring",
            "ref_type": "empty-2s",
            # this allows to pass the checks on controller sampling
            "Ts": 1,
            "ts_sim": 1e-2,
        },
        "specific": [
            {"integ_type_sim": "ERK"},
            {"integ_type_sim": "IRK"},
        ],
        "labels": ["ERK", "IRK"],
    },
    {
        "simulation": "task 2, simulate spring system (ERK vs IRK)",
        "common": {
            "ini_type": "off-balance",
            "ctrl_on": False,
            "mod_type_sim": "spring",
            "ref_type": "empty-2s",
            # this allows to pass the checks on controller sampling
            "Ts": 1,
            "ts_sim": 1e-1,
        },
        "specific": [
            {"integ_type_sim": "ERK"},
            {"integ_type_sim": "IRK"},
        ],
        "labels": ["ERK", "IRK"],
    },
    {
        "simulation": "task 2, default non-linear controller, default sampling time",
        "common": {"ini_type": "down", "ref_type": "swing-up", "Ts": 0.02, "N": 100},
        "specific": [
            {"integ_type_ocp": "ERK"},
            {"integ_type_ocp": "IRK"},
        ],
        "labels": ["ERK", "IRK"],
    },
    {
        "simulation": "task 2, default non-linear controller, higher sampling time",
        "common": {"ini_type": "down", "ref_type": "swing-up", "Ts": 0.1, "N": 20},
        "specific": [
            {"integ_type_ocp": "ERK"},
            {"integ_type_ocp": "IRK"},
        ],
        "labels": ["ERK", "IRK"],
    },
    # ################################################
    # #################### TASK 3 ####################
    # ################################################
    {
        "simulation": "task 3, NMPC vs linear MPC",
        "common": {"ini_type": "up", "ref_type": "horizontal"},
        "specific": [
            {"mod_type_ocp": "linear"},
            {"mod_type_ocp": "non-linear"},
        ],
        "labels": ["Linear", "Non-Linear"],
    },
    {
        "simulation": "task 3, swing-up maneuver with linear MPC",
        "common": {"ini_type": "down", "ref_type": "swing-up"},
        "specific": [
            {"mod_type_ocp": "linear"},
            {"mod_type_ocp": "non-linear"},
        ],
        "labels": ["Linear", "Non-Linear"],
    },
]
