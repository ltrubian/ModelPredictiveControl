from script.loop_simulations import (
    closed_loop_simulation_extended,
    closed_loop_simulation,
)
import numpy as np


if __name__ == "__main__":
    closed_loop_simulation(
        ini_type="off-balance",
        mod_type_ocp="spring",
        mod_type_sim="spring",
        ctrl_on=False,
        integ_type_sim="IRK",
        save_video=False,
    )
    # closed_loop_simulation_extended(save_video=False)
