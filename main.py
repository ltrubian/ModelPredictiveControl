from script.loop_simulations import (
    closed_loop_simulation_extended,
    closed_loop_simulation,
)
import numpy as np


if __name__ == "__main__":
    closed_loop_simulation(ini_type="off-balance")
    # closed_loop_simulation_extended(save_video=False)
