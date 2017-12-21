import numpy as np

from dmi.sst.mw_oe.constants import INPUT_VARIABLES
from dmi.sst.mw_oe.fw_model import FwModel


class Retrieval:
    S_p = np.array([[16.0, 0, 0, 0], [0, 0.81, 0, 0], [0, 0, 0.25, 0], [0, 0, 0, 1.21]], dtype=np.float64)
    S_e = np.array([[0.4823, 0.3687, 0.2836, 0.0156, 0.1638, -0.2592, 0.2384, -0.2011, 0.288, -0.137], [0.3687, 0.5226, 0.1111, -0.0448, -0.02, -0.3407, 0.1343, -0.186, 0.2587, -0.0062],
                    [0.2836, 0.1111, 0.3138, 0.0954, 0.2486, -0.0668, 0.1867, -0.1835, 0.202, -0.1610], [0.0156, -0.0448, 0.0954, 0.1305, 0.0505, 0.0452, -0.0156, -0.1091, 0.0216, -0.0086],
                    [0.1638, -0.02, 0.2486, 0.0505, 0.3760, 0.1330, 0.2045, -0.0717, 0.1701, -0.2019], [-0.2592, -0.3407, -0.0668, 0.0452, 0.133, 0.4228, -0.093, 0.122, -0.1498, 0.0071],
                    [0.2384, 0.1343, 0.1867, -0.0156, 0.2045, -0.093, 0.3748, 0.072, 0.2224, -0.2381], [-0.2011, -0.186, -0.1835, -0.1091, -0.0717, 0.122, 0.072, 0.5355, -0.1265, -0.0137],
                    [0.288, 0.2587, 0.202, 0.0216, 0.1701, -0.1498, 0.2224, -0.1265, 0.2939, -0.1433], [-0.137, -0.0062, -0.161, -0.0086, -0.2019, 0.0071, -0.2381, -0.0137, -0.1433, 0.2279]],
                   dtype=np.float64)
    S_p_inv = None
    S_e_inv = None
    fw_model = None

    eps = np.array([0.2, 0.1, 0.02, 0.25], dtype=np.float64)

    def __init__(self):
        self.S_p_inv = np.linalg.inv(self.S_p)
        self.S_e_inv = np.linalg.inv(self.S_e)

        self.fw_model = FwModel()

    def run(self, input, results, flag_coding):
        sw = 0  # Switched off ice in forward model
        num_matchups = len(input.coords["matchup_count"])

        ws = input["amsre.nwp.abs_wind_speed"].data
        tcwv = input["amsre.nwp.total_column_water_vapour"].data
        tclw = input["amsre.nwp.total_column_liquid_water"].data
        sst = input["amsre.nwp.sea_surface_temperature"].data
        sza = input["amsre.solar_zenith_angle"].data
        phi_rd = input["relative_angle"].data

        flags = flag_coding.get_flags()
        for i in range(0, num_matchups):
            # check if matchup is already flagged, if so: next one
            if flags[i] != 0:
                continue

            [p, p_0] = self.prepare_first_guess(ws[i], tcwv[i], tclw[i], sst[i], self.eps)

            theta_d = np.float64(sza[i])
            sss = np.float64(35.0)

            T_A = self.create_T_A(input)

            # Calculate brightness temps on basis of the first guess,
            # our starting point for the iteration by the forward function
            T_AO = self.fw_model.run(p[0, 2], p[1, 2], p[2, 2], p[3, 2], sw, sw, sw, theta_d, sss, phi_rd[i])

            Delta_T = T_A[i, :] - T_AO
            #print(Delta_T)

    def prepare_first_guess(self, ws, tcwv, tclw, sst, eps):
        sst = sst + 273.15  # covert sst back to K
        p = np.array(
            [[ws - self.eps[0], ws + self.eps[0], ws], [tcwv - self.eps[1], tcwv + self.eps[1], tcwv], [tclw - self.eps[2], tclw + self.eps[2], tclw], [sst - self.eps[3], sst + self.eps[3], sst]],
            dtype=np.float64)

        p_0 = p[:, 2]

        return [p, p_0]

    def create_T_A(self, dataset):
        num_matchups = len(dataset.coords["matchup_count"])
        T_A = np.empty((num_matchups, 10))
        for i in range(6, 16):
            variable_name = INPUT_VARIABLES[i]
            T_A[:, i - 6] = dataset[variable_name].data

        return T_A
