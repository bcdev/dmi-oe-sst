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
    maxit = None

    eps = np.array([0.2, 0.1, 0.02, 0.25], dtype=np.float64)

    def __init__(self):
        self.S_p_inv = np.linalg.inv(self.S_p)
        self.S_e_inv = np.linalg.inv(self.S_e)

        self.fw_model = FwModel()

        self.maxit = 10

    def run(self, input, results, flag_coding):
        sw = 0  # Switched off ice in forward model
        num_matchups = len(input.coords["matchup_count"])

        ws = input["amsre.nwp.abs_wind_speed"].data
        tcwv = input["amsre.nwp.total_column_water_vapour"].data
        tclw = input["amsre.nwp.total_column_liquid_water"].data
        sst = input["amsre.nwp.sea_surface_temperature"].data
        sza = input["amsre.satellite_zenith_angle"].data
        phi_rd = input["relative_angle"].data

        j_ite_0 = np.full([num_matchups], np.NaN, np.float64)
        di2 = np.full([num_matchups, self.maxit], np.NaN, np.float64)
        sss = np.float64(35.0)

        flags = flag_coding.get_flags()
        for matchup_index in range(0, num_matchups):
            # check if matchup is already flagged, if so: next one
            if flags[matchup_index] != 0:
                continue

            [p, p_0] = self.prepare_first_guess(ws[matchup_index], tcwv[matchup_index], tclw[matchup_index], sst[matchup_index], self.eps)

            theta_d = np.float64(sza[matchup_index])

            T_A = self.create_T_A(input)

            # ------------------------------------------------------
            # Calculate brightness temps on basis of the first guess,
            # our starting point for the iteration by the forward function
            # ------------------------------------------------------
            T_A0 = self.fw_model.run(p[0, 2], p[1, 2], p[2, 2], p[3, 2], sw, sw, sw, theta_d, sss, phi_rd[matchup_index])

            # ----------------------------------------------
            # Obs - calc, also needed to start the iteration
            # ----------------------------------------------
            Delta_T = T_A[matchup_index, :] - T_A0

            # ------------------------------
            # Results from inversion with FG
            # ------------------------------
            tb_rmse_ite0 = np.sqrt(np.mean(Delta_T * Delta_T))
            dtb_ite0 = -Delta_T
            T_A0_ite0 = T_A0
            temp = np.dot(self.S_e_inv, Delta_T)
            j_ite_0[matchup_index] = np.dot(Delta_T, temp)

            len_T_A = len(T_A0)
            ite_Std_inv = np.full([self.maxit, 4], np.NaN, np.float64)
            ite_p = np.full([self.maxit, 4], np.NaN, np.float64)
            ite_TA0 = np.full([self.maxit, len_T_A], np.NaN, np.float64)
            ite_Delta_T = np.full([self.maxit, len_T_A], np.NaN, np.float64)
            test = np.full([self.maxit], np.NaN, np.float64)
            dsi = np.full([self.maxit], np.NaN, np.float64)
            dni = np.full([self.maxit], np.NaN, np.float64)
            chi = np.full([self.maxit], np.NaN, np.float64)
            J = np.full([self.maxit], np.NaN, np.float64)
            K = np.full([len_T_A, 4], np.NaN, np.float64)
            AKi = np.full([self.maxit, 4, 4], np.NaN, np.float64)

            # -------------------------------------------------
            # Start iteration and calculation of new p estimate
            # -------------------------------------------------
            convergence_passed_flag = 0
            convergence_passed_idx = self.maxit
            for ite in range(0, self.maxit):
                # for ite in range(0, 1):
                # -------------------
                # Calculate Jacobians
                # -------------------
                K[:, 0] = (T_A0 - self.fw_model.run(p[0, 1], p[1, 2], p[2, 2], p[3, 2], sw, sw, sw, theta_d, sss, phi_rd[matchup_index])) / (p[0, 2] - p[0, 1])
                K[:, 1] = (T_A0 - self.fw_model.run(p[0, 2], p[1, 1], p[2, 2], p[3, 2], sw, sw, sw, theta_d, sss, phi_rd[matchup_index])) / (p[1, 2] - p[1, 1])
                K[:, 2] = (T_A0 - self.fw_model.run(p[0, 2], p[1, 2], p[2, 1], p[3, 2], sw, sw, sw, theta_d, sss, phi_rd[matchup_index])) / (p[2, 2] - p[2, 1])
                K[:, 3] = (T_A0 - self.fw_model.run(p[0, 2], p[1, 2], p[2, 2], p[3, 1], sw, sw, sw, theta_d, sss, phi_rd[matchup_index])) / (p[3, 2] - p[3, 1])

                # ---------------------
                # Calculate delta p
                # Use p_0 as init guess
                # ---------------------
                Delta_p = p_0 - p[:, 2]

                # ---------------------------
                # Retrieval Covariance matrix
                # ---------------------------
                temp = np.matmul(self.S_e_inv, K)
                K_Se_inv_K = np.matmul(K.transpose(), temp)
                S = self.S_p_inv + K_Se_inv_K
                S_inv = np.linalg.inv(S)

                # --------------------------------------------------------------
                # Averaging Kernel
                # The averaging kernel matrix,AK, relates the sensitivity of the
                # retrieval to the true state.
                # --------------------------------------------------------------
                AK = np.matmul(S_inv, K_Se_inv_K)

                # -----------------------------
                # Degrees of freedom for signal
                # -----------------------------
                ds = np.trace(AK)

                # ----------------------------
                # Degrees of freedom for noise
                # ----------------------------
                dn = np.trace(np.matmul(S_inv, self.S_p_inv))

                # ------------------------------------
                # Calculate update of retrieval vector
                # per iteration
                # ------------------------------------
                temp = np.matmul(self.S_e_inv, Delta_T)
                temp = np.matmul(K.transpose(), temp)
                temp = np.matmul(self.S_p_inv, Delta_p) + temp
                p_est = np.matmul(S_inv, temp)

                # ------------------------------------
                # This is the updated retrieval vector
                # ------------------------------------
                p_new = p_est + p[:, 2]

                # ----------------------------
                # We save the retrieval vector
                # of the last iteration
                # ----------------------------
                # we skip this because it is never used anywhere tb 2018-01-04

                # --------------------------------
                # And now the new retrieval vector
                # gets the old name
                # --------------------------------
                for k in range(0, 4):
                    p[k, :] = p[k, :] + p_est[k]

                # ---------------------------------
                # Certainly we now must also update
                # the deviations
                # ---------------------------------
                p[:, 0] = p_new - self.eps
                p[:, 1] = p_new + self.eps

                # ------------------------------------------------------
                # We need new brightness temps that match the updated
                # atmospheric parameters in the updated retrieval vector
                # They are calculated by using the forward model
                # ------------------------------------------------------
                T_A0 = self.fw_model.run(p[0, 2], p[1, 2], p[2, 2], p[3, 2], sw, sw, sw, theta_d, sss, phi_rd[matchup_index])

                # --------------------------------------------
                # How much do the updated simulated brightness
                # temps deviate from our measurements?
                # A L1b data test known as obs-calc
                # --------------------------------------------
                Delta_T = T_A[matchup_index] - T_A0

                # ---------------------------------------------
                # Analysis of the quality
                # of the retrieval result in order to determine
                # if we need more iterations
                # ---------------------------------------------

                # Calculate std for members of retrieval vector
                # This characterizes the quality of our retrieval results
                for k in range(0, 4):
                    ite_Std_inv[ite, k] = np.sqrt(S_inv[k, k])

                # Our retrieval result for each iterations
                ite_p[ite, :] = p[:, 2]  # collect our results per iteration

                # Simulated brightness temperatures for each iteration
                ite_TA0[ite, :] = T_A0  # collect simulations per iteration

                # OBS-CALC brightness temperatures
                ite_Delta_T[ite, :] = Delta_T  # Collect obs-calc

                # TB RMSE test
                test[ite] = np.sqrt(np.mean(Delta_T * Delta_T))  # Total "Disagreement" by Tb RMSE all channels.

                # CHI-SQUARE calc
                temp = np.matmul(self.S_e, Delta_T)
                chi[ite] = np.matmul(Delta_T, temp)  # "Disagreement" measured by a chi-square test -reduced

                # The total degree of freedom sums up to 4.
                dsi[ite] = ds  # degrees of freedom for signal
                dni[ite] = dn  # degrees of freedom for noise

                # Sensitivities
                AKi[ite, :, :] = AK

                # Calculating the cost function for each iteration
                delta_p = p_new - p_0
                temp = np.matmul(self.S_p_inv, delta_p)
                cost = np.matmul(delta_p, temp)
                temp = np.matmul(self.S_e_inv, Delta_T)
                J[ite] = cost + np.matmul(Delta_T, temp)

                # Convergence - cost function being minimized
                if ite == 0:
                    di2[matchup_index, ite] = j_ite_0[matchup_index] - J[ite]
                else:
                    di2[matchup_index, ite] = J[ite - 1] - J[ite]

                # convergence criterion
                if (di2[matchup_index, ite] < 0.1) & (di2[matchup_index, ite] > 0.0):
                    convergence_passed_flag = 1
                    # need to add one as the Matlab code counts to basis one, whereas we
                    # are in Python using zero based counting tb 2018-01-08
                    convergence_passed_idx = ite + 1

                    break

            # collect results into structure
            results.j.data[matchup_index, :] = J
            results.tb_rmse_ite.data[matchup_index, :] = test
            results.tb_rmse_ite0.data[matchup_index] = tb_rmse_ite0
            results.tb_chi_ite.data[matchup_index, :] = chi
            results.convergence_passed_flag.data[matchup_index] = convergence_passed_flag
            results.convergence_passed_idx.data[matchup_index] = convergence_passed_idx
            results.di2.data[matchup_index, :] = di2[matchup_index, :]
            results.dtb_ite0.data[matchup_index, :] = dtb_ite0
            results.TA0_ite0.data[matchup_index, :] = T_A0_ite0
            results.j_ite0[matchup_index] = j_ite_0[matchup_index]
            
            last_iteration = convergence_passed_idx - 1
            results.AK[matchup_index,:] = np.diagonal(AKi[last_iteration, :, :])
            results.chisq[matchup_index] = chi[last_iteration]
            results.tb_rmse[matchup_index] = test[last_iteration]
            results.p[matchup_index,:] = ite_p[last_iteration, :]
            results.S[matchup_index,:] = ite_Std_inv[last_iteration, :]
            results.tb_sim[matchup_index,:] = ite_TA0[last_iteration, :]
            results.dtb[matchup_index,:] = -ite_Delta_T[last_iteration, :]
            results.ds[matchup_index] = dsi[last_iteration]
            results.dn[matchup_index] = dni[last_iteration]
            results.K4[matchup_index,:] = K[:,3]
            results.ite_index[matchup_index] = convergence_passed_idx

    def prepare_first_guess(self, ws, tcwv, tclw, sst, eps):
        sst = sst + 273.15  # covert sst back to K
        p = np.array([[ws - eps[0], ws + eps[0], ws], [tcwv - eps[1], tcwv + eps[1], tcwv], [tclw - eps[2], tclw + eps[2], tclw], [sst - eps[3], sst + eps[3], sst]], dtype=np.float64)

        p_0 = np.copy(p[:, 2])

        return [p, p_0]

    def create_T_A(self, dataset):
        num_matchups = len(dataset.coords["matchup_count"])
        T_A = np.empty((num_matchups, 10))
        for i in range(6, 16):
            variable_name = INPUT_VARIABLES[i]
            T_A[:, i - 6] = dataset[variable_name].data

        return T_A
