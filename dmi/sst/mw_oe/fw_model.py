import numpy as np


class FwModel:
    # fw-model correction coefficients for correction "sst2_ws2_phir"
    COEFFS = np.array([[-822.280328393158, 0.031740043058786, -0.000556942697289299, -0.065163365855556, -0.000946513959891076, 2.68582900015897, -0.0137607698948962, -1.10122774194525,
                        309.993246640644, -1093.35038158185, -1907.59692214927, 1914.32899665864, 1924.12216877802],
                       [-2579.26387785169, 0.0409153078992824, -0.00109624795309051, 0.00785484006214451, -0.00632886309160456, 10.1238387761352, -0.447164199774581, 12.4038506140698,
                        999.411480624009, -3532.90758608412, -6015.91776477483, 6089.89862675066, 6025.28990136284],
                       [666.927107091751, 0.0058327002343355, 0.000188973821342725, -0.0927648223229551, 0.00364975624725157, -3.54651683851061, 0.0259578429885675, -0.321946965601382,
                        -273.105580828332, 924.739891300339, 1598.80315606094, -1587.57358953647, -1585.72459932043],
                       [-1032.43301761521, 0.00291117584071364, 0.000221818475044713, -0.0478880074813039, 0.00407467902062602, 4.09519750910037, -0.344042518336166, 17.4072669570761,
                        401.335811845877, -1468.05393627111, -2381.1470022439, 2478.84955764869, 2373.68911877949],
                       [2088.94523828281, -0.0270323688545631, 0.000562151802292091, -0.0718345098892131, 0.00191699152233451, -9.49846351037415, -0.323690222337714, 19.9454304268041,
                        -830.043665170263, 2772.9467728129, 4992.47069531972, -4871.72276184138, -4996.03554657514],
                       [1686.61952475376, -0.0341956372533193, 0.000502035312640694, -0.0513572034993604, 0.00516241288473255, -6.4065858773806, -0.55399351867331, 30.544102735797, -648.384788734449,
                        2147.88194540344, 4008.44207325015, -3858.22717130295, -4046.74709344951],
                       [1862.43102031746, -0.0112260117934377, 0.0011409534306883, 0.0621751958886603, -0.0039200409097119, -8.02568611593214, 1.08372770375844, -47.3639862399631, -732.323841576413,
                        2722.44764656957, 4282.65489931013, -4530.10406311117, -4250.02781521146],
                       [3970.27015564206, -0.0329386785729684, 0.000796708874891527, 0.160969377313508, -0.00552468647227517, -15.295818975743, 2.47335948008454, -107.100321762734, -1532.82359318115,
                        5784.12145672997, 9042.38896162857, -9632.72430822637, -9001.62970191328],
                       [-943.068085254121, 0.0135800502735469, -0.000172643359583031, -0.0147811055994187, -0.00235248526593811, 3.59968229016487, 0.0353202775143235, -2.9421781987135,
                        366.169223057367, -1263.89223093307, -2218.61706340032, 2206.36454155195, 2226.01237044519],
                       [-1604.15378263188, 0.0145433264174034, -0.000833696906570764, -0.0106371708585439, 0.000608083358611499, 6.84037734125216, -0.699341598809711, 28.7106783718284,
                        629.033997763998, -2293.19464528223, -3711.89956408714, 3862.05120458042, 3694.27481015444]], dtype=np.float64)

    # AMSR-E frequencies
    FREQ = np.array([6.93, 10.65, 18.70, 23.80, 36.50], dtype=np.float64)
    light_speed = 3.00E10  # Speed of light, [cm/s]
    LAMBA = light_speed / (FREQ * 1E9)

    # Model coefficients for the atmosphere
    #                   b0[K] b1[K mm-1] b2[K mm-2] b3[K mm-3] b4[K mm-4] b5  b6[K]  b7[K mm-1] a01     a02[K-1] aV1[mm-1] aV2[mm-2]
    MC_ATM = np.array([[239.5, 2.1392, -0.04606, 0.00045711, -1.684e-006, 0.5, -0.11, -0.0021, 0.00834, -4.8e-005, 7e-005, 0],
                       [239.51, 2.2519, -0.044686, 0.00039182, -1.22e-006, 0.54, -0.12, -0.0034, 0.00908, -4.7e-005, 0.00018, 0],
                       [240.24, 2.9888, -0.072593, 0.0008145, -3.607e-006, 0.61, -0.16, -0.0169, 0.01215, -6.1e-005, 0.00173, -5e-007],
                       [241.69, 3.1032, -0.081429, 0.00099893, -4.837e-006, 0.2, -0.2, -0.0521, 0.01575, -8.7e-005, 0.00514, 1.9e-006],
                       [239.45, 2.5441, -0.051284, 0.00045202, -1.436e-006, 0.58, -0.57, -0.0238, 0.04006, -0.0002, 0.00188, 9e-007]], dtype=np.float64)

    # Coefficients for Rayleigh absorption and Mie scattering
    #                   aL1     aL2     aL3     aL4  aL5
    MC_ABS = np.array([[0.0078, 0.0303, 0.0007, 0, 1.2216], [0.0183, 0.0298, 0.0027, 0.006, 1.1795], [0.0556, 0.0288, 0.0113, 0.004, 1.0636], [0.0891, 0.0281, 0.0188, 0.002, 1.022],
                       [0.2027, 0.0261, 0.0425, -0.002, 0.9546]], dtype=np.float64)

    # Model coefficients for geometric optics
    #                   v-p r0   h-p r0     v-p r1   h-p r1     v-p tab--  h-p tab-- v-p r3 h-p r3
    MC_GEO = np.array([[-0.00027, 0.00054, -2.1e-005, 3.2e-005, -2.1e-005, -2.526e-005, 0, 0], [-0.00032, 0.00072, -2.9e-005, 4.4e-005, -2.1e-005, -2.894e-005, 8e-008, -2e-008],
                       [-0.00049, 0.00113, -5.3e-005, 7e-005, -2.1e-005, -3.69e-005, 3.1e-007, -1.2e-007], [-0.00063, 0.00139, -7e-005, 8.5e-005, -2.1e-005, -4.195e-005, 4.1e-007, -2e-007],
                       [-0.00101, 0.00191, -0.000105, 0.000112, -2.1e-005, -5.451e-005, 4.5e-007, -3.6e-007]], dtype=np.float64)

    # Model coefficients m
    #                 v-p m1  h-p m1 v-p m2  h-p m2
    MC_M = np.array([[0.0002, 0.002, 0.0069, 0.006], [0.0002, 0.002, 0.0069, 0.006], [0.0014, 0.00293, 0.00736, 0.00656], [0.00178, 0.00308, 0.0073, 0.0066], [0.00257, 0.00329, 0.00701, 0.0066]],
                    dtype=np.float64)

    DEG_TO_RAD = np.pi / np.float64(180.0)

    # background radiation [K]
    T_C = np.float64(2.7)

    EMISSIVITY_FY_V = np.array([0.9204, 0.9127, 0.9373, 0.9409, 0.9347], dtype=np.float64)
    EMISSIVITY_FY_H = np.array([0.7502, 0.7738, 0.8314, 0.8490, 0.8600], dtype=np.float64)
    EMISSIVITY_MY_V = np.array([0.9692, 0.9284, 0.8843, 0.8554, 0.7813], dtype=np.float64)
    EMISSIVITY_MY_H = np.array([0.8651, 0.8356, 0.7917, 0.7792, 0.7248], dtype=np.float64)

    def run(self, W, V, L, T_ow, T_is, C_is, F_MY, theta_d, sss, phi_rd):
        # convert angles to radians
        theta_r = theta_d * self.DEG_TO_RAD
        phi_rr = phi_rd * self.DEG_TO_RAD

        # ---------------------
        # Information about ice
        # ---------------------
        F_MY = self.clamp_to_0_1(F_MY)
        C_is = self.clamp_to_0_1(C_is)

        C_MY = C_is * F_MY  # multi year ice concentration
        C_FY = C_is - C_MY  # first year ice concentration
        C_ow = 1.0 - C_is  # open water concentration

        T_is = self.calc_ice_temp(T_ow)
        T_ow = self.calc_open_water_temp(C_is, T_ow)

        T_BV_FY = T_is * self.EMISSIVITY_FY_V  # brightness temperature from FY is (vertical pol)
        T_BH_FY = T_is * self.EMISSIVITY_FY_H  # brightness temperature from FY is (horizontal pol)
        T_BV_MY = T_is * self.EMISSIVITY_MY_V  # brightness temperature from MY is (vertical pol)
        T_BH_MY = T_is * self.EMISSIVITY_MY_H  # brightness temperature from MY is (horizontal pol)

        T_S_mix = C_is * T_is + C_ow * T_ow  # temperature of mixed surfaces (ice and open water)
        T_L = (T_S_mix + 273.0) * 0.5  # temperature of the water droplets[K]; approximation: mean temp of surface and freezing level

        # ------------------------
        # Model for the Atmosphere
        # ------------------------
        T_V = self.calc_T_V(V)
        sig_TS_TV = self.calc_sig_TS_TV(T_S_mix, T_V)

        # equation (26a)
        V_SQ = V * V
        T_D = self.MC_ATM[:, 0] + self.MC_ATM[:, 1] * V + self.MC_ATM[:, 2] * V_SQ + self.MC_ATM[:, 3] * V_SQ * V + self.MC_ATM[:, 4] * V_SQ * V_SQ + self.MC_ATM[:, 5] * sig_TS_TV

        # equation (26b)
        T_U = T_D + self.MC_ATM[:, 6] + self.MC_ATM[:, 7] * V

        # Vertically integrated oxygen absorption - equation (28)
        A_0 = self.MC_ATM[:, 8] + self.MC_ATM[:, 9] * (T_D - 270.0)

        # Vertically integrated water vapor absorbtion - equation (29)
        A_V = self.MC_ATM[:, 10] * V + self.MC_ATM[:, 11] * V_SQ

        # Vertically integrated liquid water absorption - equation (33) (no rain)
        A_L = self.MC_ABS[:, 0] * (1.0 - self.MC_ABS[:, 1] * (T_L - 283.0)) * L

        # Total transmittance from the surface to the top of atmosphere - equation (22)
        cos_theta_r = np.cos(theta_r)
        tau = np.exp((-1.0 / cos_theta_r) * (A_0 + A_V + A_L))

        T_BU = T_U * (1.0 - tau)  # The upwelling effective air temperature; equation (24a)
        T_BD = T_D * (1.0 - tau)  # The downwelling effective air temperature; equation (24b)

        # --------------------------------
        # Dielectric Constant of Sea-water
        # --------------------------------
        t_ow = T_ow - 273.15  # Surface temperature T_ow [deg Celcius]
        epsilon_R = 4.44  # Dielectric constant at inf. freq.
        ny = 0.012  # Spread factor

        # equation (36) and (43)
        epsilon_S = 87.9 * np.exp(-0.004585 * t_ow) * (np.exp(-3.45E-3 * sss + 4.69E-6 * sss * sss + 1.36E-5 * sss * t_ow))

        # equation (38) and (44)
        t_ow_sq = t_ow * t_ow
        lambda_R = (3.3 * np.exp(-0.0346 * t_ow + 0.00017 * t_ow_sq)) - (6.54E-3 * (1 - 3.06E-2 * t_ow + 2.0E-4 * t_ow_sq) * sss)

        C = 0.5536 * sss  # equation (41)
        delta_t = 25.0 - t_ow  # equation (42)

        # equation (40)
        delta_t_sq = delta_t * delta_t
        qsi = 2.03E-2 + 1.27E-4 * delta_t + 2.46E-6 * delta_t_sq - C * (3.34E-5 - 4.60E-7 * delta_t + 4.60E-8 * delta_t_sq)

        sigma = 3.39E9 * np.power(C, 0.892) * np.exp(-delta_t * qsi)  # equation (39)

        # equation (35)
        lamb_r_ratio = np.complex128(1j * lambda_R / self.LAMBA)
        sig_lamb = np.complex128(1j * sigma * self.LAMBA)
        epsilon = epsilon_R + (epsilon_S - epsilon_R) / (1 + np.power(lamb_r_ratio, 1.0 - ny)) - 2 * sig_lamb / self.light_speed

        # equation (45b); horizontal pol.
        sin_theta_r = np.sin(theta_r)
        sqrt_eps_thet = np.sqrt(epsilon - sin_theta_r * sin_theta_r)
        rho_H = (cos_theta_r - sqrt_eps_thet) / (cos_theta_r + sqrt_eps_thet)
        # equation (45a); vertical pol.
        rho_V = (epsilon * cos_theta_r - sqrt_eps_thet) / (epsilon * cos_theta_r + sqrt_eps_thet)

        # Power reflectivity
        R_0H = np.abs(rho_H) * np.abs(rho_H)  # equation (46); horizontal pol.
        R_0V = np.abs(rho_V) * np.abs(rho_V) + (4.887E-8 - 6.108E-8 * np.power(T_ow - 273.0, 3.0))  # equation (46)+correction vertical pol.

        # ------------------------------
        # The wind-roughened Sea Surface
        # ------------------------------
        R_geoH = R_0H - (self.MC_GEO[:, 1] + self.MC_GEO[:, 3] * (theta_d - 53.0) + self.MC_GEO[:, 5] * (T_ow - 288.0) + self.MC_GEO[:, 7] * (theta_d - 53.0) * (T_ow - 288.0)) * W
        R_geoV = R_0V - (self.MC_GEO[:, 0] + self.MC_GEO[:, 2] * (theta_d - 53.0) + self.MC_GEO[:, 4] * (T_ow - 288.0) + self.MC_GEO[:, 6] * (theta_d - 53.0) * (T_ow - 288.0)) * W
        # print(R_geoV)

    def clamp_to_0_1(self, param):
        if param < 0.0:
            return np.float64(0.0)

        if param > 1.0:
            return np.float64(1.0)

        return param

    def calc_ice_temp(self, T_ow):
        if T_ow > 273.15:
            return np.float64(273.15)
        else:
            return 0.4 * T_ow + 163.2  # 0.6 * 272 = 163.2

    def calc_open_water_temp(self, C_is, T_ow):
        if C_is > 0.05:
            return np.float64(273.15)
        else:
            return T_ow

    def calc_T_V(self, V):
        if V <= 48.0:
            # equation (27a)
            return 273.16 + 0.8337 * V - 3.029e-5 * np.power(V, 3.33)
        else:
            # equation (27b)
            return np.float64(301.16)

    def calc_sig_TS_TV(self, T_S_mix, T_V):
        delta = T_S_mix - T_V
        abs_delta = np.abs(delta)
        if abs_delta <= 20.0:
            # equation (27c)
            return 1.05 * delta * (1.0 - (delta * delta) / 1200)
        else:
            #  equation (27d)
            return np.sign(delta) * 14.0
