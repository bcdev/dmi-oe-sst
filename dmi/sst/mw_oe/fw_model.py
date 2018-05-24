import numpy as np

from numba import jit, prange

# AMSR-E frequencies
FREQ = np.array([6.93, 10.65, 18.70, 23.80, 36.50], dtype=np.float64)

# Model coefficients m
#                 v-p m1  h-p m1 v-p m2  h-p m2
MC_M = np.array([[0.0002, 0.002, 0.0069, 0.006], [0.0002, 0.002, 0.0069, 0.006], [0.0014, 0.00293, 0.00736, 0.00656], [0.00178, 0.00308, 0.0073, 0.0066], [0.00257, 0.00329, 0.00701, 0.0066]],
                dtype=np.float64)

# fw-model correction coefficients for correction "sst2_ws2_phir"
COEFFS = np.array([[-652.312755833177, -0.0439382312736827, 0.000955338590847359, 0.160881535378189, -0.00341064373984684, 4.25243735882027, -0.295750470255661, 17.5707476258828, 284.376092825079, -1002.50096583959, -1567.52449439398, 1632.49362268677, 1521.39129861804],
                   [1808.22856612219, -0.0461017640506334, 0.00155759892294696, 0.00582409590387642, 0.00583727453048694, -6.84770564656265, -0.0668962941363641, 11.4453200751902, -694.340641333019, 2385.41844753309, 4247.82131614452, -4198.48200949011, -4276.20474027739],
                   [-1766.44142119179, -0.00526241530092911, -0.000413032586226715, 0.167192546899266, -0.00742796960622246, 8.9616277092495, -0.0261487833940575, 2.63713019051241, 720.93524838398, -2455.0633840293, -4224.80951413572, 4209.49383047924, 4190.97298434722],
                   [850.3095249472, 0.00838699262864213, -0.000686342963743578, 0.0387013308157823, -0.00449806805400808, -3.29521556304805, 0.0180478988147953, -3.82124272961968, -328.012017904043, 1164.3644828426, 1977.67180995907, -2007.38126559958, -1981.16590788889],
                   [-2469.67943078734, 0.0392230734456645, -0.00125022675243898, 0.122964594592567, -0.00467205603204534, 11.7002740397628, 0.913458636791412, -44.3505706179089, 990.987786232177, -3212.29352581122, -5974.13390647916, 5713.78907946348, 5980.42723782484],
                   [-325.628346806014, 0.0438057739590837, -0.00106462981314169, 0.0310061667341594, -0.00524168570140392, 0.681710439619565, 0.785445247477454, -40.5071525269447, 114.49063872686, -258.278593912249, -826.39402032647, 623.374988013617, 870.636795691492],
                   [-1447.69330157476, 0.00742598889335783, -0.0010776116056049, -0.00637891508008777, 0.00176768092810433, 6.52416947426867, -0.00433518051839378, 2.13779472493762, 573.379027591323, -1986.4509206929, -3417.68322669421, 3425.83687817438, 3410.2282659711],
                   [-300.391397617458, 0.0127477760776387, -0.000296602875404561, -0.170763073685212, 0.00600520371917429, -0.280350341895389, -1.62812601869475, 74.5414815573037, 88.9253805194587, -662.356816826569, -461.752331059594, 889.262472101572, 444.637625007643],
                   [585.046994882032, -0.00146808645400189, -0.000222406344724742, 0.0720324970663043, 4.74036652140285e-005, -1.40067522965137, 0.540013679858314, -22.1270169164132, -212.035147723716, 855.743249646453, 1282.62963460958, -1417.54097581635, -1288.94536422749],
                   [1464.89915764678, -0.0116209390466245, 0.000649207323554258, -0.0263594555283831, 0.00158583192543431, -6.18261390174966, 0.14400035327241, -5.01323391238247, -572.513856195598, 2007.78120499507, 3432.35475696885, -3461.57319316914, -3432.43525700592]], dtype=np.float64)

DEG_TO_RAD = np.pi / np.float64(180.0)

# background radiation [K]
T_C = np.float64(2.7)

EMISSIVITY_FY_V = np.array([0.9204, 0.9127, 0.9373, 0.9409, 0.9347], dtype=np.float64)
EMISSIVITY_FY_H = np.array([0.7502, 0.7738, 0.8314, 0.8490, 0.8600], dtype=np.float64)
EMISSIVITY_MY_V = np.array([0.9692, 0.9284, 0.8843, 0.8554, 0.7813], dtype=np.float64)
EMISSIVITY_MY_H = np.array([0.8651, 0.8356, 0.7917, 0.7792, 0.7248], dtype=np.float64)

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


class FwModel:
    light_speed = 3.00E10  # Speed of light, [cm/s]
    LAMBA = light_speed / (FREQ * 1E9)

    def run(self, W, V, L, T_ow, C_is, F_MY, theta_d, sss, phi_rd):
        # convert angles to radians
        theta_r = theta_d * DEG_TO_RAD
        phi_rr = phi_rd * DEG_TO_RAD

        # ---------------------
        # Information about ice
        # ---------------------
        F_MY = clamp_to_0_1(F_MY)
        C_is = clamp_to_0_1(C_is)

        C_MY = C_is * F_MY  # multi year ice concentration
        C_FY = C_is - C_MY  # first year ice concentration
        C_ow = 1.0 - C_is  # open water concentration

        T_is = calc_ice_temp(T_ow)
        T_ow = calc_open_water_temp(C_is, T_ow)

        T_S_mix = C_is * T_is + C_ow * T_ow  # temperature of mixed surfaces (ice and open water)
        T_L = (T_S_mix + 273.0) * 0.5  # temperature of the water droplets[K]; approximation: mean temp of surface and freezing level

        # ------------------------
        # Model for the Atmosphere
        # ------------------------
        T_V = calc_T_V(V)
        sig_TS_TV = calc_sig_TS_TV(T_S_mix, T_V)

        # equation (26a)
        V_SQ = V * V
        T_D = MC_ATM[:, 0] + MC_ATM[:, 1] * V + MC_ATM[:, 2] * V_SQ + MC_ATM[:, 3] * V_SQ * V + MC_ATM[:, 4] * V_SQ * V_SQ + MC_ATM[:, 5] * sig_TS_TV

        # equation (26b)
        T_U = T_D + MC_ATM[:, 6] + MC_ATM[:, 7] * V

        # Vertically integrated oxygen absorption - equation (28)
        A_0 = MC_ATM[:, 8] + MC_ATM[:, 9] * (T_D - 270.0)

        # Vertically integrated water vapor absorbtion - equation (29)
        A_V = MC_ATM[:, 10] * V + MC_ATM[:, 11] * V_SQ

        # Vertically integrated liquid water absorption - equation (33) (no rain)
        A_L = MC_ABS[:, 0] * (1.0 - MC_ABS[:, 1] * (T_L - 283.0)) * L

        # Total transmittance from the surface to the top of atmosphere - equation (22)
        cos_theta_r = np.cos(theta_r)
        tau = np.exp((-1.0 / cos_theta_r) * (A_0 + A_V + A_L))

        T_BU = T_U * (1.0 - tau)  # The upwelling effective air temperature; equation (24a)
        # T_BD = T_D * (1.0 - tau)  # The downwelling effective air temperature; equation (24b)

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

        sin_theta_r = np.sin(theta_r)
        sqrt_eps_thet = np.sqrt(epsilon - sin_theta_r * sin_theta_r)

        # --------------------------------------------------
        # Atmospheric Radiation Scattered by the Sea Surface
        # --------------------------------------------------
        Delta_S2 = create_Delta_S2(W)
        term_62 = Delta_S2 - 70 * np.power(Delta_S2, 3)  # Term for equation (62a+b)

        # ---------------------------------------------------------------------------------------------
        # ----- HORIZONTAL ----------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------
        T_BH = calc_horizontal_polarised_BT(C_FY, C_MY, C_ow, T_BU, T_D, T_is, T_ow, W, cos_theta_r, sqrt_eps_thet, tau, term_62, theta_d)

        # ---------------------------------------------------------------------------------------------
        # ----- VERTICAL ------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------
        T_BV = calc_vertical_polarised_BT(C_FY, C_MY, C_ow, T_BU, T_D, T_is, T_ow, W, cos_theta_r, epsilon, sqrt_eps_thet, tau, term_62, theta_d)

        # ------------------------------
        # sort output (exclude 53+89GHz)
        # ------------------------------    
        T_B = np.empty([10], dtype=np.float64)
        T_B[0] = T_BV[0]
        T_B[1] = T_BH[0]
        T_B[2] = T_BV[1]
        T_B[3] = T_BH[1]
        T_B[4] = T_BV[2]
        T_B[5] = T_BH[2]
        T_B[6] = T_BV[3]
        T_B[7] = T_BH[3]
        T_B[8] = T_BV[4]
        T_B[9] = T_BH[4]

        # -----------------------------
        # Forward model bias correction
        # -----------------------------
        T_B = bias_correction(T_B, W, phi_rr, t_ow, t_ow_sq)

        return T_B


@jit('float64[:](float64[:], float64, float64, float64, float64)', nopython=True, parallel=True)
def bias_correction(T_B, W, phi_rr, t_ow, t_ow_sq):
    for i in prange(0, 10):
        T_B[i] = T_B[i] - COEFFS[i, 0] - COEFFS[i, 1] * t_ow - COEFFS[i, 2] * t_ow_sq - COEFFS[i, 3] * W - COEFFS[i, 4] * W * W - COEFFS[i, 5] * np.cos(phi_rr) - COEFFS[i, 6] * np.sin(phi_rr) - \
                 COEFFS[i, 7] * np.cos(phi_rr * 0.5) - COEFFS[i, 8] * np.sin(phi_rr * 0.5) - COEFFS[i, 9] * np.cos(phi_rr / 3.0) - COEFFS[i, 10] * np.sin(phi_rr / 3.0) - COEFFS[i, 11] * np.cos(
            phi_rr * 0.25) - COEFFS[i, 12] * np.sin(phi_rr * 0.25)

    return T_B


@jit('float64(float64)', nopython=True)
def clamp_to_0_1(param):
    if param < 0.0:
        return np.float64(0.0)

    if param > 1.0:
        return np.float64(1.0)

    return param


@jit('float64(float64)', nopython=True)
def calc_ice_temp(T_ow):
    if T_ow > 273.15:
        return np.float64(273.15)
    else:
        return 0.4 * T_ow + 163.2  # 0.6 * 272 = 163.2


@jit('float64(float64, float64)', nopython=True)
def calc_open_water_temp(C_is, T_ow):
    if C_is > 0.05:
        return np.float64(273.15)
    else:
        return T_ow


@jit('float64(float64)', nopython=True)
def calc_T_V(V):
    if V <= 48.0:
        # equation (27a)
        return 273.16 + 0.8337 * V - 3.029e-5 * np.power(V, 3.33)
    else:
        # equation (27b)
        return np.float64(301.16)


@jit('float64(float64, float64)', nopython=True)
def calc_sig_TS_TV(T_S_mix, T_V):
    delta = T_S_mix - T_V
    abs_delta = np.abs(delta)
    if abs_delta <= 20.0:
        # equation (27c)
        return 1.05 * delta * (1.0 - (delta * delta) / 1200)
    else:
        #  equation (27d)
        return np.sign(delta) * 14.0


@jit('float64[:](float64)', nopython=True)
def calc_F_horizontal(W):
    W_1 = 7.0
    W_2 = 12.0

    if W < W_1:
        return MC_M[:, 1] * W  # equation (60a)
    elif (W >= W_1) & (W <= W_2):
        return MC_M[:, 1] * W + 0.5 * (MC_M[:, 3] - MC_M[:, 1]) * (W - W_1) * (W - W_1) / (W_2 - W_1)  # equation (60b)
    else:
        return MC_M[:, 3] * W - 0.5 * (MC_M[:, 3] - MC_M[:, 1]) * (W_2 + W_1)  # equation (60c)


@jit('float64[:](float64)', nopython=True)
def calc_F_vertical(W):
    W_1 = 3.0
    W_2 = 12.0

    if W < W_1:
        return MC_M[:, 0] * W  # equation (60a)
    elif (W >= W_1) & (W <= W_2):
        return MC_M[:, 0] * W + 0.5 * (MC_M[:, 2] - MC_M[:, 0]) * (W - W_1) * (W - W_1) / (W_2 - W_1)  # equation (60b)
    else:
        return MC_M[:, 2] * W - 0.5 * (MC_M[:, 2] - MC_M[:, 0]) * (W_2 + W_1)  # equation (60c)


@jit('float64[:](float64)', nopython=True, parallel=True)
def create_Delta_S2(W):
    delta_S2 = np.zeros((5), dtype=np.float64)

    for i in prange(0, 4):
        delta_S2[i] = 5.22e-3 * (1.0 - 0.00748 * (np.power(37.0 - FREQ[i], 1.3))) * W

    delta_S2[4] = 5.22e-3 * W

    for i in prange(0, 5):
        if delta_S2[i] > 0.069:
            delta_S2[i] = 0.069

    return delta_S2


@jit('float64[:](float64, float64, float64, float64[:], float64[:], float64, float64, float64, float64, complex128[:], float64[:], float64[:], float64 )', nopython=True)
def calc_horizontal_polarised_BT(C_FY, C_MY, C_ow, T_BU, T_D, T_is, T_ow, W, cos_theta_r, sqrt_eps_thet, tau, term_62, theta_d):
    T_BH_FY = T_is * EMISSIVITY_FY_H  # brightness temperature from FY is (horizontal pol)
    T_BH_MY = T_is * EMISSIVITY_MY_H  # brightness temperature from MY is (horizontal pol)

    # equation (45b); horizontal pol.
    rho_H = (cos_theta_r - sqrt_eps_thet) / (cos_theta_r + sqrt_eps_thet)

    # Power reflectivity
    R_0H = np.abs(rho_H) * np.abs(rho_H)  # equation (46); horizontal pol.

    # ------------------------------
    # The wind-roughened Sea Surface
    # ------------------------------
    R_geoH = R_0H - (MC_GEO[:, 1] + MC_GEO[:, 3] * (theta_d - 53.0) + MC_GEO[:, 5] * (T_ow - 288.0) + MC_GEO[:, 7] * (theta_d - 53.0) * (T_ow - 288.0)) * W
    F_H = calc_F_horizontal(W)
    R_H = (1.0 - F_H) * R_geoH  # equation (49) Composite reflectivity for open water; horizontal pol
    E_H = 1.0 - R_H  # surface emissivity for open water; Horizontal pol.; Equation (8);

    # Emissivity for mixed surface
    E_eff_H = C_ow * E_H + C_FY * EMISSIVITY_FY_H + C_MY * EMISSIVITY_MY_H

    # Reflection coefficient for mixed surface
    R_eff_H = 1.0 - E_eff_H
    OmegaH = (6.2 - 0.001 * np.square(37.0 - FREQ)) * term_62 * np.square(tau)  # equation (62a); horizontal pol.
    T_BOmegaH = ((1.0 + OmegaH) * (1.0 - tau) * (T_D - T_C) + T_C) * R_eff_H  # equation (61) horizontal pol.

    # ------
    # Output
    # ------
    T_BH_ow = E_H * T_ow  # Horizontal brightness temperature from open water
    T_BH_overflade = C_ow * T_BH_ow + C_FY * T_BH_FY + C_MY * T_BH_MY  # Horizontal brightness temperature from mixed surface

    # The result for the upwelling brightness temperature at the top of the
    # atmosphere (i.e., the value observed by Earth-orbiting satellites)
    T_BH = (T_BU + (tau * (T_BH_overflade + T_BOmegaH)))  # Equation (10); Horizontal pol.
    return T_BH


@jit('float64[:](float64, float64, float64, float64[:], float64[:], float64, float64, float64, float64, complex128[:], complex128[:], float64[:], float64[:], float64 )', nopython=True)
def calc_vertical_polarised_BT(C_FY, C_MY, C_ow, T_BU, T_D, T_is, T_ow, W, cos_theta_r, epsilon, sqrt_eps_thet, tau, term_62, theta_d):
    T_BV_FY = T_is * EMISSIVITY_FY_V  # brightness temperature from FY is (vertical pol)
    T_BV_MY = T_is * EMISSIVITY_MY_V  # brightness temperature from MY is (vertical pol)

    # equation (45a); vertical pol.
    rho_V = (epsilon * cos_theta_r - sqrt_eps_thet) / (epsilon * cos_theta_r + sqrt_eps_thet)

    # Power reflectivity
    R_0V = np.abs(rho_V) * np.abs(rho_V) + (4.887E-8 - 6.108E-8 * np.power(T_ow - 273.0, 3.0))  # equation (46)+correction vertical pol.

    # ------------------------------
    # The wind-roughened Sea Surface
    # ------------------------------

    R_geoV = R_0V - (MC_GEO[:, 0] + MC_GEO[:, 2] * (theta_d - 53.0) + MC_GEO[:, 4] * (T_ow - 288.0) + MC_GEO[:, 6] * (theta_d - 53.0) * (T_ow - 288.0)) * W
    F_V = calc_F_vertical(W)
    R_V = (1.0 - F_V) * R_geoV  # equation (49) Composite reflectivity for open water; vertical pol
    E_V = 1.0 - R_V  # Surface emissivity for open water; Vertical pol; Equation (8);

    # Emissivity for mixed surface
    E_eff_V = C_ow * E_V + C_FY * EMISSIVITY_FY_V + C_MY * EMISSIVITY_MY_V

    # Reflection coefficient for mixed surface
    R_eff_V = 1.0 - E_eff_V
    OmegaV = (2.5 + 0.018 * (37.0 - FREQ)) * term_62 * np.power(tau, 3.4)  # equation (62b); vertical pol.
    T_BOmegaV = ((1.0 + OmegaV) * (1 - tau) * (T_D - T_C) + T_C) * R_eff_V  # equation (61) vertical pol.

    # ------
    # Output
    # ------
    T_BV_ow = E_V * T_ow  # Vertical brightness temperature from open water

    # The result for the upwelling brightness temperature at the top of the
    # atmosphere (i.e., the value observed by Earth-orbiting satellites)
    T_BV_overflade = C_ow * T_BV_ow + C_FY * T_BV_FY + C_MY * T_BV_MY  # Vertical brightness temperature from mixed surface
    T_BV = (T_BU + (tau * (T_BV_overflade + T_BOmegaV)))  # Equation (10); Vertical pol.
    return T_BV