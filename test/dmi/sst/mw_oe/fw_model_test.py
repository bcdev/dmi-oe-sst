import unittest

import numpy as np

from dmi.sst.mw_oe.fw_model import FwModel


class FwModelTest(unittest.TestCase):

    fw_model = None

    def setUp(self):
        self.fw_model = FwModel()

    def test_run(self):
        W = np.float64(6.00405503911681)
        V = np.float64(59.2188758850098)
        L = np.float64(0.153278715134895)
        T_ow = np.float64(301.679042997567)
        T_is = np.float64(0.0)
        C_is = np.float64(0.0)
        F_MY = np.float64(0.0)
        theta_d = np.float64(55.19)
        sss = np.float64(34.569)
        phi_rd = np.float64(78.1866134486559)

        T_B = self.fw_model.run(W, V, L, T_ow, T_is, C_is, F_MY, theta_d, sss, phi_rd)
        self.assertAlmostEqual(174.33775927, T_B[0], 8)
        self.assertAlmostEqual(85.05875083, T_B[1], 8)
        self.assertAlmostEqual(180.15518813, T_B[2], 8)
        self.assertAlmostEqual(92.71330992, T_B[3], 8)
        self.assertAlmostEqual(216.90433147, T_B[4], 8)
        self.assertAlmostEqual(155.05875582, T_B[5], 8)
        self.assertAlmostEqual(256.58709764, T_B[6], 8)
        self.assertAlmostEqual(227.27895908, T_B[7], 8)
        self.assertAlmostEqual(238.17140221, T_B[8], 8)
        self.assertAlmostEqual(187.10250911, T_B[9], 8)

    def test_clamp_to_o_1(self):
        self.assertAlmostEqual(0.8, self.fw_model.clamp_to_0_1(0.8), 8)
        self.assertAlmostEqual(0.0, self.fw_model.clamp_to_0_1(-0.1), 8)
        self.assertAlmostEqual(1.0, self.fw_model.clamp_to_0_1(1.1), 8)

    def test_calc_ice_temp(self):
        self.assertAlmostEqual(273.15, self.fw_model.calc_ice_temp(274.2), 8)
        self.assertAlmostEqual(272.08, self.fw_model.calc_ice_temp(272.2), 8)
        self.assertAlmostEqual(271.628, self.fw_model.calc_ice_temp(271.07), 8)

    def test_calc_open_water_temp(self):
        self.assertAlmostEqual(273.15, self.fw_model.calc_open_water_temp(0.08, 274.2), 8)
        self.assertAlmostEqual(274.2, self.fw_model.calc_open_water_temp(0.047, 274.2), 8)

    def test_calc_T_V(self):
        self.assertAlmostEqual(301.14977223818812, self.fw_model.calc_T_V(47.3), 8)
        self.assertAlmostEqual(301.08790967395691, self.fw_model.calc_T_V(46.1), 8)
        self.assertAlmostEqual(301.16, self.fw_model.calc_T_V(51.2), 8)

    def test_calc_sig_TS_TV(self):
        self.assertAlmostEqual(0.545876967999981, self.fw_model.calc_sig_TS_TV(301.68, 301.16), 8)
        self.assertAlmostEqual(14.0, self.fw_model.calc_sig_TS_TV(321.68, 301.16), 8)

    def test_calc_F_horizontal(self):
        F_H = self.fw_model.calc_F_horizontal(6.0041)
        self.assertAlmostEqual(0.0120082, F_H[0], 8)
        self.assertAlmostEqual(0.0120082, F_H[1], 8)
        self.assertAlmostEqual(0.017592013, F_H[2], 8)
        self.assertAlmostEqual(0.018492628, F_H[3], 8)
        self.assertAlmostEqual(0.019753489, F_H[4], 8)

        F_H = self.fw_model.calc_F_horizontal(8.32)
        self.assertAlmostEqual(0.01733696, F_H[0], 8)
        self.assertAlmostEqual(0.01733696, F_H[1], 8)
        self.assertAlmostEqual(0.0250100912, F_H[2], 8)
        self.assertAlmostEqual(0.0262389248, F_H[3], 8)
        self.assertAlmostEqual(0.0279495344, F_H[4], 8)

        F_H = self.fw_model.calc_F_horizontal(12.46)
        self.assertAlmostEqual(0.03676, F_H[0], 8)
        self.assertAlmostEqual(0.03676, F_H[1], 8)
        self.assertAlmostEqual(0.0472526, F_H[2], 8)
        self.assertAlmostEqual(0.048796, F_H[3], 8)
        self.assertAlmostEqual(0.050791, F_H[4], 8)

    def test_calc_F_vertical(self):
        F_V = self.fw_model.calc_F_vertical(2.95)
        self.assertAlmostEqual(0.00059, F_V[0], 8)
        self.assertAlmostEqual(0.00059, F_V[1], 8)
        self.assertAlmostEqual(0.00413, F_V[2], 8)
        self.assertAlmostEqual(0.005251, F_V[3], 8)
        self.assertAlmostEqual(0.0075815, F_V[4], 8)

        F_V = self.fw_model.calc_F_vertical(6.0041)
        self.assertAlmostEqual(0.0045599829237222226, F_V[0], 8)
        self.assertAlmostEqual(0.0045599829237222226, F_V[1], 8)
        self.assertAlmostEqual(0.011393890899311112, F_V[2], 8)
        self.assertAlmostEqual(0.013454847155066667, F_V[3], 8)
        self.assertAlmostEqual(0.017656609146466665, F_V[4], 8)

        F_V = self.fw_model.calc_F_vertical(12.773)
        self.assertAlmostEqual(0.0378837, F_V[0], 8)
        self.assertAlmostEqual(0.0378837, F_V[1], 8)
        self.assertAlmostEqual(0.04930928, F_V[2], 8)
        self.assertAlmostEqual(0.0518429, F_V[3], 8)
        self.assertAlmostEqual(0.05623873, F_V[4], 8)

    def test_create_Delta_S2(self):
        Delta_S2 = self.fw_model.create_Delta_S2(6.0041)

        self.assertAlmostEqual(0.011771282646377359 , Delta_S2[0], 8)
        self.assertAlmostEqual(0.014858456479049636, Delta_S2[1], 8)
        self.assertAlmostEqual(0.021080006317788141, Delta_S2[2], 8)
        self.assertAlmostEqual(0.024630728204465633, Delta_S2[3], 8)
        self.assertAlmostEqual(0.031341401999999997, Delta_S2[4], 8)