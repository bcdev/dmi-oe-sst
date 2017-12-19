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

        self.fw_model.run(W, V, L, T_ow, T_is, C_is, F_MY, theta_d, sss, phi_rd)

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