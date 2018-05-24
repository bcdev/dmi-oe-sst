import unittest

import numpy as np

from dmi.sst.mw_oe.fw_model import FwModel, create_Delta_S2, calc_F_vertical, calc_F_horizontal, calc_sig_TS_TV, calc_T_V, calc_open_water_temp, calc_ice_temp, clamp_to_0_1, \
    calc_horizontal_polarised_BT, calc_vertical_polarised_BT


class FwModelTest(unittest.TestCase):
    fw_model = None

    def setUp(self):
        self.fw_model = FwModel()

    def test_run(self):
        W = np.float64(6.00405503911681)
        V = np.float64(59.2188758850098)
        L = np.float64(0.153278715134895)
        T_ow = np.float64(301.679042997567)
        C_is = np.float64(0.0)
        F_MY = np.float64(0.0)
        theta_d = np.float64(55.19)
        sss = np.float64(35.0)
        phi_rd = np.float64(78.1866134486559)

        T_B = self.fw_model.run(W, V, L, T_ow, C_is, F_MY, theta_d, sss, phi_rd)
        self.assertAlmostEqual(174.13541100122336, T_B[0], 8)
        self.assertAlmostEqual(84.74181288210025, T_B[1], 8)
        self.assertAlmostEqual(180.1933884638022, T_B[2], 8)
        self.assertAlmostEqual(92.7819366193778, T_B[3], 8)
        self.assertAlmostEqual(217.13998230466814, T_B[4], 8)
        self.assertAlmostEqual(155.30385272663318, T_B[5], 8)
        self.assertAlmostEqual(256.65013527564315, T_B[6], 8)
        self.assertAlmostEqual(227.4170699382585, T_B[7], 8)
        self.assertAlmostEqual(238.14396039699585, T_B[8], 8)
        self.assertAlmostEqual(187.14834061640363, T_B[9], 8)

    def test_run_atmosphere_T_V_is_switch(self):
        W = np.float64(9.20655470261406)
        V = np.float64(26.8414897918701)
        L = np.float64(0.210529339037996)
        T_ow = np.float64(291.330501737802)
        C_is = np.float64(0.0)
        F_MY = np.float64(0.0)
        theta_d = np.float64(55.23)
        sss = np.float64(34.304)
        phi_rd = np.float64(359.01513331183)

        T_B = self.fw_model.run(W, V, L, T_ow, C_is, F_MY, theta_d, sss, phi_rd)
        self.assertAlmostEqual(167.98604026913313, T_B[0], 8)
        self.assertAlmostEqual(83.41538029242838, T_B[1], 8)
        self.assertAlmostEqual(173.25540530742637, T_B[2], 8)
        self.assertAlmostEqual(89.81690057786818, T_B[3], 8)
        self.assertAlmostEqual(198.84180703778293, T_B[4], 8)
        self.assertAlmostEqual(130.27113012179473, T_B[5], 8)
        self.assertAlmostEqual(227.80402981643374, T_B[6], 8)
        self.assertAlmostEqual(181.6445511057775, T_B[7], 8)
        self.assertAlmostEqual(225.2227088044324, T_B[8], 8)
        self.assertAlmostEqual(170.7638337588519, T_B[9], 8)

    def test_run_trigger_F_H_switch(self):
        W = np.float64(16.44149177243)
        V = np.float64(51.8229067300751)
        L = np.float64(-0.0104714243105632)
        T_ow = np.float64(301.499473924529)
        C_is = np.float64(0.0)
        F_MY = np.float64(0.0)
        theta_d = np.float64(55.19)
        sss = np.float64(34.569)
        phi_rd = np.float64(78.1866134486559)

        T_B = self.fw_model.run(W, V, L, T_ow, C_is, F_MY, theta_d, sss, phi_rd)
        self.assertAlmostEqual(178.48273715969287, T_B[0], 8)
        self.assertAlmostEqual(93.96022168791501, T_B[1], 8)
        self.assertAlmostEqual(184.12959759826413, T_B[2], 8)
        self.assertAlmostEqual(102.81385414538431, T_B[3], 8)
        self.assertAlmostEqual(215.92970970257147, T_B[4], 8)
        self.assertAlmostEqual(158.90974073570703, T_B[5], 8)
        self.assertAlmostEqual(252.17880332198092, T_B[6], 8)
        self.assertAlmostEqual(223.85285102170135, T_B[7], 8)
        self.assertAlmostEqual(230.78179146408655, T_B[8], 8)

    def test_run_trigger_F_V_switch(self):
        W = np.float64(2.72486789578787)
        V = np.float64(54.0632878810604)
        L = np.float64(-0.0340107990629162)
        T_ow = np.float64(302.427070713692)
        C_is = np.float64(0.0)
        F_MY = np.float64(0.0)
        theta_d = np.float64(55.025)
        sss = np.float64(34.783)
        phi_rd = np.float64(106.824341933135)

        T_B = self.fw_model.run(W, V, L, T_ow, C_is, F_MY, theta_d, sss, phi_rd)
        self.assertAlmostEqual(174.03823292632615, T_B[0], 8)
        self.assertAlmostEqual(82.32108819034534, T_B[1], 8)
        self.assertAlmostEqual(179.18918041258235, T_B[2], 8)
        self.assertAlmostEqual(88.39021407083544, T_B[3], 8)
        self.assertAlmostEqual(212.48133684859795, T_B[4], 8)
        self.assertAlmostEqual(143.32147410812388, T_B[5], 8)
        self.assertAlmostEqual(251.7432385519055, T_B[6], 8)
        self.assertAlmostEqual(215.123603830529, T_B[7], 8)
        self.assertAlmostEqual(229.71126790780409, T_B[8], 8)
        self.assertAlmostEqual(165.5727765065683, T_B[9], 8)

    def test_run_delta_s2_clamp(self):
        W = np.float64(15.3255888024779)
        V = np.float64(35.6279405574965)
        L = np.float64(0.44920225229521)
        T_ow = np.float64(293.231309246535)
        C_is = np.float64(0.0)
        F_MY = np.float64(0.0)
        theta_d = np.float64(54.965)
        sss = np.float64(34.541)
        phi_rd = np.float64(112.33381762016)

        T_B = self.fw_model.run(W, V, L, T_ow, C_is, F_MY, theta_d, sss, phi_rd)
        self.assertAlmostEqual(173.38334700878795, T_B[0], 8)
        self.assertAlmostEqual(92.99182381324476, T_B[1], 8)
        self.assertAlmostEqual(180.115881908117, T_B[2], 8)
        self.assertAlmostEqual(103.19560474174375, T_B[3], 8)
        self.assertAlmostEqual(210.3295471861161, T_B[4], 8)
        self.assertAlmostEqual(154.28519178686173, T_B[5], 8)
        self.assertAlmostEqual(241.96600618603588, T_B[6], 8)
        self.assertAlmostEqual(209.7770317468894, T_B[7], 8)
        self.assertAlmostEqual(237.86555635133902, T_B[8], 8)

    def test_clamp_to_o_1(self):
        self.assertAlmostEqual(0.8, clamp_to_0_1(0.8), 8)
        self.assertAlmostEqual(0.0, clamp_to_0_1(-0.1), 8)
        self.assertAlmostEqual(1.0, clamp_to_0_1(1.1), 8)

    def test_calc_ice_temp(self):
        self.assertAlmostEqual(273.15, calc_ice_temp(274.2), 8)
        self.assertAlmostEqual(272.08, calc_ice_temp(272.2), 8)
        self.assertAlmostEqual(271.628, calc_ice_temp(271.07), 8)

    def test_calc_open_water_temp(self):
        self.assertAlmostEqual(273.15, calc_open_water_temp(0.08, 274.2), 8)
        self.assertAlmostEqual(274.2, calc_open_water_temp(0.047, 274.2), 8)

    def test_calc_T_V(self):
        self.assertAlmostEqual(301.14977223818812, calc_T_V(47.3), 8)
        self.assertAlmostEqual(301.08790967395691, calc_T_V(46.1), 8)
        self.assertAlmostEqual(301.16, calc_T_V(51.2), 8)

    def test_calc_sig_TS_TV(self):
        self.assertAlmostEqual(0.545876967999981, calc_sig_TS_TV(301.68, 301.16), 8)
        self.assertAlmostEqual(14.0, calc_sig_TS_TV(321.68, 301.16), 8)

    def test_calc_F_horizontal(self):
        F_H = calc_F_horizontal(6.0041)
        self.assertAlmostEqual(0.0120082, F_H[0], 8)
        self.assertAlmostEqual(0.0120082, F_H[1], 8)
        self.assertAlmostEqual(0.017592013, F_H[2], 8)
        self.assertAlmostEqual(0.018492628, F_H[3], 8)
        self.assertAlmostEqual(0.019753489, F_H[4], 8)

        F_H = calc_F_horizontal(8.32)
        self.assertAlmostEqual(0.01733696, F_H[0], 8)
        self.assertAlmostEqual(0.01733696, F_H[1], 8)
        self.assertAlmostEqual(0.0250100912, F_H[2], 8)
        self.assertAlmostEqual(0.0262389248, F_H[3], 8)
        self.assertAlmostEqual(0.0279495344, F_H[4], 8)

        F_H = calc_F_horizontal(12.46)
        self.assertAlmostEqual(0.03676, F_H[0], 8)
        self.assertAlmostEqual(0.03676, F_H[1], 8)
        self.assertAlmostEqual(0.0472526, F_H[2], 8)
        self.assertAlmostEqual(0.048796, F_H[3], 8)
        self.assertAlmostEqual(0.050791, F_H[4], 8)

    def test_calc_F_vertical(self):
        F_V = calc_F_vertical(2.95)
        self.assertAlmostEqual(0.00059, F_V[0], 8)
        self.assertAlmostEqual(0.00059, F_V[1], 8)
        self.assertAlmostEqual(0.00413, F_V[2], 8)
        self.assertAlmostEqual(0.005251, F_V[3], 8)
        self.assertAlmostEqual(0.0075815, F_V[4], 8)

        F_V = calc_F_vertical(6.0041)
        self.assertAlmostEqual(0.0045599829237222226, F_V[0], 8)
        self.assertAlmostEqual(0.0045599829237222226, F_V[1], 8)
        self.assertAlmostEqual(0.011393890899311112, F_V[2], 8)
        self.assertAlmostEqual(0.013454847155066667, F_V[3], 8)
        self.assertAlmostEqual(0.017656609146466665, F_V[4], 8)

        F_V = calc_F_vertical(12.773)
        self.assertAlmostEqual(0.0378837, F_V[0], 8)
        self.assertAlmostEqual(0.0378837, F_V[1], 8)
        self.assertAlmostEqual(0.04930928, F_V[2], 8)
        self.assertAlmostEqual(0.0518429, F_V[3], 8)
        self.assertAlmostEqual(0.05623873, F_V[4], 8)

    def test_create_Delta_S2(self):
        Delta_S2 = create_Delta_S2(6.0041)

        self.assertAlmostEqual(0.011771282646377359, Delta_S2[0], 8)
        self.assertAlmostEqual(0.014858456479049636, Delta_S2[1], 8)
        self.assertAlmostEqual(0.021080006317788141, Delta_S2[2], 8)
        self.assertAlmostEqual(0.024630728204465633, Delta_S2[3], 8)
        self.assertAlmostEqual(0.031341401999999997, Delta_S2[4], 8)

    def test_calc_horizontal_polarised_BT(self):
        C_FY = np.float64(0.0)
        C_MY = np.float64(0.0)
        C_OW = np.float64(1.0)
        T_BU = np.asarray([5.715619033241641, 8.955752553740098, 43.31807669457342, 103.71525118258761, 63.50157111928275])
        T_D = np.asarray([277.9079227797001, 281.7780772117912, 287.9742385877552, 287.9005203213625, 286.158209347942])
        T_is = np.float64(273.15)
        T_ow = np.float64(302.24587478637693)
        W = np.float64(6.851329803466797)
        cos_theta_r = np.float64(0.5731474681972756)
        sqrt_eps_thet = np.asarray(
            [(8.18068397903982 - 2.0265181632277742j), (7.835871119302394 - 2.2060117452380354j), (7.0089776625838 - 2.6333023300672145j), (6.510633581794293 - 2.781988836812065j),
             (5.509650824400042 - 2.8611309670278877j)])
        tau = np.asarray([0.9794182550786318, 0.9681861599331942, 0.849093784207826, 0.6365378317235144, 0.7768085298376742])
        term_62 = np.asarray([0.013262662593043513, 0.016613918713500434, 0.02308027884647855, 0.026552120915210645, 0.03256184688580385])
        theta_d = np.float64(55.029998779296875)

        T_BH = calc_horizontal_polarised_BT(C_FY, C_MY, C_OW, T_BU, T_D, T_is, T_ow, W, cos_theta_r, sqrt_eps_thet, tau, term_62, theta_d)

        self.assertAlmostEqual(84.55799647378079, T_BH[0], 8)
        self.assertAlmostEqual(91.30348957877695, T_BH[1], 8)
        self.assertAlmostEqual(143.45994630454751, T_BH[2], 8)
        self.assertAlmostEqual(209.38622322820117, T_BH[3], 8)
        self.assertAlmostEqual(174.85439033783067, T_BH[4], 8)

    def test_calc_vertical_polarised_BT(self):
        C_FY = np.float64(0.0)
        C_MY = np.float64(0.0)
        C_OW = np.float64(1.0)
        T_BU = np.asarray([5.715619033241641, 8.955752553740098, 43.31807669457342, 103.71525118258761, 63.50157111928275])
        T_D = np.asarray([277.9079227797001, 281.7780772117912, 287.9742385877552, 287.9005203213625, 286.158209347942])
        T_is = np.float64(273.15)
        T_ow = np.float64(302.24587478637693)
        W = np.float64(6.851329803466797)
        cos_theta_r = np.float64(0.5731474681972756)
        epsilon = np.asarray([(63.488316478725764 - 33.15660934230131j), (57.20589035788824 - 34.572047446705184j), (42.8629886927603 - 36.91351442054196j), (35.320389727939684 - 36.22501989025095j),
                              (22.841683776025178 - 31.527665182403382j)])
        sqrt_eps_thet = np.asarray(
            [(8.18068397903982 - 2.0265181632277742j), (7.835871119302394 - 2.2060117452380354j), (7.0089776625838 - 2.6333023300672145j), (6.510633581794293 - 2.781988836812065j),
             (5.509650824400042 - 2.8611309670278877j)])
        tau = np.asarray([0.9794182550786318, 0.9681861599331942, 0.849093784207826, 0.6365378317235144, 0.7768085298376742])
        term_62 = np.asarray([0.013262662593043513, 0.016613918713500434, 0.02308027884647855, 0.026552120915210645, 0.03256184688580385])
        theta_d = np.float64(55.029998779296875)

        T_BH = calc_vertical_polarised_BT(C_FY, C_MY, C_OW, T_BU, T_D, T_is, T_ow, W, cos_theta_r, epsilon, sqrt_eps_thet, tau, term_62, theta_d)

        self.assertAlmostEqual(173.50023752824694, T_BH[0], 8)
        self.assertAlmostEqual(178.80704193888405, T_BH[1], 8)
        self.assertAlmostEqual(210.09393012654223, T_BH[2], 8)
        self.assertAlmostEqual(246.4076719622741, T_BH[3], 8)
        self.assertAlmostEqual(231.17052656945668, T_BH[4], 8)
