import unittest

import numpy as np
import xarray as xr
from xarray import Variable

from dmi.sst.mw_oe.flag_coding import FlagCoding
from dmi.sst.mw_oe.mw_oe_sst_processor import MwOeSstProcessor
from dmi.sst.mw_oe.retrieval import Retrieval
from dmi.sst.util.default_data import DefaultData


class RetrievalTest(unittest.TestCase):
    BT_VARIABLE_NAMES = ["amsre.brightness_temperature6V", "amsre.brightness_temperature6H", "amsre.brightness_temperature10V", "amsre.brightness_temperature10H", "amsre.brightness_temperature18V",
                         "amsre.brightness_temperature18H", "amsre.brightness_temperature23V", "amsre.brightness_temperature23H", "amsre.brightness_temperature36V", "amsre.brightness_temperature36H"]

    retrieval = None

    def setUp(self):
        self.retrieval = Retrieval()

    def test_run(self):
        dataset = xr.Dataset()
        dataset["amsre.brightness_temperature6V"] = Variable(["matchup_count"], np.array([176.4254, 169.4794, 169.3118], dtype=np.float64))
        dataset["amsre.brightness_temperature6H"] = Variable(["matchup_count"], np.array([90.9698, 86.6282, 82.611], dtype=np.float64))
        dataset["amsre.brightness_temperature10V"] = Variable(["matchup_count"], np.array([181.49, 174.48, 173.5316], dtype=np.float64))
        dataset["amsre.brightness_temperature10H"] = Variable(["matchup_count"], np.array([98.8989, 93.5897, 86.9333], dtype=np.float64))
        dataset["amsre.brightness_temperature18V"] = Variable(["matchup_count"], np.array([212.9936, 199.0224, 192.7028], dtype=np.float64))
        dataset["amsre.brightness_temperature18H"] = Variable(["matchup_count"], np.array([153.7084, 133.4428, 115.918], dtype=np.float64))
        dataset["amsre.brightness_temperature23V"] = Variable(["matchup_count"], np.array([250.0724, 226.9596, 215.15], dtype=np.float64))
        dataset["amsre.brightness_temperature23H"] = Variable(["matchup_count"], np.array([219.4987, 184.0727, 153.9983], dtype=np.float64))
        dataset["amsre.brightness_temperature36V"] = Variable(["matchup_count"], np.array([232.129, 220.8774, 217.5726], dtype=np.float64))
        dataset["amsre.brightness_temperature36H"] = Variable(["matchup_count"], np.array([183.7511, 167.1675, 151.3251], dtype=np.float64))
        dataset["amsre.nwp.abs_wind_speed"] = Variable(["matchup_count"], np.array([6.00405503911681, 9.20655470261406, 5.10105703955726], dtype=np.float64))
        dataset["amsre.nwp.total_column_water_vapour"] = Variable(["matchup_count"], np.array([59.2188758850098, 26.8414897918701, 14.3991870880127], dtype=np.float64))
        dataset["amsre.nwp.total_column_liquid_water"] = Variable(["matchup_count"], np.array([0.153278715134895, 0.210529339037996, 0.00125914319133341], dtype=np.float64))
        dataset["amsre.nwp.sea_surface_temperature"] = Variable(["matchup_count"], np.array([301.679042997567, 291.330501737802, 295.058376493661], dtype=np.float64) - 273.15)
        dataset["amsre.satellite_zenith_angle"] = Variable(["matchup_count"], np.array([55.19, 55.23, 55.2], dtype=np.float64))
        dataset["relative_angle"] = Variable(["matchup_count"], np.array([78.1866134486559, 359.01513331183, 26.1734235301682], dtype=np.float64))

        result = MwOeSstProcessor._create_result_structure(3, 10, 10)

        self.retrieval.run(dataset, result, FlagCoding(3))

        self.assertEqual((3, 10), result.j.shape)
        self.assertAlmostEqual(7538.919, result.j.data[0, 0], 3)
        self.assertAlmostEqual(7.849407, result.j.data[1, 1], 6)
        self.assertAlmostEqual(17.829288, result.j.data[2, 2], 6)

        self.assertEqual((3, 10), result.tb_rmse_ite.shape)
        self.assertAlmostEqual(0.89614433, result.tb_rmse_ite.data[0, 3], 7)
        self.assertTrue(np.isnan(result.tb_rmse_ite.data[1, 4]))
        self.assertTrue(np.isnan(result.tb_rmse_ite.data[2, 5]))

        self.assertEqual((3, 10), result.tb_chi_ite.shape)
        self.assertTrue(np.isnan(result.tb_chi_ite.data[0, 6]))
        self.assertTrue(np.isnan(result.tb_chi_ite.data[1, 7]))
        self.assertTrue(np.isnan(result.tb_chi_ite.data[2, 8]))

        self.assertEqual((3,), result.convergence_passed_flag.shape)
        self.assertEqual(1, result.convergence_passed_flag.data[0])
        self.assertEqual(1, result.convergence_passed_flag.data[1])
        self.assertEqual(1, result.convergence_passed_flag.data[2])

        self.assertEqual((3, 10), result.di2.shape)
        self.assertTrue(np.isnan(result.di2.data[0, 9]))
        self.assertAlmostEqual(4071.0098, result.di2.data[1, 0], 4)
        self.assertAlmostEqual(184.93216, result.di2.data[2, 1], 4)

        self.assertEqual((3,), result.i.shape)
        self.assertEqual(4, result.i.data[0])
        self.assertEqual(3, result.i.data[1])
        self.assertEqual(3, result.i.data[2])

        self.assertEqual((3,), result.tb_rmse_ite0.shape)
        self.assertAlmostEqual(5.0651402, result.tb_rmse_ite0.data[0], 6)
        self.assertAlmostEqual(2.7727253, result.tb_rmse_ite0.data[1], 7)
        self.assertAlmostEqual(9.249838, result.tb_rmse_ite0.data[2], 6)

        self.assertEqual((3, 10), result.dtb_ite0.shape)
        self.assertAlmostEqual(-6.1169634, result.dtb_ite0.data[0, 3], 6)
        self.assertAlmostEqual(-0.18351583, result.dtb_ite0.data[1, 4], 7)
        self.assertAlmostEqual(-10.374541, result.dtb_ite0.data[2, 5], 6)

        self.assertEqual((3, 10), result.TA0_ite0.shape)
        self.assertAlmostEqual(256.65015, result.TA0_ite0.data[0, 6], 5)
        self.assertAlmostEqual(181.64264, result.TA0_ite0.data[1, 7], 5)
        self.assertAlmostEqual(209.93489, result.TA0_ite0.data[2, 8], 5)

        self.assertEqual((3,), result.j_ite0.shape)
        self.assertAlmostEqual(24122.828, result.j_ite0.data[0], 3)
        self.assertAlmostEqual(4573.326, result.j_ite0.data[1], 3)
        self.assertAlmostEqual(291431.88, result.j_ite0.data[2], 1)

        self.assertEqual((3, 4), result.A.shape)
        self.assertAlmostEqual(0.9962128, result.A.data[0, 0], 7)
        self.assertAlmostEqual(0.910339, result.A.data[1, 1], 7)
        self.assertAlmostEqual(0.99998724, result.A.data[2, 2], 7)

        self.assertEqual((3,), result.chisq.shape)
        self.assertAlmostEqual(2.7899823, result.chisq.data[0], 7)
        self.assertAlmostEqual(1.5961539, result.chisq.data[1], 7)
        self.assertAlmostEqual(0.710269, result.chisq.data[2], 7)

        self.assertEqual((3,), result.mu_sst.shape)
        self.assertAlmostEqual(0.49287936, result.mu_sst.data[0], 7)
        self.assertAlmostEqual(0.23875031, result.mu_sst.data[1], 7)
        self.assertAlmostEqual(0.18248218, result.mu_sst.data[2], 8)

        self.assertEqual((3, 4), result.x.shape)
        self.assertAlmostEqual(13.069875, result.x.data[0, 0], 6)
        self.assertAlmostEqual(27.391582, result.x.data[1, 1], 5)
        self.assertAlmostEqual(0.120024666, result.x.data[2, 2], 8)

        self.assertEqual((3, 4), result.p0.shape)
        self.assertAlmostEqual(26.84149, result.p0.data[1, 1], 6)
        self.assertAlmostEqual(0.0012591432, result.p0.data[2, 2], 5)
        self.assertAlmostEqual(301.67905, result.p0.data[0, 3], 5)

        self.assertEqual((3, 4), result.S.shape)
        self.assertAlmostEqual(0.123080365, result.S.data[0, 0], 8)
        self.assertAlmostEqual(0.26949105, result.S.data[1, 1], 8)
        self.assertAlmostEqual(0.0035749427, result.S.data[2, 2], 8)

        self.assertEqual((3, 10), result.F.shape)
        self.assertAlmostEqual(98.216354, result.F.data[0, 3], 5)
        self.assertAlmostEqual(199.18938, result.F.data[1, 4], 5)
        self.assertAlmostEqual(115.8715, result.F.data[2, 5], 5)

        self.assertEqual((3, 10), result.y.shape)
        self.assertAlmostEqual(98.8989, result.y.data[0, 3], 5)
        self.assertAlmostEqual(199.0224, result.y.data[1, 4], 5)
        self.assertAlmostEqual(115.918, result.y.data[2, 5], 5)

        self.assertEqual((3, 10), result.dtb.shape)
        self.assertAlmostEqual(0.93957216, result.dtb.data[0, 6], 8)
        self.assertAlmostEqual(-0.40983388, result.dtb.data[1, 7], 8)
        self.assertAlmostEqual(-0.48253775, result.dtb.data[2, 8], 8)

        self.assertEqual((3,), result.ds.shape)
        self.assertAlmostEqual(3.337153, result.ds.data[0], 7)
        self.assertAlmostEqual(3.3807921, result.ds.data[1], 7)
        self.assertAlmostEqual(3.4556863, result.ds.data[2], 7)

        self.assertEqual((3,), result.dn.shape)
        self.assertAlmostEqual(0.662847, result.dn.data[0], 7)
        self.assertAlmostEqual(0.6192079, result.dn.data[1], 7)
        self.assertAlmostEqual(0.54431367, result.dn.data[2], 8)

        self.assertEqual((3, 10), result.K4.shape)
        self.assertAlmostEqual(0.05651569, result.K4.data[0, 9], 8)
        self.assertAlmostEqual(0.55655426, result.K4.data[1, 0], 7)
        self.assertAlmostEqual(0.1891833, result.K4.data[2, 1], 7)

        self.assertEqual((3,), result.ite_index.shape)
        self.assertEqual(4, result.ite_index.data[0])
        self.assertEqual(3, result.ite_index.data[1])
        self.assertEqual(3, result.ite_index.data[2])

    def test_prepare_first_guess(self):
        ws = np.float64(22.3)
        tcwv = np.float64(33.4)
        tclw = np.float64(44.5)
        sst = np.float64(55.6)
        eps = np.array([0.2, 0.1, 0.02, 0.25], dtype=np.float64)

        [p, p_0] = self.retrieval.prepare_first_guess(ws, tcwv, tclw, sst, eps)

        self.assertEqual((4, 3), p.shape)
        self.assertAlmostEqual(22.1, p[0, 0], 8)
        self.assertAlmostEqual(33.5, p[1, 1], 8)
        self.assertAlmostEqual(44.5, p[2, 2], 8)
        self.assertAlmostEqual(328.5, p[3, 0], 8)

        self.assertEqual((4,), p_0.shape)
        self.assertAlmostEqual(22.3, p_0[0], 8)
        self.assertAlmostEqual(33.4, p_0[1], 8)
        self.assertAlmostEqual(44.5, p_0[2], 8)
        self.assertAlmostEqual(328.75, p_0[3], 8)

    def test_create_T_A(self):
        dataset = xr.Dataset()

        for i in range(0, len(self.BT_VARIABLE_NAMES)):
            data = DefaultData.create_default_vector(13, np.float32, fill_value=i * 10)
            dataset[self.BT_VARIABLE_NAMES[i]] = Variable(["matchup_count"], data)

        T_A = self.retrieval.create_T_A(dataset)
        self.assertAlmostEqual(0, T_A[0, 0], 8)
        self.assertAlmostEqual(10, T_A[0, 1], 8)
        self.assertAlmostEqual(20, T_A[0, 2], 8)
        self.assertAlmostEqual(30, T_A[0, 3], 8)
        self.assertAlmostEqual(90, T_A[0, 9], 8)
