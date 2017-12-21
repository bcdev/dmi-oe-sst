import unittest

import numpy as np
import xarray as xr
from xarray import Variable

from dmi.sst.mw_oe.flag_coding import FlagCoding
from dmi.sst.mw_oe.retrieval import Retrieval
from dmi.sst.util.default_data import DefaultData


class RetrievalTest(unittest.TestCase):

    BT_VARIABLE_NAMES = ["amsre.brightness_temperature6V", "amsre.brightness_temperature6H", "amsre.brightness_temperature10V", "amsre.brightness_temperature10H", "amsre.brightness_temperature18V",
                         "amsre.brightness_temperature18H", "amsre.brightness_temperature23V", "amsre.brightness_temperature23H", "amsre.brightness_temperature36V", "amsre.brightness_temperature36H"]

    retrieval = None

    def setUp(self):
        self.retrieval = Retrieval()

    # @todo 1 tb/tb continue here 2017-12-21
    # def test_run(self):
    #     dataset = xr.Dataset()
    #     dataset["amsre.brightness_temperature6V"] = Variable(["matchup_count"], np.array([176.425, 169.479, 169.312], dtype=np.float64))
    #     dataset["amsre.brightness_temperature6H"] = Variable(["matchup_count"], np.array([90.970, 86.628, 82.412], dtype=np.float64))
    #     dataset["amsre.brightness_temperature10V"] = Variable(["matchup_count"], np.array([181.509, 174.499, 173.551], dtype=np.float64))
    #     dataset["amsre.brightness_temperature10H"] = Variable(["matchup_count"], np.array([99.652, 94.343, 87.686], dtype=np.float64))
    #     dataset["amsre.brightness_temperature18V"] = Variable(["matchup_count"], np.array([212.392, 198.420, 192.101], dtype=np.float64))
    #     dataset["amsre.brightness_temperature18H"] = Variable(["matchup_count"], np.array([153.518, 133.252, 115.727], dtype=np.float64))
    #     dataset["amsre.brightness_temperature23V"] = Variable(["matchup_count"], np.array([250.064, 226.951, 215.141], dtype=np.float64))
    #     dataset["amsre.brightness_temperature23H"] = Variable(["matchup_count"], np.array([219.831, 184.405, 154.331], dtype=np.float64))
    #     dataset["amsre.brightness_temperature36V"] = Variable(["matchup_count"], np.array([232.464, 221.212, 217.907], dtype=np.float64))
    #     dataset["amsre.brightness_temperature36H"] = Variable(["matchup_count"], np.array([183.694, 167.110, 151.268], dtype=np.float64))
    #     dataset["amsre.nwp.abs_wind_speed"] = Variable(["matchup_count"], np.array([6.0041, 9.2066, 5.1011], dtype=np.float64))
    #     dataset["amsre.nwp.total_column_water_vapour"] = Variable(["matchup_count"], np.array([59.2189, 26.8415, 14.3992], dtype=np.float64))
    #     dataset["amsre.nwp.total_column_liquid_water"] = Variable(["matchup_count"], np.array([0.1532787, 0.2105293, 0.0012591], dtype=np.float64))
    #     dataset["amsre.nwp.sea_surface_temperature"] = Variable(["matchup_count"], np.array([301.68, 291.33, 295.06], dtype=np.float64) - 273.15)
    #     dataset["amsre.solar_zenith_angle"] = Variable(["matchup_count"], np.array([36.590, 62.030, 150.200], dtype=np.float64))
    #     dataset["relative_angle"] = Variable(["matchup_count"], np.array([78.187, 359.02, 26.173], dtype=np.float64))
    #
    #     self.retrieval.run(dataset, None, FlagCoding(3))

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

