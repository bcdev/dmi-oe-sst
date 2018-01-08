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

        self.retrieval.run(dataset, None, FlagCoding(3))

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

