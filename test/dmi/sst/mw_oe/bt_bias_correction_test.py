import unittest

import numpy as np
import xarray as xr
from xarray import Variable

from dmi.sst.mw_oe.bt_bias_correction import BtBiasCorrection
from dmi.sst.util.default_data import DefaultData


class BtBiasCorrectionTest(unittest.TestCase):
    BT_VARIABLE_NAMES = ["amsre.brightness_temperature6V", "amsre.brightness_temperature6H", "amsre.brightness_temperature10V", "amsre.brightness_temperature10H", "amsre.brightness_temperature18V",
                         "amsre.brightness_temperature18H", "amsre.brightness_temperature23V", "amsre.brightness_temperature23H", "amsre.brightness_temperature36V", "amsre.brightness_temperature36H"]

    def test_run(self):
        dataset = xr.Dataset()

        for i in range(0, len(self.BT_VARIABLE_NAMES)):
            data = DefaultData.create_default_vector(13, np.float32, fill_value=26)
            dataset[self.BT_VARIABLE_NAMES[i]] = Variable(["matchup"], data)

        bt_bias_correction = BtBiasCorrection()
        bt_bias_correction.run(dataset)

        self.assertAlmostEqual(26.3524, dataset["amsre.brightness_temperature6V"].data[0], 6)
        self.assertAlmostEqual(26.0793, dataset["amsre.brightness_temperature6H"].data[1], 6)
        self.assertAlmostEqual(25.8907, dataset["amsre.brightness_temperature10V"].data[2], 5)
        self.assertAlmostEqual(25.2479, dataset["amsre.brightness_temperature10H"].data[3], 6)
        self.assertAlmostEqual(26.6228, dataset["amsre.brightness_temperature18V"].data[4], 5)
        self.assertAlmostEqual(26.2794, dataset["amsre.brightness_temperature18H"].data[5], 6)
        self.assertAlmostEqual(26.02, dataset["amsre.brightness_temperature23V"].data[6], 5)
        self.assertAlmostEqual(25.7331, dataset["amsre.brightness_temperature23H"].data[7], 5)
        self.assertAlmostEqual(25.646, dataset["amsre.brightness_temperature36V"].data[8], 6)
        self.assertAlmostEqual(26.1464, dataset["amsre.brightness_temperature36H"].data[9], 5)
