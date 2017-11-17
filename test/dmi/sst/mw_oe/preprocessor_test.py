import unittest

import numpy as np
import xarray as xr
from xarray import Variable

from dmi.sst.mw_oe.preprocessor import Preprocessor
from dmi.sst.util.default_data import DefaultData


class PreprocessorTest(unittest.TestCase):
    def test_run_dimension_squeezed_variables(self):
        dataset = xr.Dataset()

        data = DefaultData.create_default_array_3d(1, 1, 7, np.int32)
        data[3, 0, 0] = 78
        dataset["insitu.time"] = Variable(["matchup_count", "ny", "nx"], data)

        data = DefaultData.create_default_array_3d(1, 1, 7, np.float32)
        data[4, 0, 0] = 0.89
        dataset["insitu.lon"] = Variable(["matchup_count", "ny", "nx"], data)

        preprocessor = Preprocessor()
        prep_data = preprocessor.run(dataset)

        variable = prep_data.variables["insitu.time"]
        self.assertEqual((7,), variable.shape)
        self.assertEqual(78, variable.data[3])

        variable = prep_data.variables["insitu.lon"]
        self.assertEqual((7,), variable.shape)
        self.assertAlmostEqual(0.89, variable.data[4], 7)

    def test_run_fetch_center_pixel_variables(self):
        dataset = xr.Dataset()

        data = DefaultData.create_default_array_3d(5, 5, 11, np.int16)
        data[5, 2, 2] = 101
        dataset["amsre.pixel_data_quality6V"] = Variable(["matchup_count", "ny", "nx"], data)

        data = DefaultData.create_default_array_3d(5, 5, 11, np.float64)
        data[6, 2, 2] = 1.02
        dataset["amsre.nwp.total_column_water_vapour"] = Variable(["matchup_count", "ny", "nx"], data)

        preprocessor = Preprocessor()
        prep_data = preprocessor.run(dataset)

        variable = prep_data.variables["amsre.pixel_data_quality6V"]
        self.assertEqual((11,), variable.shape)
        self.assertEqual(101, variable.data[5])

        variable = prep_data.variables["amsre.nwp.total_column_water_vapour"]
        self.assertEqual((11,), variable.shape)
        self.assertAlmostEqual(1.02, variable.data[6], 8)

    def test_run_average_variables(self):
        dataset = xr.Dataset()

        data = DefaultData.create_default_array_3d(7, 7, 5, np.int16)
        data[0, 1, :] = 102
        data[0, 2, :] = 101
        data[0, 3, :] = 103
        data[0, 4, :] = 105
        data[0, 5, :] = 104
        variable = Variable(["matchup_count", "ny", "nx"], data)
        variable.attrs["_FillValue"] = -78
        dataset["amsre.brightness_temperature6V"] = variable

        preprocessor = Preprocessor()
        prep_data = preprocessor.run(dataset)

        variable = prep_data.variables["amsre.brightness_temperature6V"]
        self.assertEqual((5,), variable.shape)
        self.assertEqual(103, variable.data[0])

        self.assertEqual(0, prep_data.variables["invalid_data"].data[0])

    def test_run_average_variables_masks_invalid(self):
        fill_value = -79
        dataset = xr.Dataset()

        data = DefaultData.create_default_array_3d(9, 9, 4, np.int16)
        data[0, 2, :] = 104
        data[0, 3, :] = 105
        data[0, 4, :] = 106
        data[0, 5, :] = 107
        data[0, 6, :] = 108

        data[0, 3, 4] = fill_value
        data[0, 4, 6] = fill_value
        variable = Variable(["matchup_count", "ny", "nx"], data)
        variable.attrs["_FillValue"] = fill_value
        dataset["amsre.brightness_temperature6H"] = variable

        preprocessor = Preprocessor()
        prep_data = preprocessor.run(dataset)

        variable = prep_data.variables["amsre.brightness_temperature6H"]
        self.assertEqual((4,), variable.shape)
        self.assertAlmostEqual(106.04348, variable.data[0], 5)

        self.assertEqual(0, prep_data.variables["invalid_data"].data[0])

    def test_run_average_variables_too_many_invalid(self):
        fill_value = -79
        dataset = xr.Dataset()

        data = DefaultData.create_default_array_3d(5, 5, 4, np.int16)
        data[0, 0, :] = 104
        data[0, 1, :] = 105
        data[0, 2, :] = 106
        data[0, 3, :] = 107
        data[0, 4, :] = 108

        data[0, 3, 0] = fill_value
        data[0, 3, 1] = fill_value
        data[0, 3, 2] = fill_value
        data[0, 4, 1] = fill_value
        variable = Variable(["matchup_count", "ny", "nx"], data)
        variable.attrs["_FillValue"] = fill_value
        dataset["amsre.brightness_temperature6H"] = variable

        preprocessor = Preprocessor()
        prep_data = preprocessor.run(dataset)

        variable = prep_data.variables["amsre.brightness_temperature6H"]
        self.assertEqual((4,), variable.shape)
        self.assertAlmostEqual(fill_value, variable.data[0], 5)

        self.assertEqual(1, prep_data.variables["invalid_data"].data[0])


