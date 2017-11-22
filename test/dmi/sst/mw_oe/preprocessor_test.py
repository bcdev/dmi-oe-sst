import unittest

import numpy as np
import xarray as xr
from xarray import Variable

from dmi.sst.mw_oe.preprocessor import Preprocessor
from dmi.sst.util.default_data import DefaultData


class PreprocessorTest(unittest.TestCase):
    dataset = None
    preprocessor = None

    def setUp(self):
        self.preprocessor = Preprocessor()

        self.dataset = xr.Dataset()
        data = DefaultData.create_default_array_3d(5, 5, 11, np.int8)
        self.dataset["amsre.nwp.log_surface_pressure"] = Variable(["macwenameifdifferentlyhere", "ny_test", "nx_test"], data)
        data = DefaultData.create_default_array_4d(5, 5, 60, 11, np.float32, fill_value=np.NaN)
        self.dataset["amsre.nwp.cloud_liquid_water"] = Variable(["macwenameifdifferentlyhere", "cloud_layers", "ny_test", "nx_test"], data)

    def test_run_dimension_squeezed_variables(self):
        data = DefaultData.create_default_array_3d(1, 1, 7, np.int32)
        data[3, 0, 0] = 78
        self.dataset["insitu.time"] = Variable(["matchup_count", "ny", "nx"], data)

        data = DefaultData.create_default_array_3d(1, 1, 7, np.float32)
        data[4, 0, 0] = 0.89
        self.dataset["insitu.lon"] = Variable(["matchup_count", "ny", "nx"], data)

        prep_data = self.preprocessor.run(self.dataset)

        variable = prep_data.variables["insitu.time"]
        self.assertEqual((7,), variable.shape)
        self.assertEqual(78, variable.data[3])

        variable = prep_data.variables["insitu.lon"]
        self.assertEqual((7,), variable.shape)
        self.assertAlmostEqual(0.89, variable.data[4], 7)

    def test_run_fetch_center_pixel_variables(self):
        data = DefaultData.create_default_array_3d(5, 5, 11, np.int16)
        data[5, 2, 2] = 101
        self.dataset["amsre.pixel_data_quality6V"] = Variable(["matchup_count", "ny", "nx"], data)

        data = DefaultData.create_default_array_3d(5, 5, 11, np.float64)
        data[6, 2, 2] = 1.02
        self.dataset["amsre.nwp.total_column_water_vapour"] = Variable(["matchup_count", "ny", "nx"], data)

        prep_data = self.preprocessor.run(self.dataset)

        variable = prep_data.variables["amsre.pixel_data_quality6V"]
        self.assertEqual((11,), variable.shape)
        self.assertEqual(101, variable.data[5])

        variable = prep_data.variables["amsre.nwp.total_column_water_vapour"]
        self.assertEqual((11,), variable.shape)
        self.assertAlmostEqual(1.02, variable.data[6], 8)

    def test_run_average_variables(self):
        data = DefaultData.create_default_array_3d(7, 7, 5, np.int16)
        data[0, 1, :] = 102
        data[0, 2, :] = 101
        data[0, 3, :] = 103
        data[0, 4, :] = 105
        data[0, 5, :] = 104
        variable = Variable(["matchup_count", "ny", "nx"], data)
        variable.attrs["_FillValue"] = -78
        self.dataset["amsre.brightness_temperature6V"] = variable

        prep_data = self.preprocessor.run(self.dataset)

        variable = prep_data.variables["amsre.brightness_temperature6V"]
        self.assertEqual((5,), variable.shape)
        self.assertEqual(103, variable.data[0])

        self.assertEqual(0, prep_data.variables["invalid_data"].data[0])

    def test_run_average_variables_masks_invalid(self):
        fill_value = -79

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
        self.dataset["amsre.brightness_temperature6H"] = variable

        prep_data = self.preprocessor.run(self.dataset)

        variable = prep_data.variables["amsre.brightness_temperature6H"]
        self.assertEqual((4,), variable.shape)
        self.assertAlmostEqual(106.04348, variable.data[0], 5)

        self.assertEqual(0, prep_data.variables["invalid_data"].data[0])

    def test_run_average_variables_too_many_invalid(self):
        fill_value = -79

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
        self.dataset["amsre.brightness_temperature6H"] = variable

        prep_data = self.preprocessor.run(self.dataset)

        variable = prep_data.variables["amsre.brightness_temperature6H"]
        self.assertEqual((4,), variable.shape)
        self.assertAlmostEqual(fill_value, variable.data[0], 5)

        self.assertEqual(1, prep_data.variables["invalid_data"].data[0])

    def test_run_total_column_water_vapour(self):
        self.dataset = xr.Dataset()

        data = DefaultData.create_default_array_3d(3, 3, 11, np.float32, fill_value=np.NaN)
        data[0, :, :] = 11.52932
        data[1, :, :] = 11.529235
        data[2, :, :] = 11.52744
        self.dataset["amsre.nwp.log_surface_pressure"] = Variable(["matchup_count", "ny", "nx"], data)

        data = DefaultData.create_default_array_4d(3, 3, 60, 11, np.float32, fill_value=np.NaN)
        data[0, :, 1, 1] = np.float32(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.3609621E-8, 3.7061102E-7, 4.273528E-6, 2.6397798E-5, 3.5652188E-6, 1.5417224E-4, 1.9648654E-4, 9.139678E-5, 8.232285E-6, 6.5089694E-7, 4.0974975E-9, 0.0]))
        data[1, :, 1, 1] = np.float32(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0075588E-7, 1.054102E-6, 1.1424428E-5, 2.2602183E-6, 1.546818E-4, 1.9482679E-4, 8.667531E-5, 4.7594845E-6, 2.708914E-9, 0.0, 0.0]))
        data[2, :, 1, 1] = np.float32(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.7519864E-9, 4.7594305E-8, 1.3165399E-6, 1.2988783E-4, 8.855922E-6, 3.5680281E-7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.dataset["amsre.nwp.cloud_liquid_water"] = Variable(["matchup_count", "cloud_layers", "ny", "nx"], data)

        prep_data = self.preprocessor.run(self.dataset)
        variable = prep_data.variables["amsre.nwp.total_column_liquid_water"]
        self.assertEqual((11,), variable.shape)
        self.assertAlmostEqual(0.064784832, variable.data[0], 8)
        self.assertAlmostEqual(0.059696469, variable.data[1], 8)
        self.assertAlmostEqual(0.028990047, variable.data[2], 8)