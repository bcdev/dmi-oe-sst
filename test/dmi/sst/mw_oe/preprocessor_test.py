import unittest

import numpy as np
import xarray as xr
from xarray import Variable

from dmi.sst.mw_oe.preprocessor import Preprocessor
from dmi.sst.util.default_data import DefaultData


class PreprocessorTest(unittest.TestCase):
    def test_run_unmodified_variables(self):
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
        data[5, 3, 3] = 101
        dataset["amsre.nwp.log_surface_pressure"] = Variable(["matchup_count", "ny", "nx"], data)

        data = DefaultData.create_default_array_3d(5, 5, 11, np.float64)
        data[6, 3, 3] = 1.02
        dataset["amsre.nwp.total_column_water_vapour"] = Variable(["matchup_count", "ny", "nx"], data)

        preprocessor = Preprocessor()
        prep_data = preprocessor.run(dataset)

        variable = prep_data.variables["amsre.nwp.log_surface_pressure"]
        self.assertEqual((11,), variable.shape)
        self.assertEqual(101, variable.data[5])

        variable = prep_data.variables["amsre.nwp.total_column_water_vapour"]
        self.assertEqual((11,), variable.shape)
        self.assertAlmostEqual(1.02, variable.data[6], 8)
