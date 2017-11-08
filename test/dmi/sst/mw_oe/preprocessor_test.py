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
        dataset["insitu.time"] = Variable(["matchup_count", "ny", "nx"], data)

        data = DefaultData.create_default_array_3d(1, 1, 7, np.float32)
        dataset["insitu.lon"] = Variable(["matchup_count", "ny", "nx"], data)

        preprocessor = Preprocessor()
        prep_data = preprocessor.run(dataset)

        variable = prep_data.variables["insitu.time"]
        self.assertEqual((7,), variable.shape)

        variable = prep_data.variables["insitu.lon"]
        self.assertEqual((7,), variable.shape)

    def test_run_fetch_center_pixel_variables(self):
        dataset = xr.Dataset()

        data = DefaultData.create_default_array_3d(5, 5, 11, np.int32)
        dataset["amsre.nwp.log_surface_pressure"] = Variable(["matchup_count", "ny", "nx"], data)

        data = DefaultData.create_default_array_3d(5, 5, 11, np.float32)
        dataset["amsre.pixel_data_quality23V"] = Variable(["matchup_count", "ny", "nx"], data)

        preprocessor = Preprocessor()
        prep_data = preprocessor.run(dataset)

        # @todo 1 tb/tb continue here 2017-11-07
        # variable = prep_data.variables["amsre.nwp.log_surface_pressure"]
        # self.assertEqual((11,), variable.shape)
        #
        # variable = prep_data.variables["amsre.pixel_data_quality23V"]
        # self.assertEqual((11,), variable.shape)
