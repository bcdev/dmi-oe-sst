import unittest

import numpy as np
import xarray as xr
from xarray import Variable

from dmi.sst.mw_oe.mmd_reader import MmdReader
from dmi.sst.util.default_data import DefaultData


class MmdReaderTest(unittest.TestCase):
    def test_scale_data_no_scaling(self):
        array = DefaultData.create_default_array(6, 12, np.int16)
        array[0, 0] = 22
        array[1, 0] = 23
        variable = Variable(["y", "x"], array)

        MmdReader._scale_data(variable)

        self.assertEqual(22, variable.data[0, 0])
        self.assertEqual(23, variable.data[1, 0])

    def test_scale_data_only_scaling(self):
        array = DefaultData.create_default_array(6, 11, np.int16)
        array[2, 0] = 22
        array[3, 0] = 23
        variable = Variable(["y", "x"], array)
        variable.attrs["SCALE_FACTOR"] = 0.2

        MmdReader._scale_data(variable)

        self.assertAlmostEqual(4.4, variable.data[2, 0], 8)
        self.assertAlmostEqual(4.6, variable.data[3, 0], 8)

    def test_scale_data_only_offset(self):
        array = DefaultData.create_default_array(6, 10, np.int16)
        array[4, 0] = 22
        array[5, 0] = 23
        variable = Variable(["y", "x"], array)
        variable.attrs["OFFSET"] = -10

        MmdReader._scale_data(variable)

        self.assertAlmostEqual(12, variable.data[4, 0], 8)
        self.assertAlmostEqual(13, variable.data[5, 0], 8)

    def test_scale_data_scaling_and_offset(self):
        array = DefaultData.create_default_array(6, 9, np.int16)
        array[0, 1] = 22
        array[1, 1] = 23
        variable = Variable(["y", "x"], array)
        variable.attrs["SCALE_FACTOR"] = 0.2
        variable.attrs["OFFSET"] = 1.6

        MmdReader._scale_data(variable)

        self.assertAlmostEqual(6.0, variable.data[0, 1], 8)
        self.assertAlmostEqual(6.2, variable.data[1, 1], 8)

    def test_get_insitu_sensor_animal(self):
        dataset = self._create_dataset_with_variable("animal-sst_bla")

        insitu_sensor = MmdReader._get_insitu_sensor(dataset)
        self.assertEqual("animal-sst", insitu_sensor)

    def test_get_insitu_sensor_radiometer(self):
        dataset = self._create_dataset_with_variable("radiometer-sst_blubb")

        insitu_sensor = MmdReader._get_insitu_sensor(dataset)
        self.assertEqual("radiometer-sst", insitu_sensor)

    def test_get_insitu_sensor_unsupported(self):
        dataset = self._create_dataset_with_variable("unsupported-sensor")

        with self.assertRaises(IOError):
            MmdReader._get_insitu_sensor(dataset)

    def _create_dataset_with_variable(self, variable_name):
        dataset = xr.Dataset()
        array = DefaultData.create_default_array(2, 2, np.int16)
        variable = Variable(["y", "x"], array)
        dataset[variable_name] = variable
        return dataset
