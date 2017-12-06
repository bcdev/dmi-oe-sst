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

    def test_add_fill_value_attributes_no_scaling_no_fill_value_int16(self):
        variable = self._create_variable(np.int16)
        MmdReader._add_fill_value_attributes(variable, "don't care")
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), variable.attrs["_FillValue"])

    def test_add_fill_value_attributes_no_scaling_no_fill_value_float32(self):
        variable = self._create_variable(np.float32)
        MmdReader._add_fill_value_attributes(variable, "don't care")
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), variable.attrs["_FillValue"])

    def test_add_fill_value_attributes_no_scaling_fill_value_untouched(self):
        variable = self._create_variable(np.int8)
        variable.attrs["_FillValue"] = -127
        MmdReader._add_fill_value_attributes(variable, "don't care")
        self.assertEqual(-127, variable.attrs["_FillValue"])

    def test_add_fill_value_attributes_scaling_brightness_temperature(self):
        variable = self._create_variable(np.int16)
        variable.attrs["OFFSET"] = 327.68
        variable.attrs["SCALE_FACTOR"] = 0.01
        MmdReader._add_fill_value_attributes(variable, "amsre.brightness_temperature23V")
        self.assertAlmostEqual(0.0, variable.attrs["_FillValue"], 8)

    def test_add_fill_value_attributes_scaling_no_fill_value_int16(self):
        variable = self._create_variable(np.int16)
        variable.attrs["OFFSET"] = 0.0
        variable.attrs["SCALE_FACTOR"] = 0.01
        MmdReader._add_fill_value_attributes(variable, "don't care")
        self.assertAlmostEqual(-327.67, variable.attrs["_FillValue"], 8)

    def test_add_fill_value_attributes_grib_data(self):
        variable = self._create_variable(np.int16)
        variable.attrs["source"] = "GRIB data"
        MmdReader._add_fill_value_attributes(variable, "don't care")
        self.assertAlmostEqual(2e20, variable.attrs["_FillValue"], 8)

    def test_add_fill_value_attributes_insitu_data(self):
        variable = self._create_variable(np.int16)
        MmdReader._add_fill_value_attributes(variable, "insitu.lon")
        self.assertAlmostEqual(-32768, variable.attrs["_FillValue"], 8)

    def _create_dataset_with_variable(self, variable_name):
        dataset = xr.Dataset()
        array = DefaultData.create_default_array(2, 2, np.int16)
        variable = Variable(["y", "x"], array)
        dataset[variable_name] = variable
        return dataset

    def _create_variable(self, data_type):
        array = DefaultData.create_default_array(2, 2, data_type)
        variable = Variable(["y", "x"], array)
        return variable
