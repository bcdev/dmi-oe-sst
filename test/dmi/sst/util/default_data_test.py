import unittest

import numpy as np

from dmi.sst.util.default_data import DefaultData


class DefaultDataTest(unittest.TestCase):

    def test_create_default_array(self):
        array = DefaultData.create_default_array(5, 4, np.float32, fill_value=np.NaN)
        self.assertEqual((4, 5), array.shape)
        self.assertTrue(np.isnan(array.data[0, 1]))

    def test_get_default_fill_value(self):
        self.assertEqual(-127, DefaultData.get_default_fill_value(np.int8))
        self.assertEqual(255, DefaultData.get_default_fill_value(np.uint8))
        self.assertEqual(-32767, DefaultData.get_default_fill_value(np.int16))
        self.assertEqual(np.uint16(-1), DefaultData.get_default_fill_value(np.uint16))
        self.assertEqual(-2147483647, DefaultData.get_default_fill_value(np.int32))
        self.assertEqual(4294967295, DefaultData.get_default_fill_value(np.uint32))
        self.assertEqual(-9223372036854775806, DefaultData.get_default_fill_value(np.int64))
        self.assertEqual(np.float32(9.96921E36), DefaultData.get_default_fill_value(np.float32))
        self.assertEqual(9.969209968386869E36, DefaultData.get_default_fill_value(np.float64))

