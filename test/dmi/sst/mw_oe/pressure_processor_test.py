import unittest

import numpy as np

from dmi.sst.mw_oe.pressure_processor import PressureProcessor


class PreprocessorTest(unittest.TestCase):
    def test_calculate_pressure_levels(self):
        sea_level_pressure = np.array([101592.46875, 101585.5859375, 101407.1953125])

        processor = PressureProcessor()
        pressure_levels = processor.calculate_pressure_levels(sea_level_pressure)
        self.assertEqual((3, 60), pressure_levels.shape)
        self.assertAlmostEqual(20.0, pressure_levels.data[0, 0], 8)
        self.assertAlmostEqual(261.6398925781250, pressure_levels.data[0, 14], 8)
        self.assertAlmostEqual(2408.3515625, pressure_levels.data[0, 28], 8)
        self.assertAlmostEqual(3582.78125, pressure_levels.data[0, 35], 8)
        self.assertAlmostEqual(3807.00390625, pressure_levels.data[0, 42], 8)
        self.assertAlmostEqual(2578.8515625, pressure_levels.data[0, 49], 8)
        self.assertAlmostEqual(240.7734375, pressure_levels.data[0, 59], 8)

        self.assertAlmostEqual(20.0, pressure_levels.data[1, 0], 8)
        self.assertAlmostEqual(261.6398925781250, pressure_levels.data[1, 14], 8)
        self.assertAlmostEqual(2408.28906250, pressure_levels.data[1, 28], 8)
        self.assertAlmostEqual(3582.5234375, pressure_levels.data[1, 35], 8)
        self.assertAlmostEqual(3806.65234375, pressure_levels.data[1, 42], 8)
        self.assertAlmostEqual(2578.609375, pressure_levels.data[1, 49], 8)
        self.assertAlmostEqual(240.7578125, pressure_levels.data[1, 59], 8)

        self.assertAlmostEqual(20.0, pressure_levels.data[2, 0], 8)
        self.assertAlmostEqual(261.6398925781250, pressure_levels.data[2, 14], 8)
        self.assertAlmostEqual(2406.587890625, pressure_levels.data[2, 28], 8)
        self.assertAlmostEqual(3575.7265625, pressure_levels.data[2, 35], 8)
        self.assertAlmostEqual(3797.5703125, pressure_levels.data[2, 42], 8)
        self.assertAlmostEqual(2572.1640625, pressure_levels.data[2, 49], 8)
        self.assertAlmostEqual(240.3359375, pressure_levels.data[2, 59], 8)
