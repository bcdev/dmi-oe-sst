import unittest

import numpy as np

from dmi.sst.mw_oe.pressure_processor import PressureProcessor


class PreprocessorTest(unittest.TestCase):
    def test_calculate_pressure_levels(self):
        sea_level_pressure = np.float32([101592.46875, 101585.5859375, 101407.1953125])

        processor = PressureProcessor()
        pressure_levels = processor.calculate_pressure_levels(sea_level_pressure)
        self.assertEqual((3, 60), pressure_levels.shape)
        self.assertAlmostEqual(np.float64(20.0), pressure_levels.data[0, 0], 8)
        self.assertAlmostEqual(np.float64(261.6398925781250), pressure_levels.data[0, 14], 8)
        self.assertAlmostEqual(np.float64(2408.353), pressure_levels.data[0, 28], 4)
        self.assertAlmostEqual(np.float64(3582.7822), pressure_levels.data[0, 35], 4)
        self.assertAlmostEqual(np.float64(3807.012), pressure_levels.data[0, 42], 4)
        self.assertAlmostEqual(np.float64(2578.8582), pressure_levels.data[0, 49], 4)
        self.assertAlmostEqual(np.float64(240.77415), pressure_levels.data[0, 59], 5)

        self.assertAlmostEqual(np.float64(20.0), pressure_levels.data[1, 0], 8)
        self.assertAlmostEqual(np.float64(261.639892578125), pressure_levels.data[1, 14], 8)
        self.assertAlmostEqual(np.float64(2408.2876), pressure_levels.data[1, 28], 5)
        self.assertAlmostEqual(np.float64(3582.52), pressure_levels.data[1, 35], 4)
        self.assertAlmostEqual(np.float64(3806.6614), pressure_levels.data[1, 42], 4)
        self.assertAlmostEqual(np.float64(2578.6096), pressure_levels.data[1, 49], 4)
        self.assertAlmostEqual(np.float64(240.75784), pressure_levels.data[1, 59], 5)

        self.assertAlmostEqual(np.float64(20.0), pressure_levels.data[2, 0], 8)
        self.assertAlmostEqual(np.float64(261.639892578125), pressure_levels.data[2, 14], 8)
        self.assertAlmostEqual(np.float64(2406.5867), pressure_levels.data[2, 28], 4)
        self.assertAlmostEqual(np.float64(3575.7249), pressure_levels.data[2, 35], 4)
        self.assertAlmostEqual(np.float64(3797.5745), pressure_levels.data[2, 42], 4)
        self.assertAlmostEqual(np.float64(2572.1658), pressure_levels.data[2, 49], 4)
        self.assertAlmostEqual(np.float64(240.33505), pressure_levels.data[2, 59], 5)
