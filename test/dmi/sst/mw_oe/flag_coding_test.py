import unittest

import numpy as np

from dmi.sst.mw_oe.flag_coding import FlagCoding

NUM_SAMPLES = 34


class FlagCodingTest(unittest.TestCase):

    flag_coding = None

    def setUp(self):
        self.flag_coding = FlagCoding(NUM_SAMPLES)

    def test_get_flag_masks(self):
        masks = FlagCoding.get_flag_masks()
        self.assertEqual("1 2 4 8 16 32 64 128", masks)

    def test_get_flag_meanings(self):
        masks = FlagCoding.get_flag_meanings()
        self.assertEqual("avg_inv_thresh amsre_flag bt_out_of_range ws_out_of_range inv_geolocation sza_out_of_range sst_out_of_range bt_pol_test_failed", masks)

    def test_get_flags_initial(self):
        flags = self.flag_coding.get_flags()
        self.assertIsNotNone(flags)

        for i in range(0, NUM_SAMPLES):
            self.assertEqual(0, flags[i])

    def test_add_avg_inv_thresh(self):
        tags = np.zeros(NUM_SAMPLES, dtype=np.bool)
        tags[5] = True
        tags[19] = True

        self.flag_coding.add_avg_inv_thresh(tags)

        flags = self.flag_coding.get_flags()
        self.assertEqual(0, flags[0])
        self.assertEqual(0, flags[1])
        self.assertEqual(1, flags[5])
        self.assertEqual(1, flags[19])

    def test_add_amsre_flag(self):
        tags = np.zeros(NUM_SAMPLES, dtype=np.bool)
        tags[6] = True
        tags[20] = True

        self.flag_coding.add_amsre_flag(tags)

        flags = self.flag_coding.get_flags()
        self.assertEqual(0, flags[1])
        self.assertEqual(0, flags[2])
        self.assertEqual(2, flags[6])
        self.assertEqual(2, flags[20])

    def test_add_bt_out_of_range(self):
        tags = np.zeros(NUM_SAMPLES, dtype=np.bool)
        tags[7] = True
        tags[21] = True

        self.flag_coding.add_bt_out_of_range(tags)

        flags = self.flag_coding.get_flags()
        self.assertEqual(0, flags[2])
        self.assertEqual(0, flags[3])
        self.assertEqual(4, flags[7])
        self.assertEqual(4, flags[21])

    def test_add_ws_out_of_range(self):
        tags = np.zeros(NUM_SAMPLES, dtype=np.bool)
        tags[8] = True
        tags[22] = True

        self.flag_coding.add_ws_out_of_range(tags)

        flags = self.flag_coding.get_flags()
        self.assertEqual(0, flags[3])
        self.assertEqual(0, flags[4])
        self.assertEqual(8, flags[8])
        self.assertEqual(8, flags[22])

    def test_add_invald_geolocation(self):
        tags = np.zeros(NUM_SAMPLES, dtype=np.bool)
        tags[9] = True
        tags[23] = True

        self.flag_coding.add_invalid_geolocation(tags)

        flags = self.flag_coding.get_flags()
        self.assertEqual(0, flags[4])
        self.assertEqual(0, flags[5])
        self.assertEqual(16, flags[9])
        self.assertEqual(16, flags[23])

    def test_add_sza_out_of_range(self):
        tags = np.zeros(NUM_SAMPLES, dtype=np.bool)
        tags[10] = True
        tags[24] = True

        self.flag_coding.add_sza_out_of_range(tags)

        flags = self.flag_coding.get_flags()
        self.assertEqual(0, flags[4])
        self.assertEqual(0, flags[5])
        self.assertEqual(32, flags[10])
        self.assertEqual(32, flags[24])

    def test_add_sst_out_of_range(self):
        tags = np.zeros(NUM_SAMPLES, dtype=np.bool)
        tags[11] = True
        tags[25] = True

        self.flag_coding.add_sst_out_of_range(tags)

        flags = self.flag_coding.get_flags()
        self.assertEqual(0, flags[5])
        self.assertEqual(0, flags[6])
        self.assertEqual(64, flags[11])
        self.assertEqual(64, flags[25])

    def test_add_bt_pol_test_failed(self):
        tags = np.zeros(NUM_SAMPLES, dtype=np.bool)
        tags[12] = True
        tags[26] = True

        self.flag_coding.add_bt_pol_test_failed(tags)

        flags = self.flag_coding.get_flags()
        self.assertEqual(0, flags[6])
        self.assertEqual(0, flags[7])
        self.assertEqual(128, flags[12])
        self.assertEqual(128, flags[26])
