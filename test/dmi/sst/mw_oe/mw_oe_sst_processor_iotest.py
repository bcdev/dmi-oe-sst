import os
import unittest

import xarray as xr

from dmi.sst.mw_oe.mw_oe_sst_processor import MwOeSstProcessor
from test.dmi.test_data_utils import TestDataUtils


class MwOeSstProcessorIoTest(unittest.TestCase):
    test_mmd = None
    output_dir = None
    expected_target_file = None

    def setUp(self):
        self.test_mmd = TestDataUtils.get_test_mmd()
        self.assertTrue(os.path.isfile(self.test_mmd))

        self.output_dir = TestDataUtils.get_output_dir()
        self.expected_target_file = os.path.join(self.output_dir, "mmd6c_sst_ship-sst_amsre-aq_2010-272_2010-273_oe-sst.nc")

    def tearDown(self):
        os.remove(self.expected_target_file)

    def test_full_integration(self):
        processor = MwOeSstProcessor()
        processor.run(["-o", self.output_dir, self.test_mmd])

        self.assertTrue(os.path.isfile(self.expected_target_file))

        # xarray can not handle the TAI 1993 time coding @todo 3 tb/th adapt if possible
        target_data = xr.open_dataset(self.expected_target_file, decode_times=False)
        try:
            self.assertIsNotNone(target_data["j"])
            self.assertIsNotNone(target_data["tb_rmse_ite"])
            self.assertIsNotNone(target_data["tb_rmse_ite0"])
            self.assertIsNotNone(target_data["tb_chi_ite"])
            self.assertIsNotNone(target_data["convergence_passed_flag"])
            self.assertIsNotNone(target_data["convergence_passed_idx"])
            self.assertIsNotNone(target_data["di2"])
            self.assertIsNotNone(target_data["dtb_ite0"])
            self.assertIsNotNone(target_data["TAO_ite0"])
            self.assertIsNotNone(target_data["j_ite0"])
            self.assertIsNotNone(target_data["AK"])
            self.assertIsNotNone(target_data["chisq"])
            self.assertIsNotNone(target_data["tb_rmse"])
            self.assertIsNotNone(target_data["p"])
            self.assertIsNotNone(target_data["S"])
            self.assertIsNotNone(target_data["tb_sim"])
            self.assertIsNotNone(target_data["dtb"])
            self.assertIsNotNone(target_data["ds"])
            self.assertIsNotNone(target_data["dn"])
            self.assertIsNotNone(target_data["K4"])
            self.assertIsNotNone(target_data["ite_index"])

            variable = target_data["flags"]
            self.assertEqual(0, variable.data[0])
            self.assertEqual(2, variable.data[114])
            self.assertEqual("1 2 4 8 16 32 64 128 256", variable.attrs["flag_masks"])
            self.assertEqual("avg_inv_thresh amsre_flag bt_out_of_range ws_out_of_range inv_geolocation sza_out_of_range sst_out_of_range bt_pol_test_failed inv_file_name", variable.attrs["flag_meanings"])
        finally:
            target_data.close()
