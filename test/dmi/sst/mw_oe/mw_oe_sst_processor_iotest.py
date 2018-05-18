import os
import unittest

import numpy as np
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
            variable = target_data["j"]
            self.assertAlmostEqual(22.505741, variable.data[118, 0], 6)
            self.assertAlmostEqual(13.014329, variable.data[118, 1], 6)

            variable = target_data["tb_rmse_ite"]
            self.assertAlmostEqual(0.7479118, variable.data[119, 2], 7)
            self.assertTrue(np.isnan(variable.data[119, 3]))

            variable = target_data["tb_rmse_ite0"]
            self.assertAlmostEqual(4.736254, variable.data[120], 6)
            self.assertAlmostEqual(6.549006, variable.data[121], 7)

            variable = target_data["tb_chi_ite"]
            self.assertTrue(np.isnan(variable.data[122, 3]))
            self.assertTrue(np.isnan(variable.data[123, 4]))

            variable = target_data["convergence_passed_flag"]
            self.assertEqual(1, variable.data[124])
            self.assertEqual(1, variable.data[125])

            variable = target_data["convergence_passed_idx"]
            self.assertEqual(3, variable.data[126])
            self.assertEqual(3, variable.data[127])

            variable = target_data["di2"]
            self.assertTrue(np.isnan(variable.data[128, 5]))
            self.assertTrue(np.isnan(variable.data[129, 6]))

            variable = target_data["dtb_ite0"]
            self.assertAlmostEqual(17.065277, variable.data[130, 7], 6)
            self.assertAlmostEqual(-0.029170848, variable.data[131, 8], 8)

            variable = target_data["TA0_ite0"]
            self.assertAlmostEqual(158.90617, variable.data[132, 9], 5)
            self.assertAlmostEqual(157.06029, variable.data[133, 0], 5)

            variable = target_data["j_ite0"]
            self.assertAlmostEqual(9482.511, variable.data[134], 3)
            self.assertAlmostEqual(9604.835, variable.data[135], 4)

            variable = target_data["AK"]
            self.assertAlmostEqual(0.86243814, variable.data[136, 1], 8)
            self.assertAlmostEqual(0.99990416, variable.data[137, 2], 8)

            variable = target_data["chisq"]
            self.assertAlmostEqual(0.61924624, variable.data[138], 8)
            self.assertAlmostEqual(0.6914389, variable.data[139], 7)

            variable = target_data["tb_rmse"]
            self.assertAlmostEqual(0.30289936, variable.data[140], 8)
            self.assertAlmostEqual(0.30990422, variable.data[141], 8)

            variable = target_data["p"]
            self.assertTrue(np.isnan(variable.data[142, 3]))
            self.assertTrue(np.isnan(variable.data[143, 0]))

            variable = target_data["S"]
            self.assertTrue(np.isnan(variable.data[144, 1]))
            self.assertTrue(np.isnan(variable.data[145, 2]))

            variable = target_data["tb_sim"]
            self.assertTrue(np.isnan(variable.data[146, 4]))
            self.assertTrue(np.isnan(variable.data[147, 5]))

            variable = target_data["dtb"]
            self.assertAlmostEqual(-0.1574104, variable.data[148, 6], 8)
            self.assertTrue(np.isnan(variable.data[149, 7]))

            variable = target_data["ds"]
            self.assertTrue(np.isnan(variable.data[150]))
            self.assertTrue(np.isnan(variable.data[151]))

            variable = target_data["dn"]
            self.assertTrue(np.isnan(variable.data[152]))
            self.assertTrue(np.isnan(variable.data[153]))

            variable = target_data["K4"]
            self.assertTrue(np.isnan(variable.data[154, 8]))
            self.assertTrue(np.isnan(variable.data[155, 9]))

            variable = target_data["ite_index"]
            self.assertTrue(np.isnan(variable.data[156]))
            self.assertTrue(np.isnan(variable.data[157]))

            variable = target_data["flags"]
            self.assertEqual(4096, variable.data[0])
            self.assertEqual(514, variable.data[114])
            self.assertEqual(0, variable.data[118])
            self.assertEqual(512, variable.data[157])
            self.assertEqual("1,2,4,8,16,32,64,128,256,512,1024,2048,4096", variable.attrs["flag_masks"])
            self.assertEqual(
                "avg_inv_thresh amsre_flag bt_out_of_range ws_out_of_range inv_geolocation sza_out_of_range sst_out_of_range bt_pol_test_failed inv_file_name rfi_possible diurnal_warming rain_possible std_dev_too_high",
                variable.attrs["flag_meanings"])
        finally:
            target_data.close()
