import unittest

import numpy as np

from dmi.sst.mw_oe.mw_oe_mmd_processor import MwOeMMDProcessor


class MwOeMMDProcessorTest(unittest.TestCase):
    def test_create_result_structure(self):
        result = MwOeMMDProcessor.create_result_structure(22, 5, 10)
        self.assertIsNotNone(result)

        j = result.variables["j"]
        self.assertEqual((22, 5), j.shape)
        self.assertTrue(np.isnan(j.data[0, 1]))
        self.assertTrue(np.isnan(j.attrs["_FillValue"]))
        # @todo tb/tb request description from DMI and add 2017-10-25

        tb_rmse_ite = result.variables["tb_rmse_ite"]
        self.assertEqual((22, 5), tb_rmse_ite.shape)
        self.assertTrue(np.isnan(tb_rmse_ite.data[1, 2]))
        self.assertTrue(np.isnan(tb_rmse_ite.attrs["_FillValue"]))
        # @todo tb/tb request description from DMI and add 2017-10-25

        tb_rmse_ite0 = result.variables["tb_rmse_ite0"]
        self.assertEqual((22,), tb_rmse_ite0.shape)
        self.assertTrue(np.isnan(tb_rmse_ite0.data[5]))
        self.assertTrue(np.isnan(tb_rmse_ite0.attrs["_FillValue"]))
        # @todo tb/tb request description from DMI and add 2017-10-25

        tb_chi_ite = result.variables["tb_chi_ite"]
        self.assertEqual((22, 5), tb_chi_ite.shape)
        self.assertTrue(np.isnan(tb_chi_ite.data[2, 3]))
        self.assertTrue(np.isnan(tb_chi_ite.attrs["_FillValue"]))
        # @todo tb/tb request description from DMI and add 2017-10-25

        conv_pass_flag = result.variables["convergence_passed_flag"]
        self.assertEqual((22,), conv_pass_flag.shape)
        self.assertEqual(255, conv_pass_flag.data[4])
        self.assertEqual(255, conv_pass_flag.attrs["_FillValue"])
        # @todo tb/tb request description from DMI and add 2017-10-25

        conv_pass_idx = result.variables["convergence_passed_idx"]
        self.assertEqual((22,), conv_pass_idx.shape)
        self.assertEqual(255, conv_pass_idx.data[5])
        self.assertEqual(255, conv_pass_idx.attrs["_FillValue"])
        # @todo tb/tb request description from DMI and add 2017-10-25

        di2 = result.variables["di2"]
        self.assertEqual((22, 5), di2.shape)
        self.assertTrue(np.isnan(di2.data[3, 4]))
        self.assertTrue(np.isnan(di2.attrs["_FillValue"]))
        # @todo tb/tb request description from DMI and add 2017-10-25

        dtb_ite0 = result.variables["dtb_ite0"]
        self.assertEqual((22, 10), dtb_ite0.shape)
        self.assertTrue(np.isnan(dtb_ite0.data[4, 5]))
        self.assertTrue(np.isnan(dtb_ite0.attrs["_FillValue"]))
        # @todo tb/tb request description from DMI and add 2017-10-25

        TAO_ite0 = result.variables["TAO_ite0"]
        self.assertEqual((22, 10), TAO_ite0.shape)
        self.assertTrue(np.isnan(TAO_ite0.data[5, 6]))
        self.assertTrue(np.isnan(TAO_ite0.attrs["_FillValue"]))
        # @todo tb/tb request description from DMI and add 2017-10-25

        j_ite0 = result.variables["j_ite0"]
        self.assertEqual((22,), j_ite0.shape)
        self.assertTrue(np.isnan(j_ite0.data[6]))
        self.assertTrue(np.isnan(j_ite0.attrs["_FillValue"]))
        # @todo tb/tb request description from DMI and add 2017-10-25
