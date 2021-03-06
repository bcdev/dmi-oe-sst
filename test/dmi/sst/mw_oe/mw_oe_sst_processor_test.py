import unittest

import numpy as np

from dmi.sst.mw_oe.mw_oe_sst_processor import MwOeSstProcessor


class MwOeMMDProcessorTest(unittest.TestCase):

    def test_create_result_structure(self):
        result = MwOeSstProcessor._create_result_structure(22, 5, 10)
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

        i = result.variables["i"]
        self.assertEqual((22,), i.shape)
        self.assertEqual(255, i.data[5])
        self.assertEqual(255, i.attrs["_FillValue"])
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

        TA0_ite0 = result.variables["TA0_ite0"]
        self.assertEqual((22, 10), TA0_ite0.shape)
        self.assertTrue(np.isnan(TA0_ite0.data[5, 6]))
        self.assertTrue(np.isnan(TA0_ite0.attrs["_FillValue"]))
        # @todo tb/tb request description from DMI and add 2017-10-25

        j_ite0 = result.variables["j_ite0"]
        self.assertEqual((22,), j_ite0.shape)
        self.assertTrue(np.isnan(j_ite0.data[6]))
        self.assertTrue(np.isnan(j_ite0.attrs["_FillValue"]))
        # @todo tb/tb request description from DMI and add 2017-10-25

        A = result.variables["A"]
        self.assertEqual((22, 4), A.shape)
        self.assertTrue(np.isnan(A.data[7, 0]))
        self.assertTrue(np.isnan(A.attrs["_FillValue"]))
        # @todo tb/tb request description from DMI and add 2017-10-25

        chisq = result.variables["chisq"]
        self.assertEqual((22,), chisq.shape)
        self.assertTrue(np.isnan(chisq.data[7]))
        self.assertTrue(np.isnan(chisq.attrs["_FillValue"]))
        # @todo tb/tb request description from DMI and add 2017-10-25

        mu_sst = result.variables["mu_sst"]
        self.assertEqual((22,), mu_sst.shape)
        self.assertTrue(np.isnan(mu_sst.data[8]))
        self.assertTrue(np.isnan(mu_sst.attrs["_FillValue"]))
        # @todo tb/tb request description from DMI and add 2017-10-25

        x = result.variables["x"]
        self.assertEqual((22, 4), x.shape)
        self.assertTrue(np.isnan(x.data[8, 1]))
        self.assertTrue(np.isnan(x.attrs["_FillValue"]))
        # @todo tb/tb request description from DMI and add 2017-10-25

        p0 = result.variables["p0"]
        self.assertEqual((22, 4), p0.shape)
        self.assertTrue(np.isnan(p0.data[8, 1]))
        self.assertTrue(np.isnan(p0.attrs["_FillValue"]))
        # @todo tb/tb request description from DMI and add 2017-10-25

        S = result.variables["S"]
        self.assertEqual((22, 4), S.shape)
        self.assertTrue(np.isnan(S.data[9, 2]))
        self.assertTrue(np.isnan(S.attrs["_FillValue"]))
        # @todo tb/tb request description from DMI and add 2017-10-25

        F = result.variables["F"]
        self.assertEqual((22, 10), F.shape)
        self.assertTrue(np.isnan(F.data[6, 7]))
        self.assertTrue(np.isnan(F.attrs["_FillValue"]))
        # @todo tb/tb request description from DMI and add 2017-10-25

        y = result.variables["y"]
        self.assertEqual((22, 10), y.shape)
        self.assertTrue(np.isnan(y.data[6, 7]))
        self.assertTrue(np.isnan(y.attrs["_FillValue"]))
        # @todo tb/tb request description from DMI and add 2018-05-24

        dtb = result.variables["dtb"]
        self.assertEqual((22, 10), dtb.shape)
        self.assertTrue(np.isnan(dtb.data[7, 8]))
        self.assertTrue(np.isnan(dtb.attrs["_FillValue"]))
        # @todo tb/tb request description from DMI and add 2017-10-25

        ds = result.variables["ds"]
        self.assertEqual((22,), ds.shape)
        self.assertTrue(np.isnan(ds.data[9]))
        self.assertTrue(np.isnan(ds.attrs["_FillValue"]))
        # @todo tb/tb request description from DMI and add 2017-10-25

        dn = result.variables["dn"]
        self.assertEqual((22,), dn.shape)
        self.assertTrue(np.isnan(dn.data[10]))
        self.assertTrue(np.isnan(dn.attrs["_FillValue"]))
        # @todo tb/tb request description from DMI and add 2017-10-25

        K4 = result.variables["K4"]
        self.assertEqual((22, 10), K4.shape)
        self.assertTrue(np.isnan(K4.data[8, 9]))
        self.assertTrue(np.isnan(K4.attrs["_FillValue"]))
        # @todo tb/tb request description from DMI and add 2017-10-25

        ite_index = result.variables["ite_index"]
        self.assertEqual((22,), ite_index.shape)
        self.assertEqual(255, ite_index.data[5])
        self.assertEqual(255, ite_index.attrs["_FillValue"])  # @todo tb/tb request description from DMI and add 2017-10-25

    def test_create_target_file_name(self):
        target_file_name = MwOeSstProcessor._create_target_file_name("mmd6c_sst_ship-sst_amsre-aq_2010-272_2010-273.nc")
        self.assertEqual("mmd6c_sst_ship-sst_amsre-aq_2010-272_2010-273_oe-sst.nc", target_file_name)
