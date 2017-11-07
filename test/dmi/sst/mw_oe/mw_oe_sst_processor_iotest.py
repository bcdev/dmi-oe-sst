import os
import unittest

from dmi.sst.mw_oe.mw_oe_sst_processor import MwOeSstProcessor
from test.dmi.test_data_utils import TestDataUtils


class MwOeSstProcessorIoTest(unittest.TestCase):
    def test_full_integration(self):
        test_mmd = TestDataUtils.get_test_mmd()
        self.assertTrue(os.path.isfile(test_mmd))

        output_dir = TestDataUtils.get_output_dir()

        processor = MwOeSstProcessor()
        processor.run(["-o", output_dir, test_mmd])

        expected_target_file = os.path.join(output_dir, "mmd6c_sst_ship-sst_amsre-aq_2010-272_2010-273_oe-sst.nc")
        self.assertTrue(os.path.isfile(expected_target_file))

        os.remove(expected_target_file)
