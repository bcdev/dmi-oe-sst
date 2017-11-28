import unittest

import numpy as np
import xarray as xr
from xarray import Variable

from dmi.sst.mw_oe.qa_processor import QaProcessor
from dmi.sst.util.default_data import DefaultData

NUM_MATCHES = 3


class QaProcessorTest(unittest.TestCase):

    dataset = None
    qa_processor = None

    def setUp(self):
        self.dataset = xr.Dataset()

        invalid_data_array = np.zeros(NUM_MATCHES, dtype=np.bool)
        self.dataset["invalid_data"] = Variable(["matchup_count"], invalid_data_array)

        self.qa_processor = QaProcessor()

    def test_run_qa_general_pixel_data_quality_pass(self):
        data = DefaultData.create_default_vector(NUM_MATCHES, np.int16, fill_value=0)
        self.dataset["amsre.pixel_data_quality10H"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset)

        self.assertFalse(self.dataset["invalid_data"].data[0])
        self.assertFalse(self.dataset["invalid_data"].data[1])
        self.assertFalse(self.dataset["invalid_data"].data[2])

    def test_run_qa_general_pixel_data_quality_fail(self):
        data = DefaultData.create_default_vector(3, np.int16, fill_value=0)
        data[1] = 2
        self.dataset["amsre.pixel_data_quality10V"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset)

        self.assertFalse(self.dataset["invalid_data"].data[0])
        self.assertTrue(self.dataset["invalid_data"].data[1])
        self.assertFalse(self.dataset["invalid_data"].data[2])

    def test_run_qa_general_pixel_flag_fail(self):
        data = DefaultData.create_default_vector(3, np.int16, fill_value=0)
        data[2] = 1
        self.dataset["insitu.sst_track_flag"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset)

        self.assertFalse(self.dataset["invalid_data"].data[0])
        self.assertFalse(self.dataset["invalid_data"].data[1])
        self.assertTrue(self.dataset["invalid_data"].data[2])

    def test_run_qa_general_brightness_temperature_pass(self):
        data = DefaultData.create_default_vector(3, np.float32, fill_value=0)
        data[0] = 32.8
        data[1] = 33.9
        data[2] = 35.0
        self.dataset["amsre.brightness_temperature18V"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset)

        self.assertFalse(self.dataset["invalid_data"].data[0])
        self.assertFalse(self.dataset["invalid_data"].data[1])
        self.assertFalse(self.dataset["invalid_data"].data[2])

    def test_run_qa_general_brightness_temperature_fail(self):
        data = DefaultData.create_default_vector(3, np.float32, fill_value=0)
        data[0] = -1.8
        data[1] = 33.9
        data[2] = 498.55
        self.dataset["amsre.amsre.brightness_temperature36V"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset)

        self.assertTrue(self.dataset["invalid_data"].data[0])
        self.assertFalse(self.dataset["invalid_data"].data[1])
        self.assertTrue(self.dataset["invalid_data"].data[2])

    def test_run_qa_general_wind_speed_pass(self):
        data = DefaultData.create_default_vector(3, np.float32, fill_value=0)
        data[0] = 11.8
        data[1] = -12.7
        data[2] = 8.445
        self.dataset["amsre.nwp.10m_east_wind_component"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset)

        self.assertFalse(self.dataset["invalid_data"].data[0])
        self.assertFalse(self.dataset["invalid_data"].data[1])
        self.assertFalse(self.dataset["invalid_data"].data[2])

    def test_run_qa_general_wind_speed_fail(self):
        data = DefaultData.create_default_vector(3, np.float32, fill_value=0)
        data[0] = 11.8
        data[1] = -52.7
        data[2] = 108.445
        self.dataset["amsre.nwp.10m_north_wind_component"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset)

        self.assertFalse(self.dataset["invalid_data"].data[0])
        self.assertTrue(self.dataset["invalid_data"].data[1])
        self.assertTrue(self.dataset["invalid_data"].data[2])

    def test_run_qa_general_longitude_pass(self):
        data = DefaultData.create_default_vector(3, np.float32, fill_value=0)
        data[0] = 12.9
        data[1] = -14.7
        data[2] = 17.445
        self.dataset["amsre.longitude"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset)

        self.assertFalse(self.dataset["invalid_data"].data[0])
        self.assertFalse(self.dataset["invalid_data"].data[1])
        self.assertFalse(self.dataset["invalid_data"].data[2])

    def test_run_qa_general_longitude_fail(self):
        data = DefaultData.create_default_vector(3, np.float32, fill_value=0)
        data[0] = 212.9
        data[1] = -214.7
        data[2] = 17.445
        self.dataset["amsre.longitude"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset)

        self.assertTrue(self.dataset["invalid_data"].data[0])
        self.assertTrue(self.dataset["invalid_data"].data[1])
        self.assertFalse(self.dataset["invalid_data"].data[2])

    def test_run_qa_general_latitude_pass(self):
        data = DefaultData.create_default_vector(3, np.float32, fill_value=0)
        data[0] = 12.9
        data[1] = -14.7
        data[2] = 17.445
        self.dataset["amsre.latitude"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset)

        self.assertFalse(self.dataset["invalid_data"].data[0])
        self.assertFalse(self.dataset["invalid_data"].data[1])
        self.assertFalse(self.dataset["invalid_data"].data[2])

    def test_run_qa_general_latitude_fail(self):
        data = DefaultData.create_default_vector(3, np.float32, fill_value=0)
        data[0] = 12.9
        data[1] = -114.7
        data[2] = 90.445
        self.dataset["amsre.latitude"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset)

        self.assertFalse(self.dataset["invalid_data"].data[0])
        self.assertTrue(self.dataset["invalid_data"].data[1])
        self.assertTrue(self.dataset["invalid_data"].data[2])

    def test_run_qa_general_SZA_pass(self):
        data = DefaultData.create_default_vector(3, np.float32, fill_value=0)
        data[0] = 12.9
        data[1] = 14.7
        data[2] = 17.445
        self.dataset["amsre.solar_zenith_angle"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset)

        self.assertFalse(self.dataset["invalid_data"].data[0])
        self.assertFalse(self.dataset["invalid_data"].data[1])
        self.assertFalse(self.dataset["invalid_data"].data[2])

    def test_run_qa_general_SZA_fail(self):
        data = DefaultData.create_default_vector(3, np.float32, fill_value=0)
        data[0] = 181.9
        data[1] = -14.7
        data[2] = 17.445
        self.dataset["amsre.solar_zenith_angle"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset)

        self.assertTrue(self.dataset["invalid_data"].data[0])
        self.assertTrue(self.dataset["invalid_data"].data[1])
        self.assertFalse(self.dataset["invalid_data"].data[2])

    def test_run_qa_general_avg_wind_speed_pass(self):
        data = DefaultData.create_default_vector(3, np.float32, fill_value=0)
        data[0] = 12.9
        data[1] = 0.89
        data[2] = 5.485
        self.dataset["amsre.nwp.abs_wind_speed"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset)

        self.assertFalse(self.dataset["invalid_data"].data[0])
        self.assertFalse(self.dataset["invalid_data"].data[1])
        self.assertFalse(self.dataset["invalid_data"].data[2])

    def test_run_qa_general_avg_wind_speed_fail(self):
        data = DefaultData.create_default_vector(3, np.float32, fill_value=0)
        data[0] = 22.9
        data[1] = -0.89
        data[2] = 5.485
        self.dataset["amsre.nwp.abs_wind_speed"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset)

        self.assertTrue(self.dataset["invalid_data"].data[0])
        self.assertTrue(self.dataset["invalid_data"].data[1])
        self.assertFalse(self.dataset["invalid_data"].data[2])

    def test_run_qa_general_sst_pass(self):
        data = DefaultData.create_default_vector(3, np.float32, fill_value=0)
        data[0] = 13.9
        data[1] = -0.89
        data[2] = 15.485
        self.dataset["amsre.nwp.sea_surface_temperature"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset)

        self.assertFalse(self.dataset["invalid_data"].data[0])
        self.assertFalse(self.dataset["invalid_data"].data[1])
        self.assertFalse(self.dataset["invalid_data"].data[2])

    def test_run_qa_general_sst_fail(self):
        data = DefaultData.create_default_vector(3, np.float32, fill_value=0)
        data[0] = 13.9
        data[1] = -2.01
        data[2] = 40.485
        self.dataset["insitu.sea_surface_temperature"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset)

        self.assertFalse(self.dataset["invalid_data"].data[0])
        self.assertTrue(self.dataset["invalid_data"].data[1])
        self.assertTrue(self.dataset["invalid_data"].data[2])

    def test_run_bt_delta_pass(self):
        data = DefaultData.create_default_vector(3, np.float32, fill_value=0)
        data[0] = 10.9
        data[1] = 20.7
        data[2] = 30.445
        self.dataset["amsre.brightness_temperature18V"] = Variable(["matchup_count"], data)        
        self.dataset["amsre.brightness_temperature23V"] = Variable(["matchup_count"], data)        
        self.dataset["amsre.brightness_temperature36V"] = Variable(["matchup_count"], data)

        data = DefaultData.create_default_vector(3, np.float32, fill_value=0)
        data[0] = 10.0
        data[1] = 20.0
        data[2] = 30.0
        self.dataset["amsre.brightness_temperature18H"] = Variable(["matchup_count"], data)
        self.dataset["amsre.brightness_temperature23H"] = Variable(["matchup_count"], data)
        self.dataset["amsre.brightness_temperature36H"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_bt_delta(self.dataset)

        self.assertFalse(self.dataset["invalid_data"].data[0])
        self.assertFalse(self.dataset["invalid_data"].data[1])
        self.assertFalse(self.dataset["invalid_data"].data[2])

    def test_run_bt_delta_fail(self):
        data = DefaultData.create_default_vector(3, np.float32, fill_value=0)
        data[0] = 10.9
        data[1] = 20.7
        data[2] = 30.445
        self.dataset["amsre.brightness_temperature18V"] = Variable(["matchup_count"], data)
        self.dataset["amsre.brightness_temperature23V"] = Variable(["matchup_count"], data)
        self.dataset["amsre.brightness_temperature36V"] = Variable(["matchup_count"], data)

        data = DefaultData.create_default_vector(3, np.float32, fill_value=0)
        data[0] = 11.0
        data[1] = 20.0
        data[2] = 31.0
        self.dataset["amsre.brightness_temperature18H"] = Variable(["matchup_count"], data)
        self.dataset["amsre.brightness_temperature23H"] = Variable(["matchup_count"], data)
        self.dataset["amsre.brightness_temperature36H"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_bt_delta(self.dataset)

        self.assertTrue(self.dataset["invalid_data"].data[0])
        self.assertFalse(self.dataset["invalid_data"].data[1])
        self.assertTrue(self.dataset["invalid_data"].data[2])
