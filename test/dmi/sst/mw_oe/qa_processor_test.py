import unittest

import numpy as np
import xarray as xr
from xarray import Variable

from dmi.sst.mw_oe.flag_coding import FlagCoding
from dmi.sst.mw_oe.qa_processor import QaProcessor
from dmi.sst.util.default_data import DefaultData

NUM_MATCHES = 3


class QaProcessorTest(unittest.TestCase):
    dataset = None
    qa_processor = None
    flag_coding = None

    def setUp(self):
        self.dataset = xr.Dataset()

        self.qa_processor = QaProcessor()
        self.flag_coding = FlagCoding(NUM_MATCHES)

    def test_run_qa_general_pixel_data_quality_pass(self):
        data = DefaultData.create_default_vector(NUM_MATCHES, np.int16, fill_value=0)
        self.dataset["amsre.pixel_data_quality10H"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset, self.flag_coding)

        flags = self.flag_coding.get_flags()
        self.assertEqual(0, flags[0])
        self.assertEqual(0, flags[1])
        self.assertEqual(0, flags[2])

    def test_run_qa_general_pixel_data_quality_fail(self):
        data = DefaultData.create_default_vector(3, np.int16, fill_value=0)
        data[1] = 2
        self.dataset["amsre.pixel_data_quality10V"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset, self.flag_coding)

        flags = self.flag_coding.get_flags()
        self.assertEqual(0, flags[0])
        self.assertEqual(2, flags[1])
        self.assertEqual(0, flags[2])

    def test_run_qa_general_pixel_flag_fail(self):
        data = DefaultData.create_default_vector(3, np.int16, fill_value=0)
        data[2] = 1
        self.dataset["insitu.sst_track_flag"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset, self.flag_coding)

        flags = self.flag_coding.get_flags()
        self.assertEqual(0, flags[0])
        self.assertEqual(0, flags[1])
        self.assertEqual(2, flags[2])

    def test_run_qa_general_brightness_temperature_pass(self):
        data = DefaultData.create_default_vector(3, np.float32, fill_value=0)
        data[0] = 32.8
        data[1] = 33.9
        data[2] = 35.0
        self.dataset["amsre.brightness_temperature18V"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset, self.flag_coding)

        flags = self.flag_coding.get_flags()
        self.assertEqual(0, flags[0])
        self.assertEqual(0, flags[1])
        self.assertEqual(0, flags[2])

    def test_run_qa_general_brightness_temperature_fail(self):
        data = DefaultData.create_default_vector(3, np.float32, fill_value=0)
        data[0] = -1.8
        data[1] = 33.9
        data[2] = 498.55
        self.dataset["amsre.amsre.brightness_temperature36V"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset, self.flag_coding)

        flags = self.flag_coding.get_flags()
        self.assertEqual(4, flags[0])
        self.assertEqual(0, flags[1])
        self.assertEqual(4, flags[2])

    def test_run_qa_general_longitude_pass(self):
        data = DefaultData.create_default_vector(3, np.float32, fill_value=0)
        data[0] = 12.9
        data[1] = -14.7
        data[2] = 17.445
        self.dataset["amsre.longitude"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset, self.flag_coding)

        flags = self.flag_coding.get_flags()
        self.assertEqual(0, flags[0])
        self.assertEqual(0, flags[1])
        self.assertEqual(0, flags[2])

    def test_run_qa_general_longitude_fail(self):
        data = DefaultData.create_default_vector(3, np.float32, fill_value=0)
        data[0] = 212.9
        data[1] = -214.7
        data[2] = 17.445
        self.dataset["amsre.longitude"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset, self.flag_coding)

        flags = self.flag_coding.get_flags()
        self.assertEqual(16, flags[0])
        self.assertEqual(16, flags[1])
        self.assertEqual(0, flags[2])

    def test_run_qa_general_latitude_pass(self):
        data = DefaultData.create_default_vector(3, np.float32, fill_value=0)
        data[0] = 12.9
        data[1] = -14.7
        data[2] = 17.445
        self.dataset["amsre.latitude"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset, self.flag_coding)

        flags = self.flag_coding.get_flags()
        self.assertEqual(0, flags[0])
        self.assertEqual(0, flags[1])
        self.assertEqual(0, flags[2])

    def test_run_qa_general_latitude_fail(self):
        data = DefaultData.create_default_vector(3, np.float32, fill_value=0)
        data[0] = 12.9
        data[1] = -114.7
        data[2] = 90.445
        self.dataset["amsre.latitude"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset, self.flag_coding)

        flags = self.flag_coding.get_flags()
        self.assertEqual(0, flags[0])
        self.assertEqual(16, flags[1])
        self.assertEqual(16, flags[2])

    def test_run_qa_general_SZA_pass(self):
        data = DefaultData.create_default_vector(3, np.float32, fill_value=0)
        data[0] = 12.9
        data[1] = 14.7
        data[2] = 17.445
        self.dataset["amsre.solar_zenith_angle"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset, self.flag_coding)

        flags = self.flag_coding.get_flags()
        self.assertEqual(0, flags[0])
        self.assertEqual(0, flags[1])
        self.assertEqual(0, flags[2])

    def test_run_qa_general_SZA_fail(self):
        data = DefaultData.create_default_vector(3, np.float32, fill_value=0)
        data[0] = 181.9
        data[1] = -14.7
        data[2] = 17.445
        self.dataset["amsre.solar_zenith_angle"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset, self.flag_coding)

        flags = self.flag_coding.get_flags()
        self.assertEqual(32, flags[0])
        self.assertEqual(32, flags[1])
        self.assertEqual(0, flags[2])

    def test_run_qa_general_avg_wind_speed_pass(self):
        data = DefaultData.create_default_vector(3, np.float32, fill_value=0)
        data[0] = 12.9
        data[1] = 0.89
        data[2] = 5.485
        self.dataset["amsre.nwp.abs_wind_speed"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset, self.flag_coding)

        flags = self.flag_coding.get_flags()
        self.assertEqual(0, flags[0])
        self.assertEqual(0, flags[1])
        self.assertEqual(0, flags[2])

    def test_run_qa_general_avg_wind_speed_fail(self):
        data = DefaultData.create_default_vector(3, np.float32, fill_value=0)
        data[0] = 22.9
        data[1] = -0.89
        data[2] = 5.485
        self.dataset["amsre.nwp.abs_wind_speed"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset, self.flag_coding)

        flags = self.flag_coding.get_flags()
        self.assertEqual(8, flags[0])
        self.assertEqual(8, flags[1])
        self.assertEqual(0, flags[2])

    def test_run_qa_general_sst_pass(self):
        data = DefaultData.create_default_vector(3, np.float32, fill_value=0)
        data[0] = 13.9
        data[1] = -0.89
        data[2] = 15.485
        self.dataset["amsre.nwp.sea_surface_temperature"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset, self.flag_coding)

        flags = self.flag_coding.get_flags()
        self.assertEqual(0, flags[0])
        self.assertEqual(0, flags[1])
        self.assertEqual(0, flags[2])

    def test_run_qa_general_sst_fail(self):
        data = DefaultData.create_default_vector(3, np.float32, fill_value=0)
        data[0] = 13.9
        data[1] = -2.01
        data[2] = 40.485
        self.dataset["insitu.sea_surface_temperature"] = Variable(["matchup_count"], data)

        self.qa_processor.run_qa_general(self.dataset, self.flag_coding)

        flags = self.flag_coding.get_flags()
        self.assertEqual(0, flags[0])
        self.assertEqual(64, flags[1])
        self.assertEqual(64, flags[2])

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

        self.qa_processor.run_qa_bt_delta(self.dataset, self.flag_coding)

        flags = self.flag_coding.get_flags()
        self.assertEqual(0, flags[0])
        self.assertEqual(0, flags[1])
        self.assertEqual(0, flags[2])

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

        self.qa_processor.run_qa_bt_delta(self.dataset, self.flag_coding)

        flags = self.flag_coding.get_flags()
        self.assertEqual(128, flags[0])
        self.assertEqual(0, flags[1])
        self.assertEqual(128, flags[2])

    def test_run_rfi_detection_pass(self):
        sat_lon = [129.6, -172.8, -81.4]
        self.dataset["amsre.longitude"] = Variable(["matchup_count"], sat_lon)

        sat_lat = [25.1, 35.4, -19.7]
        self.dataset["amsre.latitude"] = Variable(["matchup_count"], sat_lat)

        geo_refl_lon = [76.0, -112.3, -113.0]
        self.dataset["amsre.Geostationary_Reflection_Longitude"] = Variable(["matchup_count"], geo_refl_lon)

        geo_refl_lat = [24.1, 78.4, -62.5]
        self.dataset["amsre.Geostationary_Reflection_Latitude"] = Variable(["matchup_count"], geo_refl_lat)

        i_asc = [1, 1, 0]
        self.dataset["amsre.ascending"] = Variable(["matchup_count"], i_asc)

        self.qa_processor.run_qa_rfi_detection(self.dataset, self.flag_coding)

        flags = self.flag_coding.get_flags()
        self.assertEqual(0, flags[0])
        self.assertEqual(0, flags[1])
        self.assertEqual(0, flags[2])

    def test_run_rfi_detection_fail(self):
        sat_lon = [172.4, 172.4, 6.0]
        self.dataset["amsre.longitude"] = Variable(["matchup_count"], sat_lon)

        sat_lat = [-16.1, -16.1, 64.0]
        self.dataset["amsre.latitude"] = Variable(["matchup_count"], sat_lat)

        geo_refl_lon = [-171.3, 236.0, -171.3]
        self.dataset["amsre.Geostationary_Reflection_Longitude"] = Variable(["matchup_count"], geo_refl_lon)

        geo_refl_lat = [29.4, -20.3, 29.4]
        self.dataset["amsre.Geostationary_Reflection_Latitude"] = Variable(["matchup_count"], geo_refl_lat)

        i_asc = [1, 0, 0]
        self.dataset["amsre.ascending"] = Variable(["matchup_count"], i_asc)

        self.qa_processor.run_qa_rfi_detection(self.dataset, self.flag_coding)

        flags = self.flag_coding.get_flags()
        self.assertEqual(0, flags[0])
        self.assertEqual(512, flags[1])
        self.assertEqual(512, flags[2])