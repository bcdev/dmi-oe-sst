import os
import unittest

from dmi.sst.mw_oe.mmd_reader import MmdReader
from test.dmi.test_data_utils import TestDataUtils

TEST_FILE_NAME = "mmd6c_sst_ship-sst_amsre-aq_2010-272_2010-273.nc"


class MmdReaderIoTest(unittest.TestCase):
    def test_read(self):
        data_dir = TestDataUtils.get_test_data_dir()
        test_file = os.path.join(data_dir, TEST_FILE_NAME)
        self.assertTrue(os.path.isfile(test_file))

        reader = MmdReader()
        input_data = reader.read(test_file)
        self.assertIsNotNone(input_data)

        self.assertIsNotNone(input_data.variables["amsre.latitude"])
        self.assertIsNotNone(input_data.variables["amsre.longitude"])
        self.assertIsNotNone(input_data.variables["amsre.time"])  # @todo 3 tb/tb check if this is ever used 2017-10-27
        self.assertIsNotNone(input_data.variables["amsre.solar_zenith_angle"])
        self.assertIsNotNone(input_data.variables["amsre.satellite_zenith_angle"])
        self.assertIsNotNone(input_data.variables["amsre.land_ocean_flag_6"])
        self.assertIsNotNone(input_data.variables["amsre.brightness_temperature6V"])
        self.assertIsNotNone(input_data.variables["amsre.brightness_temperature6H"])
        self.assertIsNotNone(input_data.variables["amsre.brightness_temperature10V"])
        self.assertIsNotNone(input_data.variables["amsre.brightness_temperature10H"])
        self.assertIsNotNone(input_data.variables["amsre.brightness_temperature18V"])

        self.assertIsNotNone(input_data.variables["amsre.nwp.seaice_fraction"])
        self.assertIsNotNone(input_data.variables["amsre.nwp.sea_surface_temperature"])
        self.assertIsNotNone(input_data.variables["amsre.nwp.10m_east_wind_component"])
        self.assertIsNotNone(input_data.variables["amsre.nwp.10m_north_wind_component"])
        self.assertIsNotNone(input_data.variables["amsre.nwp.skin_temperature"])
        self.assertIsNotNone(input_data.variables["amsre.nwp.log_surface_pressure"])
        self.assertIsNotNone(input_data.variables["amsre.nwp.cloud_liquid_water"])
        self.assertIsNotNone(input_data.variables["amsre.nwp.total_column_water_vapour"])
        self.assertIsNotNone(input_data.variables["amsre.nwp.total_precip"])

        # @todo 1 tb/tb add a mechanism that allows to variably define the sst-insitu source name
        # self.assertIsNotNone(input_data.variables["drifter-sst_insitu.time"])  # @todo 3 tb/tb check if this is ever used 2017-10-27
        # self.assertIsNotNone(input_data.variables["drifter-sst_insitu.lat"])  # @todo 3 tb/tb check if this is ever used 2017-10-27
        # self.assertIsNotNone(input_data.variables["drifter-sst_insitu.lon"])  # @todo 3 tb/tb check if this is ever used 2017-10-27
        # self.assertIsNotNone(input_data.variables["drifter-sst_insitu.sea_surface_temperature"])
        # self.assertIsNotNone(input_data.variables["drifter-sst_insitu.insitu_depth"])  @todo 3 tb/tb check if this is ever used 2017-10-27
        # self.assertIsNotNone(input_data.variables["drifter-sst_insitu.sst_qc_flag"])
        # self.assertIsNotNone(input_data.variables["drifter-sst_insitu.sst_track_flag"])

        # @todo 1 tb/tb verify that the following variables are converted to the appropriate target data type:
        # - amsre.solar_zenith_angle
        # - amsre.brightness_temperature6V
        # - amsre.brightness_temperature6H
        # - amsre.brightness_temperature10V
        # - amsre.brightness_temperature10H
        # - amsre.brightness_temperature18V
