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
        self.assertIsNotNone(input_data.variables["amsre.satellite_azimuth_angle"])
        self.assertIsNotNone(input_data.variables["amsre.land_ocean_flag_6"])
        self.assertIsNotNone(input_data.variables["amsre.brightness_temperature6V"])
        self.assertIsNotNone(input_data.variables["amsre.brightness_temperature6H"])
        self.assertIsNotNone(input_data.variables["amsre.brightness_temperature10V"])
        self.assertIsNotNone(input_data.variables["amsre.brightness_temperature10H"])
        self.assertIsNotNone(input_data.variables["amsre.brightness_temperature18V"])
        self.assertIsNotNone(input_data.variables["amsre.brightness_temperature18H"])
        self.assertIsNotNone(input_data.variables["amsre.brightness_temperature23V"])
        self.assertIsNotNone(input_data.variables["amsre.brightness_temperature23H"])
        self.assertIsNotNone(input_data.variables["amsre.brightness_temperature36V"])
        self.assertIsNotNone(input_data.variables["amsre.brightness_temperature36H"])
        self.assertIsNotNone(input_data.variables["amsre.pixel_data_quality6V"])
        self.assertIsNotNone(input_data.variables["amsre.pixel_data_quality6H"])
        self.assertIsNotNone(input_data.variables["amsre.pixel_data_quality10V"])
        self.assertIsNotNone(input_data.variables["amsre.pixel_data_quality10H"])
        self.assertIsNotNone(input_data.variables["amsre.pixel_data_quality18V"])
        self.assertIsNotNone(input_data.variables["amsre.pixel_data_quality18H"])
        self.assertIsNotNone(input_data.variables["amsre.pixel_data_quality23V"])
        self.assertIsNotNone(input_data.variables["amsre.pixel_data_quality23H"])
        self.assertIsNotNone(input_data.variables["amsre.pixel_data_quality36V"])
        self.assertIsNotNone(input_data.variables["amsre.pixel_data_quality36H"])
        self.assertIsNotNone(input_data.variables["amsre.scan_data_quality"])
        self.assertIsNotNone(input_data.variables["amsre.Geostationary_Reflection_Latitude"])
        self.assertIsNotNone(input_data.variables["amsre.Geostationary_Reflection_Longitude"])

        self.assertIsNotNone(input_data.variables["amsre.nwp.seaice_fraction"])
        self.assertIsNotNone(input_data.variables["amsre.nwp.sea_surface_temperature"])
        self.assertIsNotNone(input_data.variables["amsre.nwp.10m_east_wind_component"])
        self.assertIsNotNone(input_data.variables["amsre.nwp.10m_north_wind_component"])
        self.assertIsNotNone(input_data.variables["amsre.nwp.skin_temperature"])
        self.assertIsNotNone(input_data.variables["amsre.nwp.log_surface_pressure"])
        self.assertIsNotNone(input_data.variables["amsre.nwp.cloud_liquid_water"])
        self.assertIsNotNone(input_data.variables["amsre.nwp.total_column_water_vapour"])
        self.assertIsNotNone(input_data.variables["amsre.nwp.total_precip"])

        self.assertIsNotNone(input_data.variables["insitu.time"])   # @todo 3 tb/tb check if this is ever used 2017-10-27
        self.assertIsNotNone(input_data.variables["insitu.lat"])    # @todo 3 tb/tb check if this is ever used 2017-10-27
        self.assertIsNotNone(input_data.variables["insitu.lon"])    # @todo 3 tb/tb check if this is ever used 2017-10-27
        self.assertIsNotNone(input_data.variables["insitu.sea_surface_temperature"])
        self.assertIsNotNone(input_data.variables["insitu.sst_depth"])  # @todo 3 tb/tb check if this is ever used 2017-10-27
        self.assertIsNotNone(input_data.variables["insitu.sst_qc_flag"])
        self.assertIsNotNone(input_data.variables["insitu.sst_track_flag"])

        # verify that the following variables are converted to the appropriate target data type:
        self.assertAlmostEqual(55.125, input_data.variables["amsre.satellite_zenith_angle"].data[0, 0, 0], 8)
        self.assertAlmostEqual(44.049999, input_data.variables["amsre.satellite_azimuth_angle"].data[1, 0, 0], 6)
        self.assertAlmostEqual(175.59999, input_data.variables["amsre.brightness_temperature6V"].data[1, 1, 0], 5)
        self.assertAlmostEqual(90.459991, input_data.variables["amsre.brightness_temperature6H"].data[1, 1, 1], 6)
        self.assertAlmostEqual(180.05, input_data.variables["amsre.brightness_temperature10V"].data[2, 1, 1], 5)
        self.assertAlmostEqual(93.819992, input_data.variables["amsre.brightness_temperature10H"].data[2, 2, 1], 6)
        self.assertAlmostEqual(200.73999, input_data.variables["amsre.brightness_temperature18V"].data[2, 2, 2], 6)
        self.assertAlmostEqual(127.59, input_data.variables["amsre.brightness_temperature18H"].data[3, 2, 2], 5)
        self.assertAlmostEqual(229.01999, input_data.variables["amsre.brightness_temperature23V"].data[3, 3, 2], 5)
        self.assertAlmostEqual(176.75999, input_data.variables["amsre.brightness_temperature23H"].data[3, 3, 3], 5)
        self.assertAlmostEqual(215.17, input_data.variables["amsre.brightness_temperature36V"].data[4, 3, 3], 5)
        self.assertAlmostEqual(152.19, input_data.variables["amsre.brightness_temperature36H"].data[4, 4, 3], 5)
        self.assertAlmostEqual(-6.7799997, input_data.variables["amsre.Geostationary_Reflection_Latitude"].data[4, 4, 4], 7)
        self.assertAlmostEqual(-22.959999, input_data.variables["amsre.Geostationary_Reflection_Longitude"].data[5, 4, 4], 6)
