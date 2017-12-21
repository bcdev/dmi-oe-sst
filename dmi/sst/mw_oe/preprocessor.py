import numpy as np
import xarray as xr
from xarray import Variable

from dmi.sst.mw_oe.pressure_processor import PressureProcessor
from dmi.sst.util.default_data import DefaultData


class Preprocessor:
    TO_SQUEEZE_NAMES = ["insitu.time", "insitu.lat", "insitu.lon", "insitu.sea_surface_temperature", "insitu.sst_depth", "insitu.sst_qc_flag", "insitu.sst_track_flag]"]
    TO_AVERAGE_NAMES = ["amsre.brightness_temperature6V", "amsre.brightness_temperature6H", "amsre.brightness_temperature10V", "amsre.brightness_temperature10H", "amsre.brightness_temperature18V",
                        "amsre.brightness_temperature18H", "amsre.brightness_temperature23V", "amsre.brightness_temperature23H", "amsre.brightness_temperature36V", "amsre.brightness_temperature36H"]
    TO_STDDEV_NAMES = ["amsre.brightness_temperature23V", "amsre.brightness_temperature23H", "amsre.brightness_temperature36V", "amsre.brightness_temperature36H"]
    TO_CENTER_EXTRACT_NAMES = ["amsre.nwp.sea_surface_temperature", "amsre.nwp.skin_temperature", "amsre.nwp.log_surface_pressure", "amsre.nwp.cloud_liquid_water",
                               "amsre.nwp.total_column_water_vapour", "amsre.nwp.total_precip", "amsre.pixel_data_quality6V", "amsre.pixel_data_quality6H", "amsre.pixel_data_quality10V",
                               "amsre.pixel_data_quality10H", "amsre.pixel_data_quality18V", "amsre.pixel_data_quality18H", "amsre.pixel_data_quality23V", "amsre.pixel_data_quality23H",
                               "amsre.pixel_data_quality36V", "amsre.pixel_data_quality36H", "amsre.solar_zenith_angle", "amsre.scan_data_quality", "amsre.satellite_zenith_angle",
                               "amsre.satellite_azimuth_angle", "amsre.Geostationary_Reflection_Latitude", "amsre.Geostationary_Reflection_Longitude", "amsre.latitude", "amsre.longitude"]
    WIND_SPEED_VARIABLES = ["amsre.nwp.10m_east_wind_component", "amsre.nwp.10m_north_wind_component"]
    NWP_SST_VARIABLES = ["amsre.nwp.sea_surface_temperature"]
    FILENAME_VARIABLES = ["amsre.l2a_filename"]

    AVERAGING_LENGTH = 5  # @todo 3 tb/tb this can be a parameter to the processor 2017-11-17
    STDDEV_LENGTH = 21  # @todo 3 tb/tb this can be a parameter to the processor 2017-12-13
    INV_GRAVITY_CONST = 1.0 / 9.80665  # s^2/m
    SST_NWP_BIAS = -0.05

    def run(self, dataset, flag_coding=None):
        preprocessed_data = xr.Dataset()

        for variable_name in dataset.variables:
            if variable_name in self.TO_SQUEEZE_NAMES:
                self.squeeze_data(dataset, preprocessed_data, variable_name)
                continue

            if variable_name in self.TO_AVERAGE_NAMES:
                if variable_name in self.TO_STDDEV_NAMES:
                    self.calc_std_dev(dataset, preprocessed_data, variable_name, flag_coding)
                self.average_subset(dataset, preprocessed_data, variable_name, flag_coding)
                continue

            if variable_name in self.TO_CENTER_EXTRACT_NAMES:
                self.extract_center_px(dataset, preprocessed_data, variable_name)
                self.convert_temperature(preprocessed_data, variable_name)
                self.apply_sst_nwp_bias(preprocessed_data, variable_name)
                continue

            if variable_name in self.WIND_SPEED_VARIABLES:
                self.process_wind_speed_and_relative_angle(dataset, preprocessed_data)
                continue

            if variable_name in self.FILENAME_VARIABLES:
                self.extract_ascending_descending(dataset, preprocessed_data, flag_coding)
                continue

        self.calculate_TCLW(preprocessed_data)

        return preprocessed_data

    def convert_temperature(self, preprocessed_data, variable_name):
        if variable_name in self.NWP_SST_VARIABLES:
            sst_data = preprocessed_data[variable_name].data
            sst_data = sst_data - 273.15
            preprocessed_data[variable_name] = Variable(["matchup"], sst_data)

    def apply_sst_nwp_bias(self, preprocessed_data, variable_name):
        if variable_name in self.NWP_SST_VARIABLES:
            sst_data = preprocessed_data[variable_name].data
            sst_data = sst_data + self.SST_NWP_BIAS
            preprocessed_data[variable_name] = Variable(["matchup"], sst_data)

    def process_wind_speed_and_relative_angle(self, dataset, preprocessed_data):
        self.extract_center_px(dataset, preprocessed_data, self.WIND_SPEED_VARIABLES[0])
        self.extract_center_px(dataset, preprocessed_data, self.WIND_SPEED_VARIABLES[1])
        east_wind_data = preprocessed_data.variables[self.WIND_SPEED_VARIABLES[0]].data
        north_wind_data = preprocessed_data.variables[self.WIND_SPEED_VARIABLES[1]].data

        abs_wind_speed_data = np.sqrt(np.square(east_wind_data) + np.square(north_wind_data))
        preprocessed_data["amsre.nwp.abs_wind_speed"] = Variable(["matchup"], abs_wind_speed_data)

        num_matchups = len(dataset.coords["matchup_count"])
        self.extract_center_px(dataset, preprocessed_data, "amsre.satellite_azimuth_angle")
        target_data = DefaultData.create_default_vector(num_matchups, np.float32, fill_value=np.NaN)
        phi_sat = preprocessed_data.variables["amsre.satellite_azimuth_angle"].data
        for i in range(0, num_matchups):
            target_data[i] = self.calculate_relative_angle(phi_sat[i], north_wind_data[i], east_wind_data[i])

        preprocessed_data["relative_angle"] = Variable(["matchup"], target_data)

    def average_subset(self, dataset, preprocessed_data, variable_name, flag_coding=None):
        num_matchups = len(dataset.coords["matchup_count"])
        invalid_data_array = np.zeros(num_matchups, dtype=np.bool)
        variable = dataset.variables[variable_name]
        fill_value = variable.attrs["_FillValue"]
        target_data = DefaultData.create_default_vector(num_matchups, np.float32, fill_value)

        width = variable.shape[2]
        height = variable.shape[1]
        center_x = int(np.floor(width / 2))
        center_y = int(np.floor(height / 2))

        offset = int(np.floor(self.AVERAGING_LENGTH / 2))
        y_min = center_y - offset
        y_max = center_y + offset + 1
        x_min = center_x - offset
        x_max = center_x + offset + 1

        max_num_invalid = int(np.ceil(self.AVERAGING_LENGTH * self.AVERAGING_LENGTH * 0.1))

        for i in range(0, num_matchups):
            layer = dataset.variables[variable_name][i, y_min:y_max, x_min: x_max]
            masked_layer = np.ma.masked_values(layer, fill_value)
            num_fills = np.ma.count_masked(masked_layer)
            if num_fills <= max_num_invalid:
                target_data[i] = np.ma.average(masked_layer)
            else:
                target_data[i] = fill_value
                invalid_data_array[i] = True

        if flag_coding is not None:
            flag_coding.add_avg_inv_thresh(invalid_data_array)

        preprocessed_data[variable_name] = Variable(["matchup"], target_data)

    # @todo 2 tb/tb refactor, this method duplicates most of the normal averaging method 2017-12-13
    def calc_std_dev(self, dataset, preprocessed_data, variable_name, flag_coding=None):
        num_matchups = len(dataset.coords["matchup_count"])
        invalid_data_array = np.zeros(num_matchups, dtype=np.bool)
        variable = dataset.variables[variable_name]
        fill_value = variable.attrs["_FillValue"]
        target_data = DefaultData.create_default_vector(num_matchups, np.float32, fill_value)

        width = variable.shape[2]
        height = variable.shape[1]
        center_x = int(np.floor(width / 2))
        center_y = int(np.floor(height / 2))

        offset = int(np.floor(self.STDDEV_LENGTH / 2))
        y_min = center_y - offset
        y_max = center_y + offset + 1
        x_min = center_x - offset
        x_max = center_x + offset + 1

        max_num_invalid = int(np.ceil(self.STDDEV_LENGTH * self.STDDEV_LENGTH * 0.1))

        for i in range(0, num_matchups):
            layer = dataset.variables[variable_name][i, y_min:y_max, x_min: x_max]
            masked_layer = np.ma.masked_values(layer, fill_value)
            num_fills = np.ma.count_masked(masked_layer)
            if num_fills <= max_num_invalid:
                target_data[i] = np.ma.std(masked_layer)
            else:
                target_data[i] = fill_value
                invalid_data_array[i] = True

        if flag_coding is not None:
            flag_coding.add_avg_inv_thresh(invalid_data_array)

        preprocessed_data[variable_name + "_stddev"] = Variable(["matchup"], target_data)

    def extract_center_px(self, dataset, preprocessed_data, variable_name):
        variable = dataset.variables[variable_name]
        if len(variable.shape) == 3:
            width = variable.shape[2]
            height = variable.shape[1]
            center_x = int(np.floor(width / 2))
            center_y = int(np.floor(height / 2))
            preprocessed_data[variable_name] = variable[:, center_y, center_x].squeeze()
        elif len(variable.shape) == 4:
            width = variable.shape[3]
            height = variable.shape[2]
            center_x = int(np.floor(width / 2))
            center_y = int(np.floor(height / 2))
            preprocessed_data[variable_name] = variable[:, :, center_y, center_x].squeeze()

    def squeeze_data(self, dataset, preprocessed_data, variable_name):
        preprocessed_data[variable_name] = dataset.variables[variable_name].squeeze()

    def calculate_TCLW(self, preprocessed_data):
        surface_pressure = np.exp(preprocessed_data["amsre.nwp.log_surface_pressure"])

        pressure_processor = PressureProcessor()
        pressure_levels = pressure_processor.calculate_pressure_levels(surface_pressure)

        clw = preprocessed_data["amsre.nwp.cloud_liquid_water"]

        tclw_tmp = clw.data * pressure_levels.data
        tclw_tmp = tclw_tmp * self.INV_GRAVITY_CONST
        tclw = np.sum(tclw_tmp, axis=1)
        preprocessed_data["amsre.nwp.total_column_liquid_water"] = Variable(["num_matchups"], tclw)

    def extract_ascending_descending(self, dataset, preprocessed_data, flag_coding=None):
        num_matchups = len(dataset.coords["matchup_count"])
        ascending_data_array = np.zeros(num_matchups, dtype=np.bool)
        invalid_data_array = np.zeros(num_matchups, dtype=np.bool)

        filename_data = dataset.variables["amsre.l2a_filename"].data
        for i in range(0, num_matchups):
            file_name = str(filename_data[i])
            if "_A." in file_name:
                ascending_data_array[i] = True
            elif "_D." in file_name:
                ascending_data_array[i] = False
            else:
                invalid_data_array[i] = True

        if flag_coding is not None:
            flag_coding.add_inv_filename(invalid_data_array)

        preprocessed_data["amsre.ascending"] = Variable(["matchup"], ascending_data_array)

    def calculate_relative_angle(self, phi_sat, north_wind, east_wind):
        if phi_sat < 0.0:
            phi_sat = phi_sat + 360.0

        phi_w = 90.0 - np.arctan2(north_wind, east_wind)
        if phi_w < 0.0:
            phi_w = phi_w + 360.0

        phi_rel = phi_sat - phi_w
        if phi_rel < 0.0:
            phi_rel = phi_rel + 360.0

        return phi_rel
