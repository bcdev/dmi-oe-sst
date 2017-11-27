import numpy as np
import xarray as xr
from xarray import Variable

from dmi.sst.mw_oe.pressure_processor import PressureProcessor
from dmi.sst.util.default_data import DefaultData


class Preprocessor:
    TO_SQUEEZE_NAMES = ["insitu.time", "insitu.lat", "insitu.lon", "insitu.sea_surface_temperature", "insitu.sst_depth", "insitu.sst_qc_flag", "insitu.sst_track_flag]"]
    TO_AVERAGE_NAMES = ["amsre.brightness_temperature6V", "amsre.brightness_temperature6H", "amsre.brightness_temperature10V", "amsre.brightness_temperature10H", "amsre.brightness_temperature18V",
                        "amsre.brightness_temperature18H", "amsre.brightness_temperature23V", "amsre.brightness_temperature23H", "amsre.brightness_temperature36V", "amsre.brightness_temperature36H"]
    TO_CENTER_EXTRACT_NAMES = ["amsre.nwp.sea_surface_temperature", "amsre.nwp.skin_temperature",
                               "amsre.nwp.log_surface_pressure", "amsre.nwp.cloud_liquid_water", "amsre.nwp.total_column_water_vapour", "amsre.nwp.total_precip", "amsre.pixel_data_quality6V",
                               "amsre.pixel_data_quality6H", "amsre.pixel_data_quality10V", "amsre.pixel_data_quality10H", "amsre.pixel_data_quality18V", "amsre.pixel_data_quality18H",
                               "amsre.pixel_data_quality23V", "amsre.pixel_data_quality23H", "amsre.pixel_data_quality36V", "amsre.pixel_data_quality36H", "amsre.solar_zenith_angle",
                               "amsre.scan_data_quality", "amsre.satellite_zenith_angle", "amsre.satellite_azimuth_angle", "amsre.Geostationary_Reflection_Latitude",
                               "amsre.Geostationary_Reflection_Longitude", "amsre.latitude", "amsre.longitude"]
    WIND_SPEED_VARIABLES = ["amsre.nwp.10m_east_wind_component", "amsre.nwp.10m_north_wind_component"]

    AVERAGING_LENGTH = 5  # @todo 3 tb/tb this can be a parameter to the processor 2017-11-17
    INV_GRAVITY_CONST = 1.0 / 9.80665  # s^2/m

    def run(self, dataset):
        preprocessed_data = xr.Dataset()
        num_matchups = len(dataset.coords["matchup_count"])
        invalid_data_array = np.zeros(num_matchups, dtype=np.bool)

        for variable_name in dataset.variables:
            if variable_name in self.TO_SQUEEZE_NAMES:
                self.squeeze_data(dataset, preprocessed_data, variable_name)
                continue

            if variable_name in self.TO_AVERAGE_NAMES:
                self.average_subset(dataset, invalid_data_array, preprocessed_data, variable_name)
                continue

            if variable_name in self.TO_CENTER_EXTRACT_NAMES:
                self.extract_center_px(dataset, preprocessed_data, variable_name)
                continue

            if variable_name in self.WIND_SPEED_VARIABLES:
                self.process_wind_speed(dataset, preprocessed_data)
                continue

        self.calculate_TCLW(preprocessed_data)

        preprocessed_data["invalid_data"] = Variable(["matchup"], invalid_data_array)
        return preprocessed_data

    def process_wind_speed(self, dataset, preprocessed_data):
        self.extract_center_px(dataset, preprocessed_data, self.WIND_SPEED_VARIABLES[0])
        self.extract_center_px(dataset, preprocessed_data, self.WIND_SPEED_VARIABLES[1])
        east_wind_data = preprocessed_data.variables[self.WIND_SPEED_VARIABLES[0]].data
        north_wind_data = preprocessed_data.variables[self.WIND_SPEED_VARIABLES[1]].data
        abs_wind_speed_data = np.sqrt(np.square(east_wind_data) + np.square(north_wind_data))
        preprocessed_data["amsre.nwp.abs_wind_speed"] = Variable(["matchup"], abs_wind_speed_data)

    def average_subset(self, dataset, invalid_data_array, preprocessed_data, variable_name):
        num_matchups = len(invalid_data_array)
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

        for i in range(0, num_matchups - 1):
            layer = dataset.variables[variable_name][i, y_min:y_max, x_min: x_max]
            masked_layer = np.ma.masked_values(layer, fill_value)
            num_fills = np.ma.count_masked(masked_layer)
            if num_fills <= max_num_invalid:
                target_data[i] = np.ma.average(masked_layer)
            else:
                target_data[i] = fill_value
                invalid_data_array[i] = True

        preprocessed_data[variable_name] = Variable(["matchup"], target_data)

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
