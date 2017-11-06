import numpy as np
import xarray as xr
from xarray import Variable

from dmi.sst.mw_oe.mmd_reader import MmdReader
from dmi.sst.util.default_data import DefaultData


class MwOeSstProcessor:
    _version = "0.0.1"

    KERNEL_SIZE = 4

    def run(self, cmd_line_args):
        input_file = cmd_line_args.input_file

        mmd_reader = MmdReader()
        mmd_data = mmd_reader.read(input_file)

        # preprocessing
        #
        # mean over 5x5
        # - amsre.nwp.seaice_fraction
        #
        # central pixel only
        # - amsre.latitude
        # - amsre.longitude
        # - amsre.time
        # - amsre.solar_zenith_angle
        # - amsre.satellite_zenith_angle
        # - amsre.nwp.sea_surface_temperature
        # - amsre.nwp.10m_east_wind_component
        # - amsre.nwp.10m_north_wind_component
        # - amsre.nwp.skin_temperature
        # - amsre.nwp.log_surface_pressure
        # - amsre.nwp.total_column_water_vapour
        # - amsre.nwp.total_precip
        # - amsre.brightness_temperature6V
        # - amsre.brightness_temperature6H
        # - amsre.brightness_temperature10V
        # - amsre.brightness_temperature10H
        # - amsre.brightness_temperature18V
        # - amsre.brightness_temperature18H
        # - amsre.brightness_temperature23V
        # - amsre.brightness_temperature23H
        # - amsre.brightness_temperature36V
        # - amsre.brightness_temperature36H
        # - amsre.pixel_data_quality6V
        # - amsre.pixel_data_quality6H
        # - amsre.pixel_data_quality10V
        # - amsre.pixel_data_quality10H
        # - amsre.pixel_data_quality18V
        # - amsre.pixel_data_quality18H
        # - amsre.pixel_data_quality23V
        # - amsre.pixel_data_quality23H
        # - amsre.pixel_data_quality36V
        # - amsre.pixel_data_quality36H
        # - amsre.scan_data_quality
        # - amsre.Geostationary_Reflection_Latitude
        # - amsre.Geostationary_Reflection_Longitude
        # - amsre.satellite_azimuth_angle
        #
        # calculate TCLW
        # pre: log_surface_pressure present
        # - amsre.nwp.cloud_liquid_water
        #
        # strange flag data processing
        # pre: sea_ice_fraction
        # - amsre.land_ocean_flag_6
        #
        # mean over 5x5 and 11x11 @todo 2tb/tb check if these are really used 2017-11-06
        # - amsre.brightness_temperature6V
        # - amsre.brightness_temperature6H
        # - amsre.brightness_temperature10V
        # - amsre.brightness_temperature10H
        # - amsre.brightness_temperature18V
        # - amsre.brightness_temperature18H
        # - amsre.brightness_temperature23H
        # - amsre.brightness_temperature36V
        # - amsre.brightness_temperature36H
        #
        #
        # nothing
        # - insitu.time
        # - insitu.lat
        # - insitu.lon
        # - insitu.sea_surface_temperature
        # - insitu.sst_depth
        # - insitu.sst_qc_flag
        # - insitu.sst_track_flag

    @staticmethod
    def create_result_structure(num_matchups, max_iterations, num_bt):
        dataset = xr.Dataset()

        # j
        variable = MwOeSstProcessor.create_2d_float_variable(max_iterations, num_matchups)
        dataset["j"] = variable

        # tb_rmse_ite
        variable = MwOeSstProcessor.create_2d_float_variable(max_iterations, num_matchups)
        dataset["tb_rmse_ite"] = variable

        # tb_rmse_ite0
        variable = MwOeSstProcessor.create_vector_float32_variable(num_matchups)
        dataset["tb_rmse_ite0"] = variable

        # tb_chi_ite
        variable = MwOeSstProcessor.create_2d_float_variable(max_iterations, num_matchups)
        dataset["tb_chi_ite"] = variable

        # convergence_passed_flag
        variable = MwOeSstProcessor.create_vector_uint8_variable(num_matchups)
        dataset["convergence_passed_flag"] = variable

        # convergence_passed_idx
        variable = MwOeSstProcessor.create_vector_uint8_variable(num_matchups)
        dataset["convergence_passed_idx"] = variable

        # di2
        variable = MwOeSstProcessor.create_2d_float_variable(max_iterations, num_matchups)
        dataset["di2"] = variable

        # dtb_ite0
        variable = MwOeSstProcessor.create_2d_float_variable(num_bt, num_matchups, dims_names=["matchup", "num_bt"])
        dataset["dtb_ite0"] = variable

        # TAO_ite0
        variable = MwOeSstProcessor.create_2d_float_variable(num_bt, num_matchups, dims_names=["matchup", "num_bt"])
        dataset["TAO_ite0"] = variable

        # j_ite0
        variable = MwOeSstProcessor.create_vector_float32_variable(num_matchups)
        dataset["j_ite0"] = variable

        # AK
        variable = MwOeSstProcessor.create_2d_float_variable(MwOeSstProcessor.KERNEL_SIZE, num_matchups,
                                                             dims_names=["matchup", "kernel_size"])
        dataset["AK"] = variable

        # chisq
        variable = MwOeSstProcessor.create_vector_float32_variable(num_matchups)
        dataset["chisq"] = variable

        # tb_rmse
        variable = MwOeSstProcessor.create_vector_float32_variable(num_matchups)
        dataset["tb_rmse"] = variable

        # p
        variable = MwOeSstProcessor.create_2d_float_variable(MwOeSstProcessor.KERNEL_SIZE, num_matchups,
                                                             dims_names=["matchup", "kernel_size"])
        dataset["p"] = variable

        # S
        variable = MwOeSstProcessor.create_2d_float_variable(MwOeSstProcessor.KERNEL_SIZE, num_matchups,
                                                             dims_names=["matchup", "kernel_size"])
        dataset["S"] = variable

        # tb_sim
        variable = MwOeSstProcessor.create_2d_float_variable(num_bt, num_matchups, dims_names=["matchup", "num_bt"])
        dataset["tb_sim"] = variable

        # dtb
        variable = MwOeSstProcessor.create_2d_float_variable(num_bt, num_matchups, dims_names=["matchup", "num_bt"])
        dataset["dtb"] = variable

        # ds
        variable = MwOeSstProcessor.create_vector_float32_variable(num_matchups)
        dataset["ds"] = variable

        # dn
        variable = MwOeSstProcessor.create_vector_float32_variable(num_matchups)
        dataset["dn"] = variable

        # K4
        variable = MwOeSstProcessor.create_2d_float_variable(num_bt, num_matchups, dims_names=["matchup", "num_bt"])
        dataset["K4"] = variable

        # ite_index
        variable = MwOeSstProcessor.create_vector_uint8_variable(num_matchups)
        dataset["ite_index"] = variable

        return dataset

    @staticmethod
    def create_vector_uint8_variable(num_matchups):
        array = DefaultData.create_default_vector(num_matchups, np.uint8)
        variable = Variable(["matchup"], array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.uint8)
        return variable

    @staticmethod
    def create_vector_float32_variable(num_matchups):
        array = DefaultData.create_default_vector(num_matchups, np.float32, fill_value=np.NaN)
        variable = Variable(["matchup"], array)
        variable.attrs["_FillValue"] = np.NaN
        return variable

    @staticmethod
    def create_2d_float_variable(max_iterations, num_matchups, dims_names=None):
        array = DefaultData.create_default_array(max_iterations, num_matchups, np.float32, dims_names=dims_names,
                                                 fill_value=np.NaN)
        if dims_names is None:
            variable = Variable(["matchup", "iterations"], array)
        else:
            variable = Variable(dims_names, array)

        variable.attrs["_FillValue"] = np.NaN
        return variable
