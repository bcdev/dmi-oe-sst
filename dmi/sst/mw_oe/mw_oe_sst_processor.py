import numpy as np
import xarray as xr
from xarray import Variable

from dmi.sst.util.default_data import DefaultData


class MwOeSstProcessor:
    _version = "0.0.1"

    KERNEL_SIZE = 4

    def run(self, cmd_line_args):
        input_file = cmd_line_args.input_file

        # xarray can not handle the TAI 1993 time coding @todo 3 tb/th adapt if possible
        xr.open_dataset(input_file, decode_times=False)

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
