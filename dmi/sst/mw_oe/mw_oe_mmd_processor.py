import argparse
import sys

import numpy as np
import xarray as xr
from xarray import Variable

from dmi.sst.util.default_data import DefaultData


class MwOeMMDProcessor:
    _version = "0.0.1"

    def run(self, cmd_line_args):
        input_file = cmd_line_args.input_file

        # xarray can not handle the TAI 1993 time coding @todo 3 tb/th adapt if possible
        xr.open_dataset(input_file, decode_times=False)

    @staticmethod
    def create_result_structure(num_matchups, max_iterations, num_bt):
        dataset = xr.Dataset()

        # j
        variable = MwOeMMDProcessor.create_2d_float_variable(max_iterations, num_matchups)
        dataset["j"] = variable

        # tb_rmse_ite
        variable = MwOeMMDProcessor.create_2d_float_variable(max_iterations, num_matchups)
        dataset["tb_rmse_ite"] = variable

        # tb_rmse_ite0
        variable = MwOeMMDProcessor.create_vector_float32_variable(num_matchups)
        dataset["tb_rmse_ite0"] = variable

        # tb_chi_ite
        variable = MwOeMMDProcessor.create_2d_float_variable(max_iterations, num_matchups)
        dataset["tb_chi_ite"] = variable

        # convergence_passed_flag
        variable = MwOeMMDProcessor.create_vector_uint8_variable(num_matchups)
        dataset["convergence_passed_flag"] = variable

        # convergence_passed_idx
        variable = MwOeMMDProcessor.create_vector_uint8_variable(num_matchups)
        dataset["convergence_passed_idx"] = variable

        # di2
        variable = MwOeMMDProcessor.create_2d_float_variable(max_iterations, num_matchups)
        dataset["di2"] = variable

        # dtb_ite0
        variable = MwOeMMDProcessor.create_2d_float_variable(num_bt, num_matchups, dims_names=["matchup", "num_bt"])
        dataset["dtb_ite0"] = variable

        # TAO_ite0
        variable = MwOeMMDProcessor.create_2d_float_variable(num_bt, num_matchups, dims_names=["matchup", "num_bt"])
        dataset["TAO_ite0"] = variable

        # j_ite0
        variable = MwOeMMDProcessor.create_vector_float32_variable(num_matchups)
        dataset["j_ite0"] = variable

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


def main(args=None) -> int:
    parser = create_cmd_line_parser()
    cmd_line_args = parser.parse_args(args)

    processor = MwOeMMDProcessor()
    processor.run(cmd_line_args)

    return 0


def create_cmd_line_parser():
    parser = argparse.ArgumentParser(description='Microwave OE SST retrieval')
    parser.add_argument('input_file')
    return parser


if __name__ == "__main__":
    sys.exit(main())
