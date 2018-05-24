import argparse
import os
import time

import numpy as np
import xarray as xr
from xarray import Variable

from dmi.sst.mw_oe.bt_bias_correction import BtBiasCorrection
from dmi.sst.mw_oe.flag_coding import FlagCoding
from dmi.sst.mw_oe.mmd_reader import MmdReader
from dmi.sst.mw_oe.preprocessor import Preprocessor
from dmi.sst.mw_oe.qa_processor import QaProcessor
from dmi.sst.mw_oe.retrieval import Retrieval
from dmi.sst.util.default_data import DefaultData

NUM_BT = 10
MAX_ITERATIONS = 10


class MwOeSstProcessor:
    _version = "0.1.2"

    KERNEL_SIZE = 4

    input_file = None
    output_directory = None

    def run(self, args):
        self.parse_cmd_line(args)

        print("reading input file: " + self.input_file)
        mmd_reader = MmdReader()
        mmd_data = mmd_reader.read(self.input_file)
        print("... success")

        start_time = time.time()

        matchup_count = mmd_data.dims["matchup_count"]
        print(matchup_count, " matches")

        flag_coding = FlagCoding(matchup_count)

        print("running preprocessing ...")
        preprocessor = Preprocessor()
        pre_proc_mmd_data = preprocessor.run(mmd_data, flag_coding)
        print("... success")

        print("running input data QA ...")
        qa_processor = QaProcessor()
        qa_processor.run_qa(pre_proc_mmd_data, flag_coding)
        print("... success")

        print("running input bias correction ...")
        bt_bias_correction = BtBiasCorrection()
        bt_bias_correction.run(pre_proc_mmd_data)
        print("... success")

        print("preparing output file ...")
        results = self._create_result_structure(matchup_count, MAX_ITERATIONS, NUM_BT)
        print("... success")

        print("running retrieval ...")
        retrieval = Retrieval()
        results = retrieval.run(pre_proc_mmd_data, results, flag_coding)
        print("... success")

        print("writing ")
        self.add_flags_variable(flag_coding, results)

        print("--- %s seconds ---" % (time.time() - start_time))

        self._write_result_data(self.output_directory, self.input_file, results)
        mmd_reader.close()
        print("... success")

    def parse_cmd_line(self, args):
        parser = self._create_cmd_line_parser()
        cmd_line_args = parser.parse_args(args)
        self.input_file = cmd_line_args.input_file
        if cmd_line_args.o is None:
            print("no output directory supplied - write to input directory")
            self.output_directory = os.path.dirname(self.input_file)
        else:
            self.output_directory = cmd_line_args.o[0]

    def add_flags_variable(self, flag_coding, results):
        flags = flag_coding.get_flags()
        variable = Variable(["matchup"], flags)
        variable.attrs["flag_masks"] = flag_coding.get_flag_masks()
        variable.attrs["flag_meanings"] = flag_coding.get_flag_meanings()
        results["flags"] = variable

    def _write_result_data(self, output_dir, input_file, results):
        target_file_name = self._create_target_file_name(input_file)
        target_path = os.path.join(output_dir, target_file_name)

        comp = dict(zlib=True, complevel=5)
        encoding = dict()
        for var_name in results.data_vars:
            var_encoding = dict(comp)
            var_encoding.update(results[var_name].encoding)
            encoding.update({var_name: var_encoding})

        results.to_netcdf(target_path, format='netCDF4', engine='netcdf4', encoding=encoding)

    @staticmethod
    def _create_result_structure(num_matchups, max_iterations, num_bt):
        dataset = xr.Dataset()

        # j
        variable = MwOeSstProcessor._create_2d_float_variable(max_iterations, num_matchups)
        dataset["j"] = variable # todo 1 tb/tb remove 2018-05-24

        # tb_rmse_ite
        variable = MwOeSstProcessor._create_2d_float_variable(max_iterations, num_matchups)
        dataset["tb_rmse_ite"] = variable # todo 1 tb/tb remove 2018-05-24

        # tb_rmse_ite0
        variable = MwOeSstProcessor._create_vector_float32_variable(num_matchups)
        dataset["tb_rmse_ite0"] = variable # todo 1 tb/tb remove 2018-05-24

        # tb_chi_ite
        variable = MwOeSstProcessor._create_2d_float_variable(max_iterations, num_matchups)
        dataset["tb_chi_ite"] = variable # todo 1 tb/tb remove 2018-05-24

        # convergence_passed_flag
        variable = MwOeSstProcessor._create_vector_uint8_variable(num_matchups)
        dataset["convergence_passed_flag"] = variable # todo 1 tb/tb remove 2018-05-24

        # convergence_passed_idx
        variable = MwOeSstProcessor._create_vector_uint8_variable(num_matchups)
        dataset["convergence_passed_idx"] = variable # todo 1 tb/tb remove 2018-05-24

        # di2
        variable = MwOeSstProcessor._create_2d_float_variable(max_iterations, num_matchups)
        dataset["di2"] = variable # todo 1 tb/tb remove 2018-05-24

        # dtb_ite0
        variable = MwOeSstProcessor._create_2d_float_variable(num_bt, num_matchups, dims_names=["matchup", "num_bt"])
        dataset["dtb_ite0"] = variable # todo 1 tb/tb remove 2018-05-24

        # TA0_ite0
        variable = MwOeSstProcessor._create_2d_float_variable(num_bt, num_matchups, dims_names=["matchup", "num_bt"])
        dataset["TA0_ite0"] = variable # todo 1 tb/tb remove 2018-05-24

        # j_ite0
        variable = MwOeSstProcessor._create_vector_float32_variable(num_matchups)
        dataset["j_ite0"] = variable # todo 1 tb/tb remove 2018-05-24

        # A
        variable = MwOeSstProcessor._create_2d_float_variable(MwOeSstProcessor.KERNEL_SIZE, num_matchups, dims_names=["matchup", "kernel_size"])
        dataset["A"] = variable

        # chisq
        variable = MwOeSstProcessor._create_vector_float32_variable(num_matchups)
        dataset["chisq"] = variable # todo 1 tb/tb remove 2018-05-24

        # mu_sst
        variable = MwOeSstProcessor._create_vector_float32_variable(num_matchups)
        dataset["mu_sst"] = variable

        # x - retrieved state vector
        variable = MwOeSstProcessor._create_2d_float_variable(MwOeSstProcessor.KERNEL_SIZE, num_matchups, dims_names=["matchup", "kernel_size"])
        dataset["x"] = variable

        # S
        variable = MwOeSstProcessor._create_2d_float_variable(MwOeSstProcessor.KERNEL_SIZE, num_matchups, dims_names=["matchup", "kernel_size"])
        dataset["S"] = variable # todo 1 tb/tb remove 2018-05-24

        # F(x) - simulated brightness temperatures from final iteration
        variable = MwOeSstProcessor._create_2d_float_variable(num_bt, num_matchups, dims_names=["matchup", "num_bt"])
        dataset["F"] = variable

        # y - observed brightness temperatures
        variable = MwOeSstProcessor._create_2d_float_variable(num_bt, num_matchups, dims_names=["matchup", "num_bt"])
        dataset["y"] = variable

        # dtb
        variable = MwOeSstProcessor._create_2d_float_variable(num_bt, num_matchups, dims_names=["matchup", "num_bt"])
        dataset["dtb"] = variable # todo 1 tb/tb remove 2018-05-24

        # ds
        variable = MwOeSstProcessor._create_vector_float32_variable(num_matchups)
        dataset["ds"] = variable # todo 1 tb/tb remove 2018-05-24

        # dn
        variable = MwOeSstProcessor._create_vector_float32_variable(num_matchups)
        dataset["dn"] = variable # todo 1 tb/tb remove 2018-05-24

        # K4
        variable = MwOeSstProcessor._create_2d_float_variable(num_bt, num_matchups, dims_names=["matchup", "num_bt"])
        dataset["K4"] = variable # todo 1 tb/tb remove 2018-05-24

        # ite_index
        variable = MwOeSstProcessor._create_vector_uint8_variable(num_matchups)
        dataset["ite_index"] = variable # todo 1 tb/tb remove 2018-05-24

        return dataset

    @staticmethod
    def _create_cmd_line_parser():
        parser = argparse.ArgumentParser(description='Microwave OE SST retrieval')
        parser.add_argument('input_file')
        parser.add_argument("-o", nargs=1)
        return parser

    @staticmethod
    def _create_vector_uint8_variable(num_matchups):
        array = DefaultData.create_default_vector(num_matchups, np.uint8)
        variable = Variable(["matchup"], array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.uint8)
        return variable

    @staticmethod
    def _create_vector_float32_variable(num_matchups):
        array = DefaultData.create_default_vector(num_matchups, np.float32, fill_value=np.NaN)
        variable = Variable(["matchup"], array)
        variable.attrs["_FillValue"] = np.NaN
        return variable

    @staticmethod
    def _create_2d_float_variable(max_iterations, num_matchups, dims_names=None):
        array = DefaultData.create_default_array(max_iterations, num_matchups, np.float32, dims_names=dims_names, fill_value=np.NaN)
        if dims_names is None:
            variable = Variable(["matchup", "iterations"], array)
        else:
            variable = Variable(dims_names, array)

        variable.attrs["_FillValue"] = np.NaN
        return variable

    @staticmethod
    def _create_target_file_name(test_mmd):
        (head, file_name) = os.path.split(test_mmd)
        (prefix, extension) = os.path.splitext(file_name)
        return prefix + "_oe-sst" + extension
