import xarray as xr

from dmi.sst.mw_oe.constants import INPUT_VARIABLES


class MmdReader:

    def read(self, input_file):
        # xarray can not handle the TAI 1993 time coding @todo 3 tb/th adapt if possible
        input_data = xr.open_dataset(input_file, decode_times=False)

        subset_data = xr.Dataset()

        for variable_name in INPUT_VARIABLES:
            subset_data[variable_name] = input_data.variables[variable_name]


        return subset_data