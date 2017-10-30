import xarray as xr

from dmi.sst.mw_oe.constants import INPUT_VARIABLES


class MmdReader:
    def read(self, input_file):
        # xarray can not handle the TAI 1993 time coding @todo 3 tb/th adapt if possible
        input_data = xr.open_dataset(input_file, decode_times=False)

        subset_data = xr.Dataset()

        for variable_name in INPUT_VARIABLES:
            variable = input_data.variables[variable_name]
            if "SCALE_FACTOR" in variable.attrs or "OFFSET" in variable.attrs:
                self._scale_data(variable)

            subset_data[variable_name] = variable

        return subset_data

    # @todo 3 tb/tb write test 2017-10-30
    def _scale_data(self, variable):
        scale_factor = 1.0
        if "SCALE_FACTOR" in variable.attrs:
            scale_factor = variable.attrs["SCALE_FACTOR"]
        offset = 0.0
        if "OFFSET" in variable.attrs:
            offset = variable.attrs["OFFSET"]
        if scale_factor != 1.0 or offset != 0.0:
            scaled_data = variable.data * scale_factor + offset
            variable.data = scaled_data
