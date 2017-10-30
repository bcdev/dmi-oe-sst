import xarray as xr

from dmi.sst.mw_oe.constants import INPUT_VARIABLES

OFFSET = "OFFSET"
SCALE_FACTOR = "SCALE_FACTOR"


class MmdReader:
    def read(self, input_file):
        # xarray can not handle the TAI 1993 time coding @todo 3 tb/th adapt if possible
        input_data = xr.open_dataset(input_file, decode_times=False)

        subset_data = xr.Dataset()

        for variable_name in INPUT_VARIABLES:
            target_variable_name = variable_name
            if "{IS_SENSOR}" in variable_name:
                in_situ_sensor = self._get_insitu_sensor(input_data)
                variable_name = variable_name.replace("{IS_SENSOR}", in_situ_sensor)
                # @todo 1 tb/tb continue here - assemble correct variable name 2017-10-30
                target_variable_name = "bla"
            variable = input_data.variables[variable_name]
            if SCALE_FACTOR in variable.attrs or OFFSET in variable.attrs:
                self._scale_data(variable)

            subset_data[target_variable_name] = variable

        return subset_data

    # @todo 3 tb/tb write test 2017-10-30
    def _scale_data(self, variable):
        scale_factor = 1.0
        if SCALE_FACTOR in variable.attrs:
            scale_factor = variable.attrs[SCALE_FACTOR]
        offset = 0.0
        if OFFSET in variable.attrs:
            offset = variable.attrs[OFFSET]
        if scale_factor != 1.0 or offset != 0.0:
            scaled_data = variable.data * scale_factor + offset
            variable.data = scaled_data

    # @todo 2 tb/tb write test 2017-10-30
    def _get_insitu_sensor(self, input_data):
        variable_keys = input_data.variables.keys()
        for variable_key in variable_keys:
            if "animal-sst" in variable_key:
                return "animal-sst"
            elif "argo-sst" in variable_key:
                return "argo-sst"
            elif "bottle-sst" in variable_key:
                return "bottle-sst"
            elif "ctd-sst" in variable_key:
                return "ctd-sst"
            elif "drifter-sst" in variable_key:
                return "drifter-sst"
            elif "gtmba-sst" in variable_key:
                return "gtmba-sst"
            elif "mbt-sst" in variable_key:
                return "mbt-sst"
            elif "radiometer-sst" in variable_key:
                return "radiometer-sst"
            elif "ship-sst" in variable_key:
                return "ship-sst"
            elif "xbt-sst" in variable_key:
                return "xbt-sst"

        raise IOError("unsupported data format")
