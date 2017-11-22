import xarray as xr

from dmi.sst.mw_oe.constants import INPUT_VARIABLES
from dmi.sst.util.default_data import DefaultData

OFFSET = "OFFSET"
SCALE_FACTOR = "SCALE_FACTOR"


class MmdReader:
    input_data = None

    def read(self, input_file):
        # xarray can not handle the TAI 1993 time coding @todo 3 tb/th adapt if possible
        self.input_data = xr.open_dataset(input_file, decode_times=False)

        subset_data = xr.Dataset()

        for variable_name in INPUT_VARIABLES:
            target_variable_name = variable_name
            if "{IS_SENSOR}" in variable_name:
                in_situ_sensor = self._get_insitu_sensor(self.input_data)
                variable_name = variable_name.replace("{IS_SENSOR}", in_situ_sensor)
                target_variable_name = variable_name[len(in_situ_sensor) + 1: len(variable_name)]
            variable = self.input_data.variables[variable_name]

            if SCALE_FACTOR in variable.attrs or OFFSET in variable.attrs:
                MmdReader._scale_data(variable)

            MmdReader._add_fill_values(variable, target_variable_name)

            subset_data[target_variable_name] = variable

        return subset_data

    def close(self):
        self.input_data.close()

    @staticmethod
    def _scale_data(variable):
        scale_factor, offset = MmdReader.get_scale_and_offset(variable)
        if scale_factor != 1.0 or offset != 0.0:
            scaled_data = variable.data * scale_factor + offset
            variable.data = scaled_data

    @staticmethod
    def get_scale_and_offset(variable):
        scale_factor = 1.0
        if SCALE_FACTOR in variable.attrs:
            scale_factor = variable.attrs[SCALE_FACTOR]
        offset = 0.0
        if OFFSET in variable.attrs:
            offset = variable.attrs[OFFSET]
        return scale_factor, offset

    @staticmethod
    def _get_insitu_sensor(input_data):
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

    @staticmethod
    def _add_fill_values(variable, target_variable_name):
        if "brightness_temperature" in target_variable_name:
            offset = variable.attrs["OFFSET"]
            scale_factor = variable.attrs["SCALE_FACTOR"]
            variable.attrs["_FillValue"] = -32768.0 * scale_factor + offset
            return

        if "insitu." in target_variable_name:
            variable.attrs["_FillValue"] = -32768
            return

        if "source" in variable.attrs:
            variable.attrs["_FillValue"] = 2e20
            return

        if not "FillValue" in variable.attrs:
            default_fill = DefaultData.get_default_fill_value(variable.dtype)
            scale_factor, offset = MmdReader.get_scale_and_offset(variable)
            if scale_factor != 1.0 or offset != 0.0:
                variable.attrs["_FillValue"] = default_fill * scale_factor + offset
            else:
                variable.attrs["_FillValue"] = default_fill
            return
