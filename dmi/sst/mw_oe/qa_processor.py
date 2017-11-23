import numpy as np


class QaProcessor():

    def run_qa(self, dataset):
        flag_array = dataset["invalid_data"].data
        for variable_name in dataset.variables:
            variable_data = dataset[variable_name].data

            if "data_quality" in variable_name or "_flag" in variable_name:
                local_mask = variable_data != 0
                flag_array = np.logical_or(flag_array, local_mask)
                continue

            if "brightness_temperature" in variable_name:
                local_mask = (variable_data <= 0.0) | (variable_data >= 320.0)
                flag_array = np.logical_or(flag_array, local_mask)
                continue

            if "wind_component" in variable_name:
                local_mask = np.absolute(variable_data) > 50.0
                flag_array = np.logical_or(flag_array, local_mask)
                continue

            if "longitude" in variable_name:
                local_mask = np.absolute(variable_data) > 180.0
                flag_array = np.logical_or(flag_array, local_mask)
                continue

            if "latitude" in variable_name:
                local_mask = np.absolute(variable_data) > 90.0
                flag_array = np.logical_or(flag_array, local_mask)
                continue

        dataset["invalid_data"].data = flag_array