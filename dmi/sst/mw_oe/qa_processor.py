import numpy as np

BT_MIN = 0.0
BT_MAX = 320.0
WS_MIN = 0.0
WS_MAX = 20.0
LON_ABS_MAX = 180.0
LAT_ABS_MAX = 90.0
SZA_MIN = 0.0
SZA_MAX = 180.0
SST_MIN = -2.0
SST_MAX = 40.0


class QaProcessor():
    def run_qa(self, dataset):
        self.run_qa_general(dataset)
        self.run_qa_bt_delta(dataset)

    def run_qa_general(self, dataset):
        flag_array = dataset["invalid_data"].data
        for variable_name in dataset.variables:
            variable_data = dataset[variable_name].data

            if "data_quality" in variable_name or "_flag" in variable_name:
                local_mask = variable_data != 0
                flag_array = np.logical_or(flag_array, local_mask)
                continue

            if "brightness_temperature" in variable_name:
                local_mask = (variable_data <= BT_MIN) | (variable_data >= BT_MAX)
                flag_array = np.logical_or(flag_array, local_mask)
                continue

            if "abs_wind_speed" in variable_name:
                local_mask = (variable_data < WS_MIN) | (variable_data > WS_MAX)
                flag_array = np.logical_or(flag_array, local_mask)
                continue

            if "longitude" in variable_name:
                local_mask = np.absolute(variable_data) > LON_ABS_MAX
                flag_array = np.logical_or(flag_array, local_mask)
                continue

            if "latitude" in variable_name:
                local_mask = np.absolute(variable_data) > LAT_ABS_MAX
                flag_array = np.logical_or(flag_array, local_mask)
                continue

            if "solar_zenith_angle" in variable_name:
                local_mask = (variable_data < SZA_MIN) | (variable_data > SZA_MAX)
                flag_array = np.logical_or(flag_array, local_mask)
                continue

            if "sea_surface_temperature" in variable_name:
                local_mask = (variable_data < SST_MIN) | (variable_data > SST_MAX)
                flag_array = np.logical_or(flag_array, local_mask)
                continue

        dataset["invalid_data"].data = flag_array

    def run_qa_bt_delta(self, dataset):
        flag_array = dataset["invalid_data"].data

        data_18V = dataset["amsre.brightness_temperature18V"].data
        data_18H = dataset["amsre.brightness_temperature18H"].data
        local_mask = (data_18V < data_18H)
        flag_array = np.logical_or(flag_array, local_mask)

        data_23V = dataset["amsre.brightness_temperature23V"].data
        data_23H = dataset["amsre.brightness_temperature23H"].data
        local_mask = (data_23V < data_23H)
        flag_array = np.logical_or(flag_array, local_mask)

        data_36V = dataset["amsre.brightness_temperature36V"].data
        data_36H = dataset["amsre.brightness_temperature36H"].data
        local_mask = (data_36V < data_36H)
        flag_array = np.logical_or(flag_array, local_mask)

        dataset["invalid_data"].data = flag_array
        pass
