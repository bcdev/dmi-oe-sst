import numpy as np

from dmi.sst.mw_oe.rfi_processor import RfiProcessor

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
DIURNAL_WS_MAX = 4
DIURNAL_SZA_MAX = 90
RAIN_BT_THRESHOLD = 240.0


class QaProcessor():
    rfi_pocessor = None

    def __init__(self):
        self.rfi_pocessor = RfiProcessor()

    def run_qa(self, dataset, flag_coding):
        self.run_qa_general(dataset, flag_coding)
        self.run_qa_bt_delta(dataset, flag_coding)
        self.run_qa_rfi_detection(dataset, flag_coding)
        self.run_qa_diurnal_warming(dataset, flag_coding)
        self.run_qa_rain_detection(dataset, flag_coding)

    def run_qa_general(self, dataset, flag_coding):
        for variable_name in dataset.variables:
            variable_data = dataset[variable_name].data

            if "data_quality" in variable_name or "_flag" in variable_name:
                local_mask = variable_data != 0
                flag_coding.add_amsre_flag(local_mask)
                continue

            if "brightness_temperature" in variable_name:
                local_mask = (variable_data <= BT_MIN) | (variable_data >= BT_MAX)
                flag_coding.add_bt_out_of_range(local_mask)
                continue

            if "abs_wind_speed" in variable_name:
                local_mask = (variable_data < WS_MIN) | (variable_data > WS_MAX)
                flag_coding.add_ws_out_of_range(local_mask)
                continue

            if "longitude" in variable_name:
                local_mask = np.absolute(variable_data) > LON_ABS_MAX
                flag_coding.add_invalid_geolocation(local_mask)
                continue

            if "latitude" in variable_name:
                local_mask = np.absolute(variable_data) > LAT_ABS_MAX
                flag_coding.add_invalid_geolocation(local_mask)
                continue

            if "solar_zenith_angle" in variable_name:
                local_mask = (variable_data < SZA_MIN) | (variable_data > SZA_MAX)
                flag_coding.add_sza_out_of_range(local_mask)
                continue

            if "sea_surface_temperature" in variable_name:
                local_mask = (variable_data < SST_MIN) | (variable_data > SST_MAX)
                flag_coding.add_sst_out_of_range(local_mask)
                continue

    def run_qa_bt_delta(self, dataset, flag_coding):
        data_18V = dataset["amsre.brightness_temperature18V"].data
        data_18H = dataset["amsre.brightness_temperature18H"].data
        local_mask = (data_18V < data_18H)
        flag_coding.add_bt_pol_test_failed(local_mask)

        data_23V = dataset["amsre.brightness_temperature23V"].data
        data_23H = dataset["amsre.brightness_temperature23H"].data
        local_mask = (data_23V < data_23H)
        flag_coding.add_bt_pol_test_failed(local_mask)

        data_36V = dataset["amsre.brightness_temperature36V"].data
        data_36H = dataset["amsre.brightness_temperature36H"].data
        local_mask = (data_36V < data_36H)
        flag_coding.add_bt_pol_test_failed(local_mask)

    def run_qa_rfi_detection(self, dataset, flag_coding):
        self.rfi_pocessor.find_rfi(dataset, flag_coding)

    def run_qa_diurnal_warming(self, dataset, flag_coding):
        wind_speed = dataset["amsre.nwp.abs_wind_speed"].data
        sza = dataset["amsre.solar_zenith_angle"].data
        local_mask = (wind_speed < DIURNAL_WS_MAX) & (np.abs(sza) < DIURNAL_SZA_MAX)
        flag_coding.add_diurnal_warming(local_mask)

    def run_qa_rain_detection(self, dataset, flag_coding):
        bt_18 = dataset["amsre.brightness_temperature18V"].data
        local_mask = (bt_18 >= RAIN_BT_THRESHOLD)
        flag_coding.add_rain_possible(local_mask)
