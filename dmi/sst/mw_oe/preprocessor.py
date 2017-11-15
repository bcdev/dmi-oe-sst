import numpy as np
import xarray as xr


class Preprocessor:
    TO_SQUEEZE_NAMES = ["insitu.time", "insitu.lat", "insitu.lon", "insitu.sea_surface_temperature", "insitu.sst_depth", "insitu.sst_qc_flag", "insitu.sst_track_flag]"]
    TO_AVERAGE_NAMES = ["amsre.brightness_temperature6V", "amsre.brightness_temperature6H", "amsre.brightness_temperature10V", "amsre.brightness_temperature10H", "amsre.brightness_temperature18V",
                        "amsre.brightness_temperature18H", "amsre.brightness_temperature23V", "amsre.brightness_temperature23H", "amsre.brightness_temperature36V", "amsre.brightness_temperature36H"]
    TO_CENTER_EXTRACT_NAMES = ["amsre.nwp.seaice_fraction", "amsre.nwp.sea_surface_temperature", "amsre.nwp.10m_east_wind_component", "amsre.nwp.10m_north_wind_component",
                               "amsre.nwp.skin_temperature", "amsre.nwp.log_surface_pressure", "amsre.nwp.cloud_liquid_water", "amsre.nwp.total_column_water_vapour", "amsre.nwp.total_precip"]

    def run(self, dataset):

        preprocessed_data = xr.Dataset()
        num_matchups = len(dataset.coords["matchup_count"])
        preprocessed_data["mask_vector"] = np.zeros(num_matchups)

        for variable_name in dataset.variables:
            if variable_name in self.TO_SQUEEZE_NAMES:
                preprocessed_data[variable_name] = dataset.variables[variable_name].squeeze()
                continue

            if variable_name in self.TO_AVERAGE_NAMES:
                for i in range(0, num_matchups - 1):
                    layer = dataset.variables[variable_name][i, :, :]
                continue

            if variable_name in self.TO_CENTER_EXTRACT_NAMES:
                # @todo 3 tb/tb read dimensions from variable and dynamically calculate center column 2017-11-15
                preprocessed_data[variable_name] = dataset.variables[variable_name][:, 3, 3].squeeze()
                continue

        return preprocessed_data
