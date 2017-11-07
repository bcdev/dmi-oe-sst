import xarray as xr


class Preprocessor:
    TO_SQUEEZE_NAMES = ["insitu.time", "insitu.lat", "insitu.lon", "insitu.sea_surface_temperature", "insitu.sst_depth", "insitu.sst_qc_flag", "insitu.sst_track_flag]"]

    def run(self, dataset):

        preprocessed_data = xr.Dataset()

        for variable_name in dataset.variables:
            if variable_name in self.TO_SQUEEZE_NAMES:
                preprocessed_data[variable_name] = dataset.variables[variable_name].squeeze()


        return preprocessed_data
