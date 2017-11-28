import numpy as np


class RfiProcessor:
    # @todo 3 tb/tb improve performance_ shift location arrays to -180/+180 range 2017-11-28
    xlon_reflected = np.float64([[235, 279], [316, 346], [340, 360], [0, 5], [0, 50], [350, 360], [0, 10], [340, 360], [0, 5], [4, 25], [289, 305], [311, 327], [295, 315], [4, 25]])
    xlat_reflected = np.float64([[-22, 19], [-16, 10], [-15, 20], [-14, 9], [-30, 20], [-10, 14], [-10, 14], [-15, 4], [-15, 4], [-10, 13], [-8, 12], [-4, 8], [-7, 10], [-10, 13]])
    asc_reflected = np.int8([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    xlon_ground = np.float64([[1, 4], [5, 9], [3, 5], [11, 13], [70, 73], [1, 4], [5, 9], [3, 5], [2, 4], [70, 73], [11, 13]])
    xlat_ground = np.float64([[58, 62], [63, 67], [52, 55], [34.5, 37], [17.5, 21], [58, 62], [63, 67], [52, 55], [56, 58], [17.5, 21], [34.5, 37]])
    asc_ground = np.int8([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0])

    num_reflected = len(xlon_reflected)
    num_ground = len(xlon_ground)

    def find_rfi(self, dataset):
        sat_lon_data = dataset["amsre.longitude"].data
        sat_lat_data = dataset["amsre.latitude"].data
        geo_lon_data = dataset["amsre.Geostationary_Reflection_Longitude"].data
        geo_lat_data = dataset["amsre.Geostationary_Reflection_Latitude"].data
        asc_data = dataset["amsre.ascending"].data
        flag_array = dataset["invalid_data"].data

        num_matchups = len(sat_lon_data)

        for i in range(0, num_matchups):
            sat_lon = sat_lon_data[i]
            sat_lat = sat_lat_data[i]
            if sat_lon < 0.0:
                sat_lon += 360.0

            geo_lon = geo_lon_data[i]
            geo_lat = geo_lat_data[i]
            if geo_lon < 0.0:
                geo_lon += 360.0

            asc = asc_data[i]

            # check if reflected vector at geostationary sat location
            for j in range(0, self.num_reflected):
                if asc == self.asc_reflected[j]:
                    lon_range = self.xlon_reflected[j]
                    lat_range = self.xlat_reflected[j]
                    if (geo_lon >= lon_range.min()) & (geo_lon <= lon_range.max()) & (geo_lat >= lat_range.min()) & (geo_lat <= lat_range.max()):
                        flag_array[i] = True
                        break

            # check if observation location at groundbased rfi location
            for j in range(0, self.num_ground):
                if asc == self.asc_ground[j]:
                    lon_range = self.xlon_ground[j]
                    lat_range = self.xlat_ground[j]
                    if (sat_lon >= lon_range.min()) & (sat_lon <= lon_range.max()) & (sat_lat >= lat_range.min()) & (sat_lat <= lat_range.max()):
                        flag_array[i] = True
                        break

        dataset["invalid_data"].data = flag_array
