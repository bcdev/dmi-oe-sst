INPUT_VARIABLES = ["amsre.latitude",                        # float, degrees north, fill: default
                   "amsre.longitude",                       # float, degrees east, fill: default
                   "amsre.time",                            # double, s, TAI 1993 format, fill: default
                   "amsre.solar_zenith_angle",              # float, degree, fill: default
                   "amsre.satellite_zenith_angle",          # float, degree, fill: default
                   "amsre.satellite_azimuth_angle",         # float, degree, fill: default
                   "amsre.land_ocean_flag_6",               # int8, fill=255
                   "amsre.brightness_temperature6V",        # float, Kelvin, fill=-32768
                   "amsre.brightness_temperature6H",        # float, Kelvin, fill=-32768
                   "amsre.brightness_temperature10V",       # float, Kelvin, fill=-32768
                   "amsre.brightness_temperature10H",       # float, Kelvin, fill=-32768
                   "amsre.brightness_temperature18V",       # float, Kelvin, fill=-32768
                   "amsre.brightness_temperature18H",       # float, Kelvin, fill=-32768
                   "amsre.brightness_temperature23V",       # float, Kelvin, fill=-32768
                   "amsre.brightness_temperature23H",       # float, Kelvin, fill=-32768
                   "amsre.brightness_temperature36V",       # float, Kelvin, fill=-32768
                   "amsre.brightness_temperature36H",       # float, Kelvin, fill=-32768
                   "amsre.pixel_data_quality6V",            # int16, fill=-32767
                   "amsre.pixel_data_quality6H",            # int16, fill=-32767
                   "amsre.pixel_data_quality10V",           # int16, fill=-32767
                   "amsre.pixel_data_quality10H",           # int16, fill=-32767
                   "amsre.pixel_data_quality18V",           # int16, fill=-32767
                   "amsre.pixel_data_quality18H",           # int16, fill=-32767
                   "amsre.pixel_data_quality23V",           # int16, fill=-32767
                   "amsre.pixel_data_quality23H",           # int16, fill=-32767
                   "amsre.pixel_data_quality36V",           # int16, fill=-32767
                   "amsre.pixel_data_quality36H",           # int16, fill=-32767
                   "amsre.scan_data_quality",               # int32, fill=-2147483647
                   "amsre.Geostationary_Reflection_Latitude", # float, degree, fill=-32767
                   "amsre.Geostationary_Reflection_Longitude", # float, degree, fill=-32767
                   "amsre.nwp.seaice_fraction",             # float, fill=2E20                                  pre: extract_center
                   "amsre.nwp.sea_surface_temperature",     # float, Kelvin, fill=2E20                          pre: extract_center
                   "amsre.nwp.10m_east_wind_component",     # float, m/s, fill=2E20                             pre: extract_center
                   "amsre.nwp.10m_north_wind_component",    # float, m/s, fill=2E20                             pre: extract_center
                   "amsre.nwp.skin_temperature",            # float, Kelvin, fill=2E20                          pre: extract_center
                   "amsre.nwp.log_surface_pressure",        # float, fill=2E20                                  pre: extract_center
                   "amsre.nwp.cloud_liquid_water",          # float, kg/kg, fill=2E20                           pre: extract_center
                   "amsre.nwp.total_column_water_vapour",   # float, kg/m^2, fill=2E20                          pre: extract_center
                   "amsre.nwp.total_precip",                # float, m, fill=2E20                               pre: extract_center
                   "{IS_SENSOR}_insitu.time",               # int32, s, seconds since 1978-01-01, fill=-32768   pre: squeeze
                   "{IS_SENSOR}_insitu.lat",                # float, degrees north,fill=-32768.0                pre: squeeze
                   "{IS_SENSOR}_insitu.lon",                # float, degrees east,fill=-32768.0                 pre: squeeze
                   "{IS_SENSOR}_insitu.sst_depth",          # float, m ,fill=-32768.0                           pre: squeeze
                   "{IS_SENSOR}_insitu.sea_surface_temperature", # float, Celsius, fill=-32768.0                pre: squeeze
                   "{IS_SENSOR}_insitu.sst_qc_flag",       # int16, fill=-32768                                 pre: squeeze
                   "{IS_SENSOR}_insitu.sst_track_flag",    # int16, fill=-32768                                 pre: squeeze
                   ]