INPUT_VARIABLES = ["amsre.latitude",                        # float, degrees north
                   "amsre.longitude",                       # float, degrees east
                   "amsre.time",                            # double, s, TAI 1993 format
                   "amsre.solar_zenith_angle",              # float, degree
                   "amsre.satellite_zenith_angle",          # float, degree
                   "amsre.land_ocean_flag_6",               # int8, fill=255
                   "amsre.brightness_temperature6V",        # float, Kelvin
                   "amsre.brightness_temperature6H",        # float, Kelvin
                   "amsre.brightness_temperature10V",       # float, Kelvin
                   "amsre.brightness_temperature10H",       # float, Kelvin
                   "amsre.brightness_temperature18V",       # float, Kelvin
                   "amsre.nwp.seaice_fraction",             # float, fill=2E20
                   "amsre.nwp.sea_surface_temperature",     # float, Kelvin, fill=2E20
                   "amsre.nwp.10m_east_wind_component",     # float, m/s
                   "amsre.nwp.10m_north_wind_component",    # float, m/s
                   "amsre.nwp.skin_temperature",            # float, Kelvin
                   "amsre.nwp.log_surface_pressure",        # float
                   "amsre.nwp.cloud_liquid_water",          # float, kg/kg
                   "amsre.nwp.total_column_water_vapour",   # float, kg/m^2
                   "amsre.nwp.total_precip",                # float, m
                   #"drifter-sst_insitu.time",              # int32, s, seconds since 1978-01-01, fill=-32768
                   #"drifter-sst_insitu.sea_surface_temperature", # float, Celsius, fill=-32768.0
                   #"drifter-sst_insitu.sst_qc_flag",       # int16, fill=-32768
                   #"drifter-sst_insitu.sst_track_flag",    # int16, fill=-32768
                   ]