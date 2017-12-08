import unittest

import xarray as xr
from xarray import Variable

from dmi.sst.mw_oe.flag_coding import FlagCoding
from dmi.sst.mw_oe.rfi_processor import RfiProcessor

NUM_MATCHES = 12


class RfiProcessorTest(unittest.TestCase):
    def setUp(self):
        self.dataset = xr.Dataset()

    def test_find_rfi(self):
        sat_lon = [128.6643, -171.7969, -80.4683, -19.5940, -19.2279, 166.5237, -2.9735, -174.1847, 174.9277, 171.3845, 171.3845, 6.0]
        self.dataset["amsre.longitude"] = Variable(["matchup_count"], sat_lon)

        sat_lat = [24.1321, 36.5892, -18.7240, -25.0698, -13.3672, -52.1502, 69.3439, 11.3338, -13.4490, -16.0562, -16.0562, 64.0]
        self.dataset["amsre.latitude"] = Variable(["matchup_count"], sat_lat)

        geo_refl_lon = [75.910, -111.240, -111.920, -53.200, -10.180, -104.260, -117.880, 170.480, 125.520, -170.290, 236.0, -170.290]
        self.dataset["amsre.Geostationary_Reflection_Longitude"] = Variable(["matchup_count"], geo_refl_lon)

        geo_refl_lat = [23.0600, 77.3500, -61.4200, -68.8300, 33.7200, -56.7100, 53.7500, 57.8800, -26.3100, 28.3900, -20.3, 28.3900]
        self.dataset["amsre.Geostationary_Reflection_Latitude"] = Variable(["matchup_count"], geo_refl_lat)

        i_asc = [1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0]
        self.dataset["amsre.ascending"] = Variable(["matchup_count"], i_asc)

        flag_coding = FlagCoding(12)
        rfi_processor = RfiProcessor()
        rfi_processor.find_rfi(self.dataset, flag_coding)

        flags = flag_coding.get_flags()
        for i in range(0, 9):
            self.assertEqual(0, flags[i])

        self.assertEqual(512, flags[10])
        self.assertEqual(512, flags[11])
