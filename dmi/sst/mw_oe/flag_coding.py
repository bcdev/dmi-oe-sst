import numpy as np


# This class contaisn the flag coding for the DMI_OE_SST processor. Flags defined are:
#
# Name                  Bit Value   Description
# avg_inv_thresh        0   1       Input discarded as more that 10 percent of the brightness temperature values to average contain _FillValue
# amsre_flag            1   2       AMSR-E input data flag "pixel_data_qualityXX" raised
# bt_out_of_range       2   4       AMSR-E brightness temperature out of valid range (0, 320)K
# ws_out_of_range       3   8       In-Situ wind-speed out of valid range (0, 20)m/s
# inv_geolocation       4   16      AMSR-E geolocation data invalid
# sza_out_of_range      5   32      AMSR-E sun zenith angle out of valid range (0, 180)deg
# sst_out_of_range      6   64      EraInterim NWP or in-situ SST values out of range (-2, 40)Celsius
# bt_pol_test_failed    7   128     AMSR-E brightness temperature polarization test failed (BTv < BTh)
# inv_file_name         8   256     AMSR-E filename in MMD is not following conventions, cannot extract ascending/descending infomation
# rfi_possible          9   512     Pixel is located in a RFI contaminated area

class FlagCoding():
    flags = None

    AVG_INV_THRESH = 1
    AMSRE_FLAG = 2
    BT_OUT_OF_RANGE = 4
    WS_OUT_OF_RANGE = 8
    INV_GEOLOCATION = 16
    SZA_OUT_OF_RANGE = 32
    SST_OUT_OF_RANGE = 64
    BT_POL_TEST_FAILED = 128
    INV_FILE_NAME = 256
    RFI_POSSIBLE = 512

    def __init__(self, num_samples):
        self.flags = np.zeros(num_samples, dtype=np.int16)

    def get_flags(self):
        return self.flags

    @staticmethod
    def get_flag_masks():
        return "1 2 4 8 16 32 64 128 256 512"

    @staticmethod
    def get_flag_meanings():
        return "avg_inv_thresh amsre_flag bt_out_of_range ws_out_of_range inv_geolocation sza_out_of_range sst_out_of_range bt_pol_test_failed inv_file_name rfi_possible"

    def add_avg_inv_thresh(self, tags):
        self._add_flag(tags, self.AVG_INV_THRESH)

    def add_amsre_flag(self, tags):
        self._add_flag(tags, self.AMSRE_FLAG)

    def add_bt_out_of_range(self, tags):
        self._add_flag(tags, self.BT_OUT_OF_RANGE)

    def add_ws_out_of_range(self, tags):
        self._add_flag(tags, self.WS_OUT_OF_RANGE)

    def add_invalid_geolocation(self, tags):
        self._add_flag(tags, self.INV_GEOLOCATION)

    def add_sza_out_of_range(self, tags):
        self._add_flag(tags, self.SZA_OUT_OF_RANGE)

    def add_sst_out_of_range(self, tags):
        self._add_flag(tags, self.SST_OUT_OF_RANGE)

    def add_bt_pol_test_failed(self, tags):
        self._add_flag(tags, self.BT_POL_TEST_FAILED)

    def add_inv_filename(self, tags):
        self._add_flag(tags, self.INV_FILE_NAME)

    def add_rfi_possible(self, tags):
        self._add_flag(tags, self.RFI_POSSIBLE)

    def _add_flag(self, tags, flag_value):
        self.flags = np.bitwise_or(self.flags, tags.astype(np.int16) * flag_value)
