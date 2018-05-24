import numpy as np

class BtBiasCorrection:
    BT_VARIABLE_NAMES = ["amsre.brightness_temperature6V", "amsre.brightness_temperature6H", "amsre.brightness_temperature10V", "amsre.brightness_temperature10H", "amsre.brightness_temperature18V",
                         "amsre.brightness_temperature18H", "amsre.brightness_temperature23V", "amsre.brightness_temperature23H", "amsre.brightness_temperature36V", "amsre.brightness_temperature36H"]

    BT_BIAS = np.float64([0.3524, 0.0793, -0.1093, -0.7521, 0.6228, 0.2794, 0.0200, -0.2669, -0.3540, 0.1464])

    def run(self, dataset):
        for i in range(0, len(self.BT_VARIABLE_NAMES)):
            data = dataset[self.BT_VARIABLE_NAMES[i]]
            bias_corrected = data + self.BT_BIAS[i]
            dataset[self.BT_VARIABLE_NAMES[i]] = bias_corrected


