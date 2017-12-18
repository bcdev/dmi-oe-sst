import numpy as np

class BtBiasCorrection:
    BT_VARIABLE_NAMES = ["amsre.brightness_temperature6V", "amsre.brightness_temperature6H", "amsre.brightness_temperature10V", "amsre.brightness_temperature10H", "amsre.brightness_temperature18V",
                         "amsre.brightness_temperature18H", "amsre.brightness_temperature23V", "amsre.brightness_temperature23H", "amsre.brightness_temperature36V", "amsre.brightness_temperature36H"]

    BT_BIAS = np.float64([0.5234, 0.1990, -0.0192, -0.7531, 0.6020, 0.1908, 0.0088, -0.3325, -0.3346, 0.0575])

    def run(self, dataset):
        for i in range(0, len(self.BT_VARIABLE_NAMES)):
            data = dataset[self.BT_VARIABLE_NAMES[i]]
            bias_corrected = data + self.BT_BIAS[i]
            dataset[self.BT_VARIABLE_NAMES[i]] = bias_corrected


