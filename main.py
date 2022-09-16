from distutils.command.config import config
from unet import UNet
from jproperties import Properties

def readPropertied():
    configs = Properties()
    with open('properties.properties', 'rb') as config_file:
        configs.load(config_file)
    return configs


def main ():
    configs = readPropertied()

    # print(configs.get("TRAIN_FOLDER_SIMPLE").data)

    model = UNet(n_channels=3, n_classes=2)
    print(model)

if __name__ == "__main__":
    main()