from distutils.command.config import config
from unet import UNet
from jproperties import Properties
from dataset import ImagesDataset
import os
from skimage import io
from matplotlib import pyplot as plt


def readPropertied():
    configs = Properties()
    with open('properties.properties', 'rb') as config_file:
        configs.load(config_file)
    return configs


def main ():
    configs = readPropertied()

    model = UNet(n_channels=3, n_classes=2)
    # print(model)

    image_path = os.path.join(configs.get("TRAIN_FOLDER_SIMPLE").data, configs.get("TRAIN_FOLDER_SIMPLE_IMAGE").data)
    vessel_path = os.path.join(configs.get("TRAIN_FOLDER_SIMPLE").data, configs.get("TRAIN_FOLDER_SIMPLE_VESSEL").data)
    filled_path = os.path.join(configs.get("TRAIN_FOLDER_SIMPLE").data, configs.get("TRAIN_FOLDER_SIMPLE_FILLED").data)
    print(image_path)

    db = ImagesDataset(x_path=image_path, y_vessel_path=vessel_path, y_filled_path=filled_path)
    x,y = db.__getitem__(1)
    print (len(db))
    io.imshow(x)
    plt.show()
    io.imshow(y)
    plt.show()

if __name__ == "__main__":
    main()