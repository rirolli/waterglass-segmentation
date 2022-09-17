import torch
import torch.jit
import torch.nn as nn
import torchvision.transforms as tf
from itertools import repeat
from dataset_trasformer_abstract import DatasetTransformer


def checkIfTuple(e, n:int):
    if type(e) is tuple:
        if len(e)==n:
            return e
        else:
            raise Exception("Dimension of the tuple not valid.")
    else:
        return tuple(repeat(e, n))

class ImagesTransforms:

    def __init__(self) -> None:
        pass

    def _getTransformImage(self, height, width):
        """Function to transform the sample images"""
        transforms = nn.Sequential(
            tf.ToPILImage(),
            tf.Resize((height, width)),
            tf.ToTensor(),
            tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # according to https://towardsdatascience.com/train-neural-net-for-semantic-segmentation-with-pytorch-in-50-lines-of-code-830c71a6544f
        )

        return torch.jit.script(transforms)

    def _getTransformLabel(self, height, width):
        """Function to transform the sample label"""
        transforms = nn.Sequential(
            tf.ToPILImage(),
            tf.Resize((height, width)),
            tf.ToTensor()
        )

        return torch.jit.script(transforms)

    def getTransformedDataset(self, x, y, window, batch_size=3, train_test_split=False, test_size=0.2):
        """Returns a transformed dataset.
        Parameters:
        - x """
        _window = checkIfTuple(window)
        height = _window[0]
        width = _window[1]

        transformer_images = self._getTransformImage(window[0], window[1])
        transformer_labels = self._getTransformLabel(window[0], window[1])

        x_transformed = transformer_images(x)
        y_transformed = transformer_labels(y)