import torch
import torch.jit
import torch.nn as nn
import torchvision.transforms as tf

class ImagesTransforms:

    def __init__(self, height, width, train_folder, test_folder, batch_size=3) -> None:
        self.height = height
        self.width = width

        self.transformImages = self._getTransformImage()
        self.transformLabels = self._getTransformLabel()

        self.train_folder = train_folder
        self.test_folder = test_folder

    def _getTransformImage(self):
        transforms = nn.Sequential(
            tf.ToPILImage(),
            tf.Resize((self.height, self.width)),
            tf.ToTensor(),
            tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        )

        return torch.jit.script(transforms)

    def _getTransformLabel(self):
        transforms = nn.Sequential(
            tf.ToPILImage(),
            tf.Resize((self.height, self.width)),
            tf.ToTensor()
        )

        return torch.jit.script(transforms)