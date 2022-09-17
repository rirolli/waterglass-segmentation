import torch
import os
from torch.utils.data import Dataset
from skimage import io
import numpy as np

class ImagesDataset(Dataset):
    def __init__(self, x_path, y_vessel_path, y_filled_path, x_transforms=None, y_transforms=None) -> None:
        super().__init__()

        # Get the path of images and labels
        self.x_path = x_path
        self.y_vessel_path = y_vessel_path
        self.y_filled_path = y_filled_path

        # Get the images and labels list name
        x_images = [e.replace("jpg","png") for e in os.listdir(self.x_path)]
        y_vessel_images = [e for e in os.listdir(self.y_vessel_path)]
        y_filled_images = [e for e in os.listdir(self.y_filled_path)]

        # Gets list of file that are in all the directory
        common_images = sorted([e for e in x_images if e in y_vessel_images and e in y_filled_images])

        # Cleaned images and labels list name
        self.x_images = [e.replace("png","jpg") for e in common_images]
        self.y_images = common_images

        if len(self.x_images) != len(self.y_images):
            raise Exception(f"The length of X ({len(self.x_images)}) does not match the length of Y ({len(self.y_images)}).")

        # Transforms
        self.transforms = {"x_transforms":x_transforms, "y_transforms":y_transforms}

    def __len__(self):
        return len(self.x_images)

    def __getitem__(self, index: int):
        x_path_joined = os.path.join(self.x_path, self.x_images[index])
        y_vessel_path_joined = os.path.join(self.y_vessel_path, self.y_images[index])
        y_filled_path_joined = os.path.join(self.y_filled_path, self.y_images[index])

        sample_x = io.imread(x_path_joined)
        sample_y_vessel = io.imread(y_vessel_path_joined)
        sample_y_filled = io.imread(y_filled_path_joined)

        sample_y = self._mergeVesselFilledImages(vessel_img=sample_y_vessel, filled_img=sample_y_filled)

        if self.transforms["x_transforms"] is not None:
            sample_x = self.transforms["x_transforms"](sample_x)
        if self.transforms["y_transforms"] is not None:
            sample_y = self.transforms["y_transforms"](sample_y)

        return sample_x, sample_y

    def _mergeVesselFilledImages(self, vessel_img, filled_img):
        merged_img = np.zeros(vessel_img.shape[0:2], np.float32)
        if vessel_img is not None:
            merged_img[vessel_img==1] = 1
        if filled_img is not None:
            merged_img[filled_img==1] = 2
        return merged_img