import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import random # want to match to different images every time

class HumanDogDataset(Dataset):
    def __init__(self, path_dset1, path_dset2, transform=None):
        self.dset1 = path_dset1
        self.dset2 = path_dset2
        self.transform = transform

        self.dset1_dir = os.listdir(path_dset1)
        self.dset1_len = len(self.dset1_dir)

        self.dset2_dir = os.listdir(path_dset2)
        self.dset2_len = len(self.dset2_dir)

        self.dataset_length = max(self.dset1_len, self.dset2_len)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        dset1_image_filename= self.dset1_dir[index % self.dset1_len] # wrap around if index is greater than length of human image list
        dset2_image_filename = self.dset2_dir[random.randint(0,self.dset2_len-1)] # use random image to match

        path_to_dset1 = os.path.join(self.dset1, dset1_image_filename)
        path_to_dset2 = os.path.join(self.dset2, dset2_image_filename)

        dset1_img = Image.open(path_to_dset1).convert("RGB")
        dset2_img = Image.open(path_to_dset2).convert("RGB")

        if self.transform:
            dset1_img = self.transform(dset1_img)
            dset2_img = self.transform(dset2_img)

        return dset1_img, dset2_img






