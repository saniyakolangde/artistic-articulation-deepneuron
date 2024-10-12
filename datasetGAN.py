import os
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, path_to_dataset, transform=None):
        self.path_to_dataset = path_to_dataset
        self.transform = transform

        self.directory = os.listdir(self.path_to_dataset)
        self.length_of_dset = len(self.directory)

    def __len__(self):
        return self.length_of_dset

    def __getitem__(self, index):
        image_filename = self.directory[index]
        path_to_image = os.path.join(self.path_to_dataset, image_filename)
        image = Image.open(path_to_image).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image






