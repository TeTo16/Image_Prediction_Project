from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch


class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None, device=None):
        self.data = dataframe
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pixels = self.data.iloc[idx].iloc[1]
        image = Image.fromarray(np.fromstring(pixels, dtype=int, sep=' ').reshape(48, 48).astype(np.uint8))
        image = image.convert('RGB')
        label = self.data.iloc[idx].iloc[0]
        label = torch.tensor(label, dtype=torch.uint8)

        if self.transform:
            image = self.transform(image)

        image, label = image.to(self.device), label.to(self.device)

        return image, label
