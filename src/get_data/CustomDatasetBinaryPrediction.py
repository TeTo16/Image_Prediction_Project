from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch


class CustomDatasetBinaryPrediction(Dataset):
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

        if label == 0:
            label = np.array([1., 0.])
        elif label == 1:
            label = np.array([0., 1.])
        else:
            raise ValueError("El valor debe ser 0 o 1.")

        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        image, label = image.to(self.device), label.to(self.device)

        return image, label
