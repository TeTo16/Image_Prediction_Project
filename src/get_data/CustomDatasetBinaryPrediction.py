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
        # label = torch.tensor(label, dtype=torch.float32)
        # label = label.float()

        label_tensor = torch.zeros(2)
        label_tensor[float(label)] = 1

        if self.transform:
            image = self.transform(image)

        image, label_tensor = image.to(self.device), label_tensor.to(self.device)

        return image, label
