import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class ButterflyDataset(Dataset):
    def __init__(self, file_path, base_dir, transform=None):
        self.data = self._load_data(file_path)
        self.base_dir = base_dir
        self.transform = transform

    def _load_data(self, file_path):
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        data = []
        for line in lines:
            parts = line.strip().split()
            img_path = parts[0]  # 
            label = int(parts[1])  # 
            data.append((img_path, label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        full_img_path = os.path.join(self.base_dir, img_path)

        
        image = Image.open(full_img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label