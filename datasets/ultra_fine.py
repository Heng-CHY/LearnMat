import os
from torch.utils.data import Dataset
from PIL import Image

class SoybeanDataset(Dataset):
    def __init__(self, image_dir, anno_dir, split='train', transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = []
        
        if split == 'train':
            self._load_annotations(os.path.join(anno_dir, 'train.txt'))
            
        elif split == 'test':
            self._load_annotations(os.path.join(anno_dir, 'test.txt'))
    
    def _load_annotations(self, anno_file):
        with open(anno_file, 'r') as f:
            for line in f:
                image_path, label = line.strip().split()
                self.images.append((image_path, int(label)))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path, label = self.images[index]
        label = label -1
        image = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
