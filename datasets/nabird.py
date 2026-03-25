import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class NABirdsDataset(Dataset):
    def __init__(self, file_path, split='train', transform=None):

        self.data = pd.read_csv(file_path, sep=' ', header=None, names=['id', 'label', 'image_path', 'is_train'])


        self.label_mapping = {label: idx for idx, label in enumerate(sorted(self.data['label'].unique()))}
        
      
        self.data['mapped_label'] = self.data['label'].map(self.label_mapping)
        

        if split == 'train':
            self.data = self.data[self.data['is_train'] == 1]
        elif split == 'test':
            self.data = self.data[self.data['is_train'] == 0]
        else:
            raise ValueError("split must be 'train' or 'test'")
        
        self.transform = transform
        self.base_path = '/home/kzp/lzw/DataSet/nabirds/nabirds/images/'


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data.iloc[idx]
        label = int(sample['mapped_label']) 
        image_path = os.path.join(self.base_path, sample['image_path'])
        

        image = Image.open(image_path).convert('RGB')


        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == '__main__':
   
    dataset_path = '/home/kzp/lzw/DataSet/nabirds/nabirds/combined_data.txt'
    
 
    train_dataset = NABirdsDataset(file_path=dataset_path, split='train')


    test_dataset = NABirdsDataset(file_path=dataset_path, split='test')


    print(f'Train Dataset size: {len(train_dataset)}')
    print(f'Test Dataset size: {len(test_dataset)}')


    image, label = train_dataset[0]
    print(f'Train Image shape: {image.size}, Label: {label}')
