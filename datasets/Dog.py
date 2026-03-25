import os
from torch.utils.data import Dataset
from PIL import Image

class DogDataset(Dataset):
    def __init__(self, file_path, transform=None):
        """

        """
        self.data = []
        self.labels = []
        self.transform = transform
        self.base_dir = '/home/kzp/lzw/DataSet/Standford_dog/images/Images'
  
        with open(file_path, 'r') as f:
            for line in f:
              
                line = line.strip()
                if line:  
                    
                    path, label = eval(line)  #
                    self.data.append(path[0])  
                    self.labels.append(int(label))  

    def __len__(self):
        
        return len(self.data)

    def __getitem__(self, idx):
        """

        """
        img_path = self.data[idx]
        image_full_path = os.path.join(self.base_dir, img_path)
        label = self.labels[idx] - 1
        

        img = Image.open(image_full_path).convert('RGB')  
        

        if self.transform:
            img = self.transform(img)
        
        return img, label

