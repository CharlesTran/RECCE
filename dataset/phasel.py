import torch
import numpy as np
from os.path import join
from torch.utils.data.dataset import Dataset
from PIL import Image

class Phasel(Dataset):
    
    def __init__(self, cfg, img=None, transforms=None):
        # super().__init__(cfg, seed, transforms, transform, target_transform)
        print(f"Loading data from 'Phasel' of split '{cfg['split']}'"
              f"\nPlease wait patiently...")  
        self.root = cfg['root']

        self.split = cfg['split']
        self.images = img['path']
        self.targets = img['target']
        self.num_real = self.targets.value_counts()[0]
        self.num_fake = self.targets.value_counts()[1]
        assert len(self.images) == len(self.targets), "Length of images and targets not the same!"
        print(f"Data from 'Phasel' loaded.")
        print(f"Real: {self.num_real}, Fake: {self.num_fake}.")
        print(f"Dataset contains {len(self.images)} images\n")
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = None
        self.categories = ['original', 'fake']
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, torch.from_numpy(np.array(self.targets[index]))
    
    def __len__(self):
        return len(self.images)