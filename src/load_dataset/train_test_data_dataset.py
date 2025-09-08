import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch
import numpy as np

# +-----------------------------------------------------------------------------------------+
# A train_test_data_dataset class is used to load the noisy, denoised, and clean images. 
# These images are transformed into tensors through a transform that is passed to the
# constructor, and it returns a list of tensors [2,h,w] along with the quality map.
# +-----------------------------------------------------------------------------------------+

class train_test_dataset(Dataset):
    def __init__(self, noisy_dir, denoised_dir, clean_dir, transform=None):
        self.clean_files = sorted([f for f in os.listdir(clean_dir) 
                            if f.lower().endswith(('.png', '.tif'))])
        self.noisy_files = sorted([f for f in os.listdir(noisy_dir) 
                            if f.lower().endswith(('.png', '.tif'))])
        self.denoised_files = sorted([f for f in os.listdir(denoised_dir) 
                            if f.lower().endswith(('.png', '.tif'))])
        
         # Verifica che i file siano allineati
        assert self.noisy_files == self.denoised_files == self.clean_files, \
            "Le cartelle devono contenere gli stessi file con lo stesso ordine!"
        
        self.noisy_dir = noisy_dir
        self.denoised_dir = denoised_dir
        self.clean_dir = clean_dir
        self.transform = transform or transforms.ToTensor()
    
    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        # Carica tutte e tre le immagini
        noisy = Image.open(os.path.join(self.noisy_dir, self.noisy_files[idx])).convert('L')
        denoised = Image.open(os.path.join(self.denoised_dir, self.denoised_files[idx])).convert('L')
        clean = Image.open(os.path.join(self.clean_dir, self.clean_files[idx])).convert('L')

        if self.transform:
            noisy = self.transform(noisy)
            denoised = self.transform(denoised)
            clean = self.transform(clean)

        quality_map = torch.abs(denoised - clean) 
        input_img = torch.cat([noisy,denoised],dim = 0) # [2,h,w]

        return input_img,quality_map