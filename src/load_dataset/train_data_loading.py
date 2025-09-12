import torch
from torch.utils.data import random_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from .train_test_data_dataset import train_test_dataset
from PIL import Image
import numpy as np

# +--------------------------------------------------------------------------------------+
# train_data_loading is used to create the loaders for network training and validation.
# In the transforms, a clipping operation is applied to avoid outliers.
# +--------------------------------------------------------------------------------------+

class PercentileClipping:
    def __init__(self, q=0.99):
        self.q = q

    def __call__(self, img):
        im_np = np.array(img, dtype=np.float32)
        p = np.percentile(im_np, self.q)
        p = max(p, 1e-6)
        im_np = im_np / p
        im_np = np.nan_to_num(im_np, nan=0.0, posinf=1.0, neginf=0.0)
        im_np = np.clip(im_np, 0, 1)

        return torch.from_numpy(im_np[None, :, :])
    

def load_train_datasets(noisy_dir, denoised_dir, clean_dir, train_percentage=0.8, batch_size=8, seed=42):
    full_dataset = train_test_dataset(
        noisy_dir = noisy_dir,
        denoised_dir = denoised_dir,
        clean_dir = clean_dir,
        transform = transforms.Compose([            
            PercentileClipping(0.99)
            ])
    )

    train_size = int(train_percentage * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator= torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    
    return train_loader, val_loader