import torch
from torch.utils.data import random_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from .dataset import QualityMapDataset

def load_datasets(noisy_dir, denoised_dir, clean_dir, train_percentage=0.8, batch_size=8, seed=42):
    full_dataset = QualityMapDataset(
        noisy_dir = noisy_dir,
        denoised_dir = denoised_dir,
        clean_dir = clean_dir,
        transform = transforms.Compose([
            transforms.ToTensor(), #vedere se conviene fare una traformazione più utile in futuro      
            ])
    )

    # Split train/val
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