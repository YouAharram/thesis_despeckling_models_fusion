import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from .dataset import QualityMapDataset

def load_test_datasets(noisy_dir, denoised_dir, clean_dir, batch_size=8):
    full_dataset = QualityMapDataset(
        noisy_dir = noisy_dir,
        denoised_dir = denoised_dir,
        clean_dir = clean_dir,
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    )

    test_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return test_loader