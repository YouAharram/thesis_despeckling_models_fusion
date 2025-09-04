import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from .train_test_dataset import train_test_dataset

def load_test_datasets(noisy_dir, denoised_dir, clean_dir, batch_size=8):
    full_dataset = train_test_dataset(
        noisy_dir = noisy_dir,
        denoised_dir = denoised_dir,
        clean_dir = clean_dir,
        transform = transforms.Compose([
            transforms.ToTensor()  #da chidedere a Chiara per il clip
        ])
    )

    test_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return test_loader