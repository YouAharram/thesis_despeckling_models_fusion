import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from .train_test_data_dataset import train_test_dataset

# +-------------------------------------------------------------------------+
# test_data_loading is used to create the loaders for testing the network. 
# In the transforms, a clipping operation is applied to avoid outliers.
# +-------------------------------------------------------------------------+

class PercentileClipping:
    def __init__(self, q=0.99):
        self.q = q

    def __call__(self, img):
        # img è un PIL Image
        im_np = np.array(img).astype(np.float32)  # mantiene valori originali, es. 0-255

        # calcolo del 99° percentile
        p = np.percentile(im_np, self.q * 100)

        # normalizzazione e clipping
        im_np = im_np / p
        im_np[im_np > 1] = 1.0

        # aggiungo la dimensione canale [1,H,W] per PyTorch
        im_np = im_np[None, :, :]

        # converti in tensore torch
        return torch.from_numpy(im_np)



def load_test_datasets(noisy_dir, denoised_dir, clean_dir, batch_size=8):
    full_dataset = train_test_dataset(
        noisy_dir = noisy_dir,
        denoised_dir = denoised_dir,
        clean_dir = clean_dir,
        transform = transforms.Compose([
            PercentileClipping(0.99)
        ])
    )

    test_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return test_loader