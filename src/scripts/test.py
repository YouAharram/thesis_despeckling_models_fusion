import torch
from torch import optim, nn
from tqdm import tqdm
from .unet import UNet 
from src.load_dataset.test_data_loading import load_test_datasets
import os
import matplotlib.pyplot as plt

def test_model(test_loader, model_path, device):
    model = UNet(in_channels=2, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    criterion = nn.L1Loss()
    test_loss = 0.0
    
    preds, targets = [], []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for inputs, q_map_in in pbar:
            inputs = inputs.to(device)
            q_map_in = q_map_in.to(device)

            q_map_out = model(inputs)

            loss = criterion(q_map_out, q_map_in)
            test_loss += loss.item()

            preds.append(q_map_out.cpu())
            targets.append(q_map_in.cpu())

            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        test_loss /= len(test_loader)
        print(f"\nFinal test loss: {test_loss:.4f}")

        return preds, targets, test_loss

if __name__ == "__main__":
    CLEAN_PATH = "datasets/testing/clean/airplane/"
    NOISY_PATH = "datasets/testing/look4/noisy/airplane/"
    DENOISED_PATH = "datasets/testing/look4/SAR-CAM/denoised/airplane/"
    MODEL_PATH = "models/best_unet_quality_map.pth"

    device = "cuda" if torch.cuda.is_available else "cpu"

    test_loader = load_test_datasets(NOISY_PATH, DENOISED_PATH, CLEAN_PATH)

    preds, targets, loss = test_model(test_loader, MODEL_PATH, device) # capire se si puo fare un plot che mostri dove preds e targets differiscono
