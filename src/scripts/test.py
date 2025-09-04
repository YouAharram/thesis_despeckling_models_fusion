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

    criterion = nn.MSELoss()
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


def save_quality_map_comparison(preds, targets, save_dir="test_results", max_samples=10):
    os.makedirs(save_dir, exist_ok=True)

    num_samples = min(len(preds), max_samples)

    for i in range(num_samples):
        pred = preds[i].squeeze()  # rimuove dimensioni 1
        target = targets[i].squeeze()

        # Se rimane (C, H, W), prendo il primo canale
        if pred.ndim == 3:
            pred = pred[0]
        if target.ndim == 3:
            target = target[0]

        pred = pred.numpy()
        target = target.numpy()
        diff = abs(pred - target)

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        
        axs[0].imshow(target, cmap="viridis")
        axs[0].set_title("Noisy")
        axs[0].axis("off")

        axs[1].imshow(pred, cmap="viridis")
        axs[1].set_title("Prediction")
        axs[1].axis("off")

        axs[2].imshow(diff, cmap="magma")
        axs[2].set_title("Difference |pred - target|")
        axs[2].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"comparison_{i}.png"))
        plt.close(fig)

    print(f"Saved {num_samples} comparison plots in '{save_dir}'")



if __name__ == "__main__":
    CLEAN_PATH = "datasets/testing/clean/airplane/"
    NOISY_PATH = "datasets/testing/look4/noisy/airplane/"
    DENOISED_PATH = "datasets/testing/look4/SAR-CAM/denoised/airplane/"
    MODEL_PATH = "models/MSELOSS_L4_SAR-CAM.pth"

    device = "cuda" if torch.cuda.is_available else "cpu"

    test_loader = load_test_datasets(NOISY_PATH, DENOISED_PATH, CLEAN_PATH)

    preds, targets, loss = test_model(test_loader, MODEL_PATH, device) # capire se si puo fare un plot che mostri dove preds e targets differiscono

    save_quality_map_comparison(preds, targets)