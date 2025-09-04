import torch 
from .unet import UNet
from src.load_dataset.models_img_denoised_loading import models_img_loading 
import os


def fusion_of_models(models, test_loader1, test_loader2, test_loader3, device):
    imgs_final_despeckled = []

    with torch.no_grad():
        for inputs1, inputs2, inputs3 in zip(test_loader1, test_loader2, test_loader3):
            inputs1 = inputs1.unsqueeze(0).to(device)  # da ragionare
            inputs2 = inputs2.unsqueeze(0).to(device)
            inputs3 = inputs3.unsqueeze(0).to(device)
            
            q_map_out1 = models[0](inputs1)
            q_map_out2 = models[1](inputs2)
            q_map_out3 = models[2](inputs3)

            img = (inputs1 * q_map_out1 + inputs2 * q_map_out2 + inputs3 * q_map_out3)/(q_map_out1 + q_map_out2 + q_map_out3)
            imgs_final_despeckled.append(img)        

    return imgs_final_despeckled



import matplotlib.pyplot as plt
import os

def save_image(img, save_path, cmap="gray"):
    img = img.squeeze()   # rimuove dimensioni inutili

    # Se è (C,H,W), prendo il primo canale
    if img.ndim == 3:
        img = img[0]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.imshow(img.numpy(), cmap=cmap)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Immagine salvata in {save_path}")







if __name__ == "__main__":
    NOISY_PATH1 = "datasets/testing/look1/noisyTest/airplane/"
    DENOISED_PATH1 = "datasets/testing/look1/SAR-CAM-IMG/airplane/"

    NOISY_PATH2 = "datasets/testing/look1/noisyTest/airplane/"
    DENOISED_PATH2 = "datasets/testing/look1/FANS-IMG/airplane/"

    NOISY_PATH3 = "datasets/testing/look1/noisyTest/airplane/"
    DENOISED_PATH3 = "datasets/testing/look1/SARBM3D-IMG/airplane/"
    
    MODEL_PATH1 = "models/MSELOSS_L1_SAR-CAM-T.pth"
    MODEL_PATH2 = "models/MSELOSS_L1_FANST.pth"
    MODEL_PATH3 = "models/MSELOSS_L1_BM3DT.pth"

    device = "cuda" if torch.cuda.is_available else "cpu"

    test_loader1 = models_img_loading(NOISY_PATH1, DENOISED_PATH1)
    test_loader2 = models_img_loading(NOISY_PATH2, DENOISED_PATH2)
    test_loader3 = models_img_loading(NOISY_PATH3, DENOISED_PATH3)

    model1 = UNet(in_channels=2, n_classes=1)
    model2 = UNet(in_channels=2, n_classes=1)
    model3 = UNet(in_channels=2, n_classes=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model1.load_state_dict(torch.load(MODEL_PATH1, map_location=device))
    model2.load_state_dict(torch.load(MODEL_PATH2, map_location=device))
    model3.load_state_dict(torch.load(MODEL_PATH3, map_location=device))
    
    model1.to(device)
    model2.to(device)
    model3.to(device)

    models = [model1, model2, model3] 

    img = fusion_of_models(models, test_loader1, test_loader2, test_loader3, device)

    save_image(img[0], "test_results/img[0].png")


