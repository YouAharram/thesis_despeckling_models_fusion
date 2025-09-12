import torch 
from .unet import UNet
from src.load_dataset.models_img_denoised_loading import models_img_loading 
import os
import imageio.v2 as imageio
import numpy as np

# +-------------------------------------------------------------------+
# fusion_with_weight_average is used to take the denoised images from
# each model, generate the quality map for each image using the
# corresponding models, and then perform the fusion. Finally, it saves
#  everything in the test_results folder.
# +-------------------------------------------------------------------+

def fusion_of_models(models, test_loader1, test_loader2, test_loader3, device):
    imgs_final_despeckled = []

    with torch.no_grad():
        for inputs1, inputs2, inputs3 in zip(test_loader1, test_loader2, test_loader3):
            # [2, 256, 256]--> [1, 2, 256, 256] for models
            inputs1 = inputs1.unsqueeze(0).to(device)  
            inputs2 = inputs2.unsqueeze(0).to(device)
            inputs3 = inputs3.unsqueeze(0).to(device)

            # q_map_out? size --> [1, 1, 256, 256]
            q_map_out1 = models[0](inputs1) 
            q_map_out2 = models[1](inputs2)
            q_map_out3 = models[2](inputs3)

            q_maps = torch.cat([q_map_out1, q_map_out2, q_map_out3], dim=1)
            q_maps_softmax = torch.softmax(q_maps, dim=1)

            w1, w2, w3 = q_maps_softmax[:, 0:1, :, :], q_maps_softmax[:, 1:2, :, :], q_maps_softmax[:, 2:3, :, :]

            # [1, 1, 256, 256] only denoised images
            input1 = inputs1[ : , 1:2 , :, :] 
            input2 = inputs2[ : , 1:2 , :, :]
            input3 = inputs3[ : , 1:2 , :, :]            

            img = (input1 * w1 + input2 * w2 + input3 * w3)/(w1 + w2 + w3)
            imgs_final_despeckled.append(img)        

    return imgs_final_despeckled

def save_image(img, save_path):
    img = img.detach().cpu().numpy()

    # Normalizza tra 0-255 (uint8)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.imwrite(save_path, img)  # salva come 256x256, 1 canale
    print(f"Immagine salvata in {save_path}")


if __name__ == "__main__":
    NOISY_PATH1 = "datasets/testing/look1/noisyTest/golfcourse/"
    DENOISED_PATH1 = "datasets/testing/look1/SAR-CAM-IMG/golfcourse/"

    NOISY_PATH2 = "datasets/testing/look1/noisyTest/golfcourse/"
    DENOISED_PATH2 = "datasets/testing/look1/FANS-IMG/golfcourse/"

    NOISY_PATH3 = "datasets/testing/look1/noisyTest/golfcourse/"
    DENOISED_PATH3 = "datasets/testing/look1/SARBM3D-IMG/golfcourse/"
    
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

    imgs = fusion_of_models(models, test_loader1, test_loader2, test_loader3, device)

    i  = 0
    for img in imgs:
        img  = img.squeeze()
        save_image(img, f"test_results/golfcourse{i:02}.tif")
        i+=1