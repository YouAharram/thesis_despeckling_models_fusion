import torch
import os
from src.load_dataset.models_img_denoised_loading import models_img_loading


def image_overlap(test_loader1, test_loader2, test_loader3):
    imgs  = []

    for inputs1,inputs2,inputs3 in zip(test_loader1, test_loader2, test_loader3):
        imgs.append(torch.cat([inputs1[1].unsqueeze(0), inputs2[1].unsqueeze(0), inputs3[1].unsqueeze(0)], dim=0)) # [3,h,w]

    return imgs


if __name__ == "__main__":
    NOISY_PATH1 = "datasets/testing/look1/noisyTest/airplane/"
    DENOISED_PATH1 = "datasets/testing/look1/SAR-CAM-IMG/airplane/"

    NOISY_PATH2 = "datasets/testing/look1/noisyTest/airplane/"
    DENOISED_PATH2 = "datasets/testing/look1/FANS-IMG/airplane/"

    NOISY_PATH3 = "datasets/testing/look1/noisyTest/airplane/"
    DENOISED_PATH3 = "datasets/testing/look1/SARBM3D-IMG/airplane/"

    device = "cuda" if torch.cuda.is_available else "cpu"

    test_loader1 = models_img_loading(NOISY_PATH1, DENOISED_PATH1)
    test_loader2 = models_img_loading(NOISY_PATH2, DENOISED_PATH2)
    test_loader3 = models_img_loading(NOISY_PATH3, DENOISED_PATH3)

    imgs_denoised = image_overlap(test_loader1,test_loader2, test_loader3)
    
    