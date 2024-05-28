import numpy as np

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel

def _addx(np_img, value=-1):
    new = np_img[:,:value,:]
    added = np.append(np_img, new, axis=1)
    return added

def _addy(np_img, value=-1):
    new = np_img[:value,:,:]
    added = np.append(np_img, new, axis=0)
    return added


def padding(np_image):
    if np_image.shape[1] < np_image.shape[0]:
        while np_image.shape[1] < np_image.shape[0]:
            value = np_image.shape[0] - np_image.shape[1]
            if value < np_image.shape[1]:
                np_image = _addx(np_image, value)
            else:
                np_image = _addx(np_image)

    elif np_image.shape[0] < np_image.shape[1]: 
        while np_image.shape[0] < np_image.shape[1]:
            value = np_image.shape[1] - np_image.shape[0]
            if value < np_image.shape[0]:
                np_image = _addy(np_image, value)
            else:
                np_image = _addy(np_image)

    return np_image


def load_models(pretrained_model_name):
    
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name, subfolder='tokenizer')
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name, subfolder='scheduler')
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name, subfolder='text_encoder')
    vae = AutoencoderKL.from_pretrained(pretrained_model_name, subfolder='vae')
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name, subfolder='unet')

    vae.requires_grad_(False)
    unet.requires_grad_(False)

    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    return tokenizer, noise_scheduler, text_encoder, vae, unet
