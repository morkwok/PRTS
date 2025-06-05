import numpy as np
from PIL import Image

import torch
import torch.multiprocessing
from torchvision import transforms
from einops import rearrange

import vision_transformer as vits

torch.multiprocessing.set_sharing_strategy("file_system")


def get_vit256(pretrained_weights=None, arch="vit_small", device=torch.device("cuda:0")):

    checkpoint_key = "teacher"
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    model256 = vits.__dict__[arch](patch_size=16, num_classes=0)
    
    for p in model256.parameters():
        p.requires_grad = False
    model256.eval() 
    model256.to(device)

    if pretrained_weights is not None:
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        
        if checkpoint_key is not None and checkpoint_key in state_dict:
            state_dict = state_dict[checkpoint_key]
            
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

        model256.load_state_dict(state_dict, strict=False)
        
    return model256



def eval_transforms():

    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    eval_t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return eval_t


def roll_batch2img(batch: torch.Tensor, w: int, h: int, patch_size=256):

    batch = batch.reshape(w, h, 3, patch_size, patch_size)
    img = rearrange(batch, "p1 p2 c w h-> c (p1 w) (p2 h)").unsqueeze(dim=0)
    return Image.fromarray(tensorbatch2im(img)[0])


def tensorbatch2im(input_image, imtype=np.uint8):
   
    if not isinstance(input_image, np.ndarray):
        image_numpy = input_image.cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)