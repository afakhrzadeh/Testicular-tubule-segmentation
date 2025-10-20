import os
import logging

import timm
import torch
import torch.nn as nn
from torchvision import transforms
from typing import Optional, Tuple, Type

from .models.resnet50_trunc import resnet50_trunc_imagenet
from .common import LayerNorm2d
def get_norm_constants(which_img_norm: str = 'imagenet'):
    constants_zoo = {
        'imagenet': {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)},
        'ctranspath': {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)},
        'openai_clip':{'mean': (0.48145466, 0.4578275, 0.40821073), 'std': (0.26862954, 0.26130258, 0.27577711)},
        'uniform': {'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)}
    }
    constants = constants_zoo[which_img_norm]
    return constants.get('mean'), constants.get('std')


def get_eval_transforms(
        which_img_norm: str = 'imagenet', 
        img_resize: int = 224, 
        center_crop: bool = False
):
    r"""
    Gets the image transformation for normalizing images before feature extraction.

    Args:
        - which_img_norm (str): transformation type

    Return:
        - eval_transform (torchvision.Transform): PyTorch transformation function for images.
    """
    
    eval_transform = []

    if img_resize > 0:
        eval_transform.append(transforms.Resize(img_resize))

        if center_crop:
            eval_transform.append(transforms.CenterCrop(img_resize))

    mean, std = get_norm_constants(which_img_norm)

    eval_transform.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    eval_transform = transforms.Compose(eval_transform)
    return eval_transform


class uni_image_encoder(nn.Module):
    def __init__(
        self,
        enc_name='vit_large_patch16_224.dinov2.uni_mass100k', 
        which_img_norm='imagenet', 
        img_resize=224, 
        embed_dim=768,
        out_chans=256,
        device=None,
        ) -> None:
        
        """      
            Get image encoder with pretrained weights and the their normalization.

            Args:
                - img_size (int): Input image size.
                - enc_name (str): Name of the encoder (finds folder that is named <enc_name>, which the model checkpoint assumed to be in this folder)
                - checkpoint (str): Name of the checkpoint file (including extension)
                - assets_dir (str): Path to where checkpoints are saved.

            Return:
                - model (torch.nn): PyTorch model used as image encoder.
                - eval_transforms (torchvision.transforms): PyTorch transformation function for images.

        """
        super().__init__()
        self.img_size = img_resize
        self.patch_size = 16
        self.in_chans = 3
        self.embed_dim = embed_dim
        self.out_chans = out_chans


        enc_name_presets = {
            'resnet50_trunc': ('resnet50.supervised.trunc_in1k_transfer', None, 'imagenet'),
            'uni': ('vit_large_patch16_224.dinov2.uni_mass100k', 'imagenet'),
        }
        
        if enc_name in enc_name_presets.keys():
            enc_name, which_img_norm = enc_name_presets[enc_name]
        
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        ### ResNet50 Truncated Encoder, Dim=1024, Pretrained on ImageNet
        if enc_name == 'resnet50trunc.supervised.in1k_transfer':
            self.model = resnet50_trunc_imagenet()
            assert which_img_norm == 'imagenet'

        ### UNI
        elif enc_name == 'vit_large_patch16_224.dinov2.uni_mass100k':
            uni_kwargs = {
                'model_name': 'vit_large_patch16_224',
                'img_size': 224, 
                # 'patch_size': self.patch_size, 
                'init_values': 1e-5, 
                'num_classes': 0, 
                'dynamic_img_size': True
            }
            self.model = timm.create_model(**uni_kwargs)
            
            ''' select all the forward features from UNI image encoder
                because UNI just output the head for classification we add this line
            '''
            self.model.forward_features = lambda res: self.model.patch_embed(res)
            
            self.neck = nn.Sequential(
                        nn.Conv2d(
                            self.embed_dim,
                            self.out_chans,
                            kernel_size=1,
                            bias=False,
                        ),
                        LayerNorm2d(self.out_chans),
                        nn.Conv2d(
                            self.out_chans,
                            self.out_chans,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        LayerNorm2d(self.out_chans),
                    )

        else:
            return None, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x should be (batch_size, 3, 224, 224)
        with torch.no_grad():
            features = self.model.forward_features(x)
            features  = self.neck(features.permute(0, 3, 1, 2))
        return features
    
