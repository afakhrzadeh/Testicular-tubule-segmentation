# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial
import logging
from segment_anything.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
from segment_anything.modeling.UNI_image_encoder import uni_image_encoder
import os

def build_sam_vit_h(checkpoint=None,
                    uni_checkpoint=None,
                    neck_checkpoint=None,
                    mode=None,
                    model_encoder="SAM"):
    print("==> model type: vit_h")
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        uni_checkpoint=uni_checkpoint,
        neck_checkpoint=neck_checkpoint,
        mode=mode,
        model_encoder=model_encoder,
    )


def build_sam_vit_l(checkpoint=None,
                    uni_checkpoint=None,
                    neck_checkpoint=None,
                    mode=None,
                    model_encoder="SAM"):
    print("==> model type: vit_l")
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        uni_checkpoint=uni_checkpoint,
        neck_checkpoint=neck_checkpoint,
        mode=mode,
        model_encoder=model_encoder,
    )


def build_sam_vit_b(checkpoint=None,
                    uni_checkpoint=None,
                    neck_checkpoint=None,
                    mode=None,
                    model_encoder="SAM"):
    print("==> model type: vit_b")
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        uni_checkpoint=uni_checkpoint,
        neck_checkpoint=neck_checkpoint,
        mode=mode,
        model_encoder=model_encoder,
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    uni_checkpoint=None,
    neck_checkpoint=None,
    mode=None,
    model_encoder="SAM"
):
    print(f"====> model encoder: {model_encoder}")
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    if model_encoder == "SAM":
        encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        )
    elif model_encoder == "UNI":
        encoder=uni_image_encoder(
            enc_name='vit_large_patch16_224.dinov2.uni_mass100k', 
            which_img_norm='imagenet', 
            img_resize=1024,
            embed_dim=encoder_embed_dim,
            out_chans=prompt_embed_dim,
            device=None,
                )
        
    sam = Sam(
        image_encoder=encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    if model_encoder == "SAM" or mode == "Inference" or mode == "reTrain":
        if checkpoint is not None:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f)
            missing_keys, unexpected_keys = sam.load_state_dict(state_dict, strict=True)

            logging.warning(f'Model encoder name {model_encoder} ----> \n ===>\n Missing Keys in model: {missing_keys} ===>\n Unexpected Keys in model: {unexpected_keys}')
            
    elif mode == "train" and model_encoder == "UNI":
        # Load encoder checkpoint
        if uni_checkpoint is not None or neck_checkpoint is not None:
            new_state_dict = {}

            # Load and process uni_checkpoint
            if uni_checkpoint is not None:
                with open(uni_checkpoint, "rb") as f:
                    encoder_state_dict = torch.load(f)
                    # Add "model." prefix to all keys
                    new_state_dict.update({f"model.{k}": v for k, v in encoder_state_dict.items()})

            # Load and process neck_checkpoint
            if neck_checkpoint is not None:
                with open(neck_checkpoint, "rb") as f:
                    neck_state_dict = torch.load(f)
                    # Add "model." prefix to all keys
                    new_state_dict.update({f"neck.{k}": v for k, v in neck_state_dict.items()})

            # Load the merged state dictionary
            encoder_missing_keys, encoder_unexpected_keys = sam.image_encoder.load_state_dict(
                new_state_dict,
                strict=True
            )

        if checkpoint is not None:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f)
            missing_keys, unexpected_keys = sam.load_state_dict(state_dict, strict=False)

        # logging.warning(f'Model encoder name {model_encoder} ----> \n ===>\n Missing Keys in model: {encoder_missing_keys} ===>\n Unexpected Keys in model: {encoder_unexpected_keys} \n Model decoder and prompt ----> \n  ===>\n Missing Keys in model: {missing_keys} ===>\n Unexpected Keys in model: {unexpected_keys}')

    # logging.info(str(sam))
    return sam
