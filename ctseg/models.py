"""
Model Module (model.py)
-------------------
This module defines the neural network architectures for CT segmentation.
"""

import torch
from monai.networks.nets import UNet, SegResNet

def create_unet_model(in_channels=1, out_channels=3, device="cuda"):
    """
    Create a standard 3D UNet model.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels (classes).
        device (str or torch.device): Device to place the model on.
        
    Returns:
        nn.Module: UNet model.
    """
    model = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=3,
        norm='INSTANCE',
        dropout=0.2
    ).to(device)
    
    return model

def create_segresnet_model(in_channels=1, out_channels=3, device="cuda"):
    """
    Create a SegResNet model, which often has better performance than UNet.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels (classes).
        device (str or torch.device): Device to place the model on.
        
    Returns:
        nn.Module: SegResNet model.
    """
    model = SegResNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        init_filters=32,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.2
    ).to(device)
    
    return model
