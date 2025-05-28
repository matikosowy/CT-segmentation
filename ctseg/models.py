"""
Model Module (model.py)
-------------------
This module defines the neural network architectures for CT segmentation.
"""

import torch
from monai.networks.nets import UNet, SegResNet


def create_unet_model(
    out_channels,
    dims,
    in_channels=1,
    channels=(64, 128, 256, 512),
    strides=(2, 2, 2),
    num_res_units=2,
    norm="batch",
    dropout=0.2,
    device="cuda",
):
    """
    Create a 2D UNet model for slice-by-slice segmentation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels (classes).
        dims (int): Spatial dimensions (2 for 2D, 3 for 3D).
        device (str or torch.device): Device to place the model on.
        channels (tuple): Number of channels at each level.
        strides (tuple): Strides for downsampling.
        num_res_units (int): Number of residual units in each block.
        norm (str): Normalization type ('batch', 'instance').
        dropout (float): Dropout probability.
        device (str or torch.device): Device to place the model on.

    Returns:
        nn.Module: UNet 2D model.
    """
    print("=" * 50)
    print(f"Creating {dims}D UNet model...")

    model = UNet(
        spatial_dims=dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        norm=norm,
        dropout=dropout,
    ).to(device)

    # Init weights - He
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d)):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)

    print(f"Done! Number of parameters: {sum(p.numel() for p in model.parameters())}")

    return model


def create_segresnet_model(
    out_channels,
    dims,
    in_channels=1,
    device="cuda",
    init_filters=16,
    use_conv_final=True,
    blocks_down=(1, 1, 1, 2),
    blocks_up=(1, 1, 1),
    dropout_prob=0.2,
):
    """
    Create an optimized SegResNet model to outperform UNet.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels (classes).
        dims (int): Spatial dimensions (2 for 2D, 3 for 3D).
        device (str or torch.device): Device to place the model on.
        init_filters (int): Number of initial filters.
        use_conv_final (bool): Whether to use final convolution layer.
        blocks_down (tuple): Number of blocks in downsampling path.
        blocks_up (tuple): Number of blocks in upsampling path.
        dropout_prob (float): Dropout probability.

    Returns:
        nn.Module: SegResNet model.
    """
    print("=" * 50)
    print(f"Creating {dims}D SegResNet model...")

    model = SegResNet(
        spatial_dims=dims,
        in_channels=in_channels,
        out_channels=out_channels,
        init_filters=init_filters,
        blocks_down=blocks_down,
        blocks_up=blocks_up,
        dropout_prob=dropout_prob,
        use_conv_final=use_conv_final,
    ).to(device)

    # Init weights - Kaiming He
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d)):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)

    print(f"Done! Number of parameters: {sum(p.numel() for p in model.parameters())}")

    return model
