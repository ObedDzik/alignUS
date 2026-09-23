"""Miscellaneous UNet builders"""

from medAI.modeling.registry import register_model


@register_model
def unet_small(): 
    from monai.networks.nets import UNet

    net = UNet(
        spatial_dims=2, in_channels=3, out_channels=1, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2,
    )

    return net


@register_model
def unet_medium(): 
    from monai.networks.nets import UNet

    # medium UNet, 5 downsampling layers
    net = UNet(
        spatial_dims=2, in_channels=3, out_channels=1, channels=(16, 32, 64, 128, 256, 512), strides=(2, 2, 2, 2, 2), num_res_units=2,
    )

    return net
