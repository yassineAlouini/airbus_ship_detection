from segmentation_models import Unet


def build_pretrained_unet_model():
    """Build a pretrained Unet model. """
    return Unet(backbone_name='resnet34', encoder_weights='imagenet')
