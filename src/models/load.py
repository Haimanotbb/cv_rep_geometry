from typing import Dict

import torch
import torchvision.models as tvm
import timm

def load_models(device: torch.device) -> Dict[str, torch.nn.Module]:
    """
    Loads pretrained models (eval mode) and moves them to device. Returns a dictionary: name -> model
    """
    models: Dict[str, torch.nn.Module] = {}
    resnet = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
    resnet.eval()
    resnet.to(device)
    models["resnet50"] = resnet

    convnext = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.DEFAULT)
    convnext.eval()
    convnext.to(device)
    models["convnext_tiny"] = convnext

    vit = timm.create_model("vit_base_patch16_224", pretrained=True)
    vit.eval()
    vit.to(device)
    models["vit_base_patch16_224"] = vit

    return models
