import os

from pathlib import Path
import numpy as np
import torch
from src.data.cifar10 import get_cifar10_loaders
from src.models.load import load_models


def _vectorize_activations(act: torch.Tensor) -> torch.Tensor:
    """
        Convert each layer's activation tensor vector to be B X D dimensions where B is
        the number of images (batch size) and D is the number of features
    """

    if act.dim() == 4:
        # CNN: (B, C, H, W) -> average across H and W -> (B, C)
        return act.mean(dim=(2,3))
    if act.dim() == 3:
        #ViT tokens: (B, T, C) -> take first token -> (B, C)
        return act[:,0,:]
    
    if act.dim() == 2:
        #already in the appropriate shape
        return act
    
    raise ValueError(f"Unexpected activation shape: {tuple(act.shape)}")


    
def _register_hooks(model, layer_map):
    """
        layer_map: layer_name => module
        fundtion to create a map of layer naes to latest activation batch 
    """
    stash = {} #layer_name -> latest activation tensor 

    hooks = [] # a list of hook objects

    def make_hook(name):
        #create a hook function for a specific layer name

        def hook(_module, _inp, out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            
            stash[name] = out.detach() #save the output to the corresponding layer 
        
        return hook 
    
    for name, module in layer_map.items():
        hooks.append(module.register_forward_hook(make_hook(name)))
    
    return hooks, stash


def _get_layer_map(model_name: str, model: torch.nn.Module):
    """
    Pick set of layers per architecture.
    """
    if model_name == "resnet50":
        return {
            "layer1": model.layer1,  
            "layer2": model.layer2,  
            "layer3": model.layer3,
            "layer4": model.layer4,  
        }

    if model_name == "convnext_tiny":
        return {
            "features_1": model.features[1],
            "features_3": model.features[3],
            "features_5": model.features[5],
            "features_7": model.features[7],
        }

    if model_name == "vit_base_patch16_224":
        idxs = [0, 2, 5, 8, 11]
        return {f"block{idx}": model.blocks[idx] for idx in idxs}

    raise ValueError(f"Unknown model name for layer map: {model_name}")


@torch.no_grad()
def extract_and_save_for_model(
    model_name: str,
    model: torch.nn.Module, 
    test_loader,
    device:torch.device,
    out_root: str,
):
    out_dir = Path(out_root) / "activations" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_map = _get_layer_map(model_name, model)
    hooks, stash = _register_hooks(model, layer_map)

    acc = {lname:[] for lname in layer_map.keys()} #layer_name -> list of batches where each batch is (B,D)

    labels_acc = [] # store the labels from each batch 

    #for x, y in test_loader:
    for i, (x, y) in enumerate(test_loader):
        x = x.to(device)

        _ = model(x) #run the model foward once on this batch.
        #stash[layer_name] gets updated 

        labels_acc.append(y.clone())

        for lname in layer_map.keys():
            act = stash[lname]

            vec = _vectorize_activations(act).cpu()

            acc[lname].append(vec)
        
        if i == 2:
            break
    
    #remove hooks
    for h in hooks:
        h.remove()
    
    labels = torch.cat(labels_acc, dim=0).numpy() #combine all label batches into one big vector (size N) then convert to NumPy

    np.save(out_dir / "labels.npy", labels)

    for lname, chunks in acc.items():
        mat = torch.cat(chunks, dim=0).numpy() #stack all batches into one big matrix of shape (N,D)

        np.save(out_dir/ f"test_{lname}.npy", mat)
    print(f"[DONE] Saved activations for {model_name} to {out_dir}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    data_root = "./data"

    batch_size = 64

    num_workers = 0 

    image_size = 224

    _, test_loader = get_cifar10_loaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
    )

    models = load_models(device)

    out_root = "./outputs"

    for name, model in models.items():

        extract_and_save_for_model(name, model, test_loader, device, out_root)

if __name__ == "__main__":
    main()










