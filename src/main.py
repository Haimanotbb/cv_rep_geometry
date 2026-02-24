import torch

from src.data.cifar10 import get_cifar10_loaders
from src.models.load import load_models

@torch.no_grad()
def smoke_test_forward(models, batch, device):
    x, y = batch
    x = x.to(device)

    shapes = {}
    for name, model in models.items():
        out = model(x)
        shapes[name] = tuple(out.shape)
    return shapes

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_root = "./data"
    batch_size = 64
    num_workers = 2
    image_size = 224

    train_loader, test_loader = get_cifar10_loaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
    )

    models = load_models(device)
    print("Loaded models:", list(models.keys()))

    batch = next(iter(test_loader))
    shapes = smoke_test_forward(models, batch, device)

    print("Output shapes from one forward pass:")
    for name, shape in shapes.items():
        print(f"  {name}: {shape}")

if __name__ == "__main__":
    main()
