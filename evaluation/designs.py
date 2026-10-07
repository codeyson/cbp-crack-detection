"""
designs.py - the 3 alternative designs: ImageNet-pretrained torchvision classifiers, block crop -> grade A/B/C.
All three use this same training and prediction code, so only the architecture differs.

    D1 MobileNetV3-Large  5.5M params   speed-first (depthwise convolutions, NAS-designed)
    D2 EfficientNet-B0    5.3M params   accuracy-per-parameter (compound scaling)
    D3 ResNet18          11.7M params   classic residual baseline

    train("d1", "dataset", epochs=30, imgsz=224, out="models")   # dataset/{train,val}/{A,B,C}/*.png
    predict = load("d1", "models/d1_mobilenet_v3_large.pth")
    predict(cv2.imread("crop.png"))                               # -> "A"

Maintainability of a design = this file + its torchvision architecture file (see source_files()).
"""
import os

import cv2
import torch
import torchvision.models
from torch import nn
from torchvision import datasets, models, transforms

DESIGNS = dict(d1="mobilenet_v3_large", d2="efficientnet_b0", d3="resnet18")
SOURCE = dict(d1="mobilenetv3.py", d2="efficientnet.py", d3="resnet.py")   # in torchvision/models/
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]                   # ImageNet


def build(design, n_classes=3, pretrained=True):
    model = getattr(models, DESIGNS[design])(weights="DEFAULT" if pretrained else None)
    if hasattr(model, "fc"):   # ResNet
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    else:                      # MobileNet, EfficientNet: last layer of the classifier
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
    return model


def weights_path(design, out="models"):
    return os.path.join(out, f"{design}_{DESIGNS[design]}.pth")


def train(design, data_dir, epochs=30, imgsz=224, out="models", batch=32, lr=1e-3, seed=0, pretrained=True):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    norm = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    train_ds = datasets.ImageFolder(f"{data_dir}/train", transforms.Compose(
        [transforms.RandomResizedCrop(imgsz, scale=(0.8, 1.0)), transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip()] + norm))
    val_ds = datasets.ImageFolder(f"{data_dir}/val", transforms.Compose([transforms.Resize((imgsz, imgsz))] + norm))
    train_dl = torch.utils.data.DataLoader(train_ds, batch, shuffle=True, num_workers=2)
    val_dl = torch.utils.data.DataLoader(val_ds, batch, num_workers=2)

    model = build(design, len(train_ds.classes), pretrained).to(device)
    opt = torch.optim.Adam(model.parameters(), lr)
    loss_fn = nn.CrossEntropyLoss()
    path, best = weights_path(design, out), -1.0
    os.makedirs(out, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        for x, y in train_dl:
            opt.zero_grad()
            loss_fn(model(x.to(device)), y.to(device)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            correct = sum((model(x.to(device)).argmax(1).cpu() == y).sum().item() for x, y in val_dl)
        acc = correct / len(val_ds)
        print(f"{design} epoch {epoch + 1}/{epochs}  val_acc {acc:.3f}")
        if acc > best:   # keep the best epoch on val, never on test
            best = acc
            torch.save(dict(state_dict=model.state_dict(), classes=train_ds.classes, imgsz=imgsz), path)
    return path


def load(design, weights=None, imgsz=224):
    """weights=None: untrained model, only for timing/memory before the dataset exists."""
    classes = ["A", "B", "C"]
    model = build(design, len(classes), pretrained=False)
    if weights:
        ckpt = torch.load(weights, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        classes, imgsz = ckpt["classes"], ckpt["imgsz"]
    model.eval()
    mean, std = torch.tensor(MEAN).view(3, 1, 1), torch.tensor(STD).view(3, 1, 1)

    def predict(img_bgr):
        rgb = cv2.cvtColor(cv2.resize(img_bgr, (imgsz, imgsz)), cv2.COLOR_BGR2RGB)
        x = (torch.from_numpy(rgb).permute(2, 0, 1).float() / 255 - mean) / std
        with torch.no_grad():
            return classes[model(x[None]).argmax(1).item()]
    return predict


def source_files(design):
    """The code a future researcher maintains for this design: our code + the architecture's implementation."""
    return [os.path.abspath(__file__), os.path.join(os.path.dirname(torchvision.models.__file__), SOURCE[design])]
