"""
designs.py - the 3 alternative designs: ImageNet-pretrained torchvision classifiers, block crop -> grade A/B/C.
All three use this same training and prediction code, so only the architecture differs.

    D1 MobileNetV3-Large        4.37M params  224 px  speed-first (depthwise convolutions, NAS-designed)
    D2 EfficientNet-B0, slim    2.90M params  160 px  stage 7 + 1280-ch head removed, smaller input
    D3 ResNet18, slim           2.82M params  224 px  last residual stage (layer4) removed
    (params with the classifier head: the last Linear of each design is replaced by FC 128 + ReLU -> Dropout 0.2 ->
    Linear 3, see head(); before the head 4.21M / 2.88M / 2.78M)

D2 and D3 were slimmed to compete with D1 on inference time and memory: memory differences between the designs are
mostly model size, inference time is depth x input size. Input sizes were picked on CPU (untrained, real crops) as
the largest that keeps the design about as fast as D1: D1 224 px 23.2 ms, D2 160 px 24.5 ms, D3 224 px 22.8 ms.
The full D2/D3 at 192 px were also tried: slower (29.5 / 25.0 ms) and not consistently more accurate.

    train("d1", "dataset", epochs=30, out="models")   # dataset/{train,val}/{A,B,C}/*.png, input size IMGSZ["d1"]
    predict = load("d1", "models/d1_mobilenet_v3_large.pth")
    predict(cv2.imread("crop.png"))                               # -> "A"

Maintainability of a design = this file + the part of its torchvision architecture file it uses (see source_code()).
"""
import ast
import os
import sys

import cv2
import torch
import torchvision.models
from torch import nn
from torchvision import datasets, models, transforms

DESIGNS = dict(d1="mobilenet_v3_large", d2="efficientnet_b0", d3="resnet18")
SOURCE = dict(d1="mobilenetv3.py", d2="efficientnet.py", d3="resnet.py")   # in torchvision/models/
IMGSZ = dict(d1=224, d2=160, d3=224)                                         # input px per design (see top)
HIDDEN, DROPOUT = 128, 0.2                                                   # classifier head (see head())
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]                   # ImageNet


def head(in_features, n_classes):
    """Classifier head shared by all designs, replaces the last Linear: FC 128 + ReLU -> Dropout 0.2 -> Linear."""
    return nn.Sequential(nn.Linear(in_features, HIDDEN), nn.ReLU(inplace=True), nn.Dropout(DROPOUT),
                         nn.Linear(HIDDEN, n_classes))


def build(design, n_classes=3, pretrained=True):
    model = getattr(models, DESIGNS[design])(weights="DEFAULT" if pretrained else None)
    if design == "d2":         # slim: drop stage 7 and the 1280-channel head conv, stage 6 gives 192 channels
        model.features = model.features[:7]
        model.classifier[-1] = head(192, n_classes)
    elif design == "d3":       # slim: drop the last residual stage, layer3 gives 256 channels
        model.layer4 = nn.Identity()
        model.fc = head(256, n_classes)
    else:                      # MobileNet: after the existing FC 1280 of the classifier
        model.classifier[-1] = head(model.classifier[-1].in_features, n_classes)
    return model


def weights_path(design, out="models"):
    return os.path.join(out, f"{design}_{DESIGNS[design]}.pth")


def train(design, data_dir, epochs=30, imgsz=None, out="models", batch=32, lr=1e-3, seed=0, pretrained=True):
    imgsz = imgsz or IMGSZ[design]
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


def load(design, weights=None, imgsz=None):
    """weights=None: untrained model, only for timing/memory before the dataset exists."""
    classes, imgsz = ["A", "B", "C"], imgsz or IMGSZ[design]
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


def source_code(design):
    """The code a future researcher maintains for this design: [this file, the used part of its torchvision file].
    Used = every top-level class/function of the torchvision file with a line that runs while the model is built and
    predicts once, plus the imports. The rest of the file (other family members, e.g. ResNet50-152 in resnet.py, and
    the pretrained-weight metadata classes) is left out."""
    arch = os.path.join(os.path.dirname(torchvision.models.__file__), SOURCE[design])
    ran = set()

    def trace(frame, event, arg):
        if frame.f_code.co_filename == arch:
            ran.add(frame.f_lineno)
            return trace

    sys.settrace(trace)
    try:
        with torch.no_grad():
            build(design, pretrained=False).eval()(torch.zeros(1, 3, IMGSZ[design], IMGSZ[design]))
    finally:
        sys.settrace(None)
    with open(arch, encoding="utf-8") as f:
        src = f.read()
    lines, used = src.splitlines(), []
    for node in ast.parse(src).body:
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        if isinstance(node, (ast.Import, ast.ImportFrom)) or any(start <= n <= node.end_lineno for n in ran):
            used.append("\n".join(lines[start - 1:node.end_lineno]))
    with open(__file__, encoding="utf-8") as f:
        return [f.read(), "\n\n".join(used)]
