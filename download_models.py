"""
download_models.py - fetch the trained weights (and the evaluation test set) from the GitHub Releases. Run once
after cloning.

    python download_models.py

Files already present are skipped. New weights: upload them to a new Release and change the tag below.
"""
import os
import urllib.request
import zipfile

REPO = "https://github.com/codeyson/cbp-crack-detection/releases/download"
FILES = [("models-v1", "best.pt"),                          # block detector (default)
         ("models-v1", "STABLE_block_yolov8n_best.pt"),     # older block detector, use with --weights
         ("models-v1", "unet_efficientnet-b0_best.pth"),    # crack U-Net
         ("models-v1", "height_detection.pt"),              # side-camera height detector
         ("designs-v1", "d1_mobilenet_v3_large.pth"),       # evaluation designs (evaluation/README.md)
         ("designs-v1", "d2_efficientnet_b0.pth"),
         ("designs-v1", "d3_resnet18.pth")]
TEST = ("designs-v1", "dataset_test.zip")                   # -> dataset/test/{A,B,C}/, for evaluation/evaluate.py


def get(tag, name, path):
    print(f"get   {path} ...", flush=True)
    urllib.request.urlretrieve(f"{REPO}/{tag}/{name}", path + ".part")   # .part: an interrupted download is not done
    os.replace(path + ".part", path)


os.makedirs("models", exist_ok=True)
for tag, name in FILES:
    path = os.path.join("models", name)
    if os.path.exists(path):
        print(f"have  {path}")
    else:
        get(tag, name, path)

if os.path.isdir(os.path.join("dataset", "test")):
    print("have  dataset/test")
else:
    get(*TEST, TEST[1])
    with zipfile.ZipFile(TEST[1]) as z:
        z.extractall(".")                                   # the zip holds dataset/test/...
    os.remove(TEST[1])
print("models ready")
