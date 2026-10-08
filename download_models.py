"""
download_models.py - fetch the trained weights from the GitHub Release into models/. Run once after cloning.

    python download_models.py

Files already in models/ are skipped. New weights: upload them to a new Release and change TAG.
"""
import os
import urllib.request

TAG = "models-v1"
URL = f"https://github.com/codeyson/cbp-crack-detection/releases/download/{TAG}/"
FILES = ["best.pt",                          # block detector (default)
         "STABLE_block_yolov8n_best.pt",     # older block detector, use with --weights
         "unet_efficientnet-b0_best.pth",
         "height_detection.pt"]    # crack U-Net

os.makedirs("models", exist_ok=True)
for name in FILES:
    path = os.path.join("models", name)
    if os.path.exists(path):
        print(f"have  {path}")
        continue
    print(f"get   {path} ...", flush=True)
    urllib.request.urlretrieve(URL + name, path + ".part")   # .part: an interrupted download is not mistaken for done
    os.replace(path + ".part", path)
print("models ready")
