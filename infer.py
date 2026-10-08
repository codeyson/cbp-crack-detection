"""
infer.py - run the trained crack-segmentation U-Net on an image, no training code needed.

Large images are cut into overlapping tiles (the model was trained on 256x256 crops),
predicted in small batches, and blended back into one probability map.

Setup on the mini PC:   pip install numpy scipy scikit-image opencv-python-headless
                        + torch (CPU build) + segmentation-models-pytorch  (same versions as Colab)

Self-test (needs only numpy):   python infer.py --selftest
Run on a photo:
    python infer.py --img block.jpg --out results --threads 4      (models/unet_efficientnet-b0_best.pth by default)
"""
import argparse
import os
import time

import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # ImageNet stats: must match training
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def tile_starts(length, tile, stride):
    """Tile start offsets covering [0, length); the last tile is anchored to the far edge."""
    starts = list(range(0, max(length - tile, 0) + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def predict_tiled(img_rgb, predict_fn, tile=256, overlap=64, batch=8):
    """
    img_rgb    : HxWx3 uint8, RGB order (cv2 loads BGR, convert first)
    predict_fn : (N,tile,tile,3) uint8 -> (N,tile,tile) float probabilities
    Overlapping predictions are blended with a triangular window so tile borders
    (where U-Nets are least reliable) carry less weight.
    """
    assert tile % 32 == 0, "tile must be a multiple of 32 for the U-Net"
    assert 0 <= overlap < tile
    h, w = img_rgb.shape[:2]
    ph, pw = max(tile - h, 0), max(tile - w, 0)
    img = np.pad(img_rgb, ((0, ph), (0, pw), (0, 0)), mode="symmetric") if (ph or pw) else img_rgb
    H, W = img.shape[:2]
    stride = tile - overlap
    coords = [(y, x) for y in tile_starts(H, tile, stride) for x in tile_starts(W, tile, stride)]

    tri = np.minimum(np.arange(1, tile + 1), np.arange(tile, 0, -1)).astype(np.float32)
    win = np.outer(tri, tri)
    acc = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        tiles = np.stack([img[y:y + tile, x:x + tile] for y, x in chunk])
        probs = predict_fn(tiles)
        for (y, x), p in zip(chunk, probs):
            acc[y:y + tile, x:x + tile] += p * win
            wsum[y:y + tile, x:x + tile] += win
    return (acc / wsum)[:h, :w]


def load_predict_fn(ckpt, backbone, threads=0, tile=256):
    """Model setup shared by infer.py and pipeline.py: threads (0 = library default), cuda if available,
    load the checkpoint, one warm-up tile. Returns (predict_fn, device)."""
    import torch
    import segmentation_models_pytorch as smp
    if threads > 0:
        torch.set_num_threads(threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.Unet(backbone, encoder_weights=None, in_channels=3, classes=1)  # weights come from ckpt
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model = model.to(device).eval()

    def fn(tiles):
        x = (tiles.astype(np.float32) / 255.0 - MEAN) / STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).to(device)
        with torch.inference_mode():
            p = torch.sigmoid(model(x))[:, 0]
        return p.float().cpu().numpy()
    fn(np.zeros((1, tile, tile, 3), np.uint8))              # warm-up, excluded from timing
    return fn, device


def selftest():
    """Tiling/blending logic only, using a fake predictor. Does NOT test the neural network."""
    rng = np.random.default_rng(0)
    ok_all = True
    for shape in [(500, 733), (256, 256), (300, 300), (100, 120), (260, 1000)]:
        img = rng.integers(0, 256, (*shape, 3), dtype=np.uint8)
        n_tiles = [0]

        def fake(tiles):
            n_tiles[0] += len(tiles)
            return tiles[..., 0].astype(np.float32) / 255.0   # 'model' = red channel

        out = predict_tiled(img, fake, tile=256, overlap=64, batch=5)
        ok = out.shape == shape and np.allclose(out, img[..., 0] / 255.0, atol=1e-5)
        ok_all &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {shape}: output shape {out.shape}, {n_tiles[0]} tiles, blends back to input")
    print("selftest:", "ALL PASS" if ok_all else "FAILURES")
    return ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--img")
    ap.add_argument("--ckpt", default="models/unet_efficientnet-b0_best.pth")
    ap.add_argument("--backbone", default="efficientnet-b0")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = library default)")
    a = ap.parse_args()

    if a.selftest:
        raise SystemExit(0 if selftest() else 1)
    if not a.img:
        ap.error("--img is required")

    import cv2

    bgr = cv2.imread(a.img)
    assert bgr is not None, f"could not read {a.img}"
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    fn, device = load_predict_fn(a.ckpt, a.backbone, a.threads, a.tile)

    t0 = time.time()
    prob = predict_tiled(img, fn, a.tile, a.overlap, a.batch)
    dt = time.time() - t0
    mask = prob > a.thresh
    print(f"device: {device} | image {img.shape[1]}x{img.shape[0]} | inference: {dt:.2f} s")

    os.makedirs(a.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.img))[0]
    cv2.imwrite(f"{a.out}/{stem}_mask.png", (mask * 255).astype(np.uint8))
    over = img.copy(); over[mask] = (255, 0, 0)
    cv2.imwrite(f"{a.out}/{stem}_overlay.png", cv2.cvtColor(over, cv2.COLOR_RGB2BGR))

    print("saved to", a.out)


if __name__ == "__main__":
    main()
