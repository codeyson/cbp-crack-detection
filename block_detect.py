"""
block_detect.py - find the 3 blocks in one rig photo with the trained YOLO model, label them Left/Middle/Right
and crop each one for the crack model (objective 1.1). Model: train_block_yolov8.ipynb.

Webcam:     python block_detect.py --cam 0 --weights models/block_yolov8n_best.pt --conf 0.3 --out captures
One photo:  python block_detect.py --img rig.jpg --weights models/block_yolov8n_best.pt
Evaluate:   python block_detect.py --yolo_dir "CBP Block Detection.v3i.yolov8" --weights models/block_yolov8n_best.pt
Self-test:  python block_detect.py --selftest

In the pipeline:
    res = detect_blocks(img_bgr, YOLO(weights))
    if res["status"] != "ok": ask the operator to re-place the blocks (res["message"])
    for b in res["blocks"]:   # run the crack model on b["crop"], then
        crack_length_mm(mask, mm_per_px, valid_region=b["interior"])

Boxes are axis-aligned: place blocks square to the camera, or crops include background.
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np


def detect_blocks(img, model, n_expected=3, conf=0.5, erode_px=6):
    """Returns {"status", "message", "blocks"}. ALWAYS check status == "ok" before using blocks.
    status: ok | wrong_count | touches_border. Each block: label, box (x1, y1, x2, y2), crop, interior, touches_border.
    erode_px: interior mask is the box shrunk by this much, to keep block edges out of the crack measurement."""
    H, W = img.shape[:2]
    boxes = np.asarray(model.predict(img, conf=conf, verbose=False)[0].boxes.xyxy.cpu()).round().astype(int)
    boxes = sorted((tuple(np.clip(b, 0, [W, H, W, H])) for b in boxes), key=lambda b: b[0])  # left to right
    names = ["Left", "Middle", "Right"] if n_expected == 3 and len(boxes) == 3 else [f"Block {i + 1}" for i in range(len(boxes))]
    blocks = []
    for name, (x1, y1, x2, y2) in zip(names, boxes):
        interior = np.zeros((y2 - y1, x2 - x1), bool)
        interior[erode_px:-erode_px or None, erode_px:-erode_px or None] = True
        blocks.append(dict(label=name, box=(x1, y1, x2, y2), crop=img[y1:y2, x1:x2].copy(), interior=interior,
                           touches_border=x1 <= 1 or y1 <= 1 or x2 >= W - 1 or y2 >= H - 1))
    res = dict(status="ok", message="", blocks=blocks)
    if len(blocks) != n_expected:
        res.update(status="wrong_count", message=f"Found {len(blocks)} blocks, expected {n_expected}.")
    elif any(b["touches_border"] for b in blocks):
        res.update(status="touches_border", message="A block touches the image border, so it may be cut off.")
    return res


def draw_overlay(img, res):
    vis = img.copy()
    col = (0, 200, 0) if res["status"] == "ok" else (0, 0, 255)
    s = max(0.6, vis.shape[1] / 1500.0)
    th = max(2, round(s * 2))
    for b in res["blocks"]:
        x1, y1, x2, y2 = b["box"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, th)
        cv2.putText(vis, b["label"], (x1 + 10, y1 + int(40 * s)), cv2.FONT_HERSHEY_SIMPLEX, s, col, th)
    cv2.putText(vis, f"{res['status']} ({len(res['blocks'])} found)", (10, int(35 * s)), cv2.FONT_HERSHEY_SIMPLEX, s, col, th)
    return vis


def save(img, res, out, stem):
    os.makedirs(out, exist_ok=True)
    cv2.imwrite(f"{out}/{stem}_overlay.png", draw_overlay(img, res))
    for b in res["blocks"]:
        cv2.imwrite(f"{out}/{stem}_{b['label']}_crop.png", b["crop"])
        cv2.imwrite(f"{out}/{stem}_{b['label']}_interior.png", b["interior"].astype(np.uint8) * 255)
    with open(f"{out}/{stem}_summary.json", "w") as f:
        json.dump(dict(status=res["status"], message=res["message"],
                       blocks=[dict(label=b["label"], box=[int(v) for v in b["box"]]) for b in res["blocks"]]), f, indent=1)
    print(f"{res['status']}: {res['message'] or 'ok'} -> {out}/{stem}_*")


def open_camera(cam, width=2560, height=1440):
    """Returns (cap, first_frame). Also used by dimensions.py."""
    cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # most webcams only give full resolution as MJPG
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)                              # drivers may ignore this; check focus by eye
    ok, frame = cap.read()
    assert ok, f"could not read camera {cam}"
    print(f"camera delivers {frame.shape[1]}x{frame.shape[0]} (asked {width}x{height})")
    return cap, frame


def run_camera(cam, out, **kw):
    """Live preview. SPACE saves the raw frame and the crops (only when status is ok), q quits."""
    cap, frame = open_camera(cam)
    ok = True
    while ok:
        res = detect_blocks(frame, **kw)
        vis = draw_overlay(frame, res)
        cv2.imshow("blocks (SPACE save, q quit)", cv2.resize(vis, None, fx=1280 / vis.shape[1], fy=1280 / vis.shape[1]))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            if res["status"] != "ok":
                print(f"not saved: {res['message']}")
            else:
                stem = time.strftime("%Y%m%d_%H%M%S")
                os.makedirs(out, exist_ok=True)
                cv2.imwrite(f"{out}/{stem}_raw.png", frame)            # raw frame kept for offline re-runs
                save(frame, res, out, stem)
        ok, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()


def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih
    return iw * ih / union if union > 0 else 0.0


def evaluate_yolo_dir(root, iou_ok=0.8, **kw):
    """Pass rule per photo: status ok, as many blocks as annotated, each annotated box matched with IoU >= iou_ok.
    root: a Roboflow "YOLOv8" export (*/images + */labels). Returns the number of passing photos."""
    paths = sorted(glob.glob(os.path.join(root, "**", "images", "*.*"), recursive=True))
    passed = 0
    for ip in paths:
        img = cv2.imread(ip)
        H, W = img.shape[:2]
        lp = os.path.join(os.path.dirname(os.path.dirname(ip)), "labels", os.path.splitext(os.path.basename(ip))[0] + ".txt")
        gt = []
        for line in open(lp).read().splitlines():
            p = line.split()
            if len(p) == 5:                                          # class cx cy w h, normalised
                cx, cy, w, h = (float(v) for v in p[1:])
                gt.append(((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
        res = detect_blocks(img, **kw)
        worst = min((max((iou(g, b["box"]) for b in res["blocks"]), default=0.0) for g in gt), default=0.0)
        ok = res["status"] == "ok" and len(res["blocks"]) == len(gt) and worst >= iou_ok
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {os.path.basename(ip)}: {res['status']}, "
              f"found {len(res['blocks'])}/{len(gt)}, worst IoU {worst:.3f}")
    print(f"{passed} of {len(paths)} images: all blocks found with IoU >= {iou_ok}")
    return passed, len(paths)


def selftest():
    from types import SimpleNamespace as NS

    class FakeYOLO:                                                  # mimics model.predict(...)[0].boxes.xyxy.cpu()
        def __init__(self, boxes): self.boxes = np.array(boxes, np.float32)
        def predict(self, img, conf, verbose):
            return [NS(boxes=NS(xyxy=NS(cpu=lambda: self.boxes)))]

    img = np.zeros((900, 1600, 3), np.uint8)
    row = [[1150, 375, 1450, 525], [150, 375, 450, 525], [650, 375, 950, 525]]   # out of order on purpose
    r = detect_blocks(img, FakeYOLO(row))
    assert r["status"] == "ok", r
    assert [b["label"] for b in r["blocks"]] == ["Left", "Middle", "Right"]
    assert all(b["crop"].shape == (150, 300, 3) for b in r["blocks"])
    assert r["blocks"][0]["interior"].sum() == (150 - 12) * (300 - 12)          # shrunk 6 px on every side
    assert detect_blocks(img, FakeYOLO(row[:2]))["status"] == "wrong_count"
    assert detect_blocks(img, FakeYOLO(row[:2] + [[0, 375, 100, 525]]))["status"] == "touches_border"
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1 and iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0
    print("selftest: ALL PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--weights", help="trained YOLO block detector, e.g. models/block_yolov8n_best.pt")
    ap.add_argument("--cam", type=int, help="webcam index for live mode, e.g. 0")
    ap.add_argument("--img")
    ap.add_argument("--yolo_dir", help="Roboflow YOLOv8 export: compare detections to the annotated boxes")
    ap.add_argument("--out", default="blocks_out")
    ap.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--erode_px", type=int, default=6)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.weights:
        ap.error("--weights is required")
    from ultralytics import YOLO
    kw = dict(model=YOLO(a.weights), n_expected=a.n, conf=a.conf, erode_px=a.erode_px)
    if a.yolo_dir:
        passed, total = evaluate_yolo_dir(a.yolo_dir, **kw)
        raise SystemExit(0 if total and passed == total else 1)
    if a.cam is not None:
        return run_camera(a.cam, a.out, **kw)
    if not a.img:
        ap.error("--img, --cam or --yolo_dir is required")
    img = cv2.imread(a.img)
    assert img is not None, f"could not read {a.img}"
    res = detect_blocks(img, **kw)
    save(img, res, a.out, os.path.splitext(os.path.basename(a.img))[0])
    raise SystemExit(0 if res["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
