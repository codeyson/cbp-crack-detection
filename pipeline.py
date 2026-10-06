"""
pipeline.py - one capture -> 3 blocks -> dimensions + crack mask + crack length per block (1.1 + 1.2 + 1.3).
No grading yet (thresholds not signed off).

    detect + crop (block_detect) -> L/W/H (dimensions.measure) -> crack U-Net on each crop (infer)
    -> crack length on the block interior (crack_length) -> overlay window + saved record

Live:     python pipeline.py --cam 0          (models/best.pt + models/unet_efficientnet-b0_best.pth by default)
          SPACE runs everything (laser off), L reruns with the current frame as the laser-ON frame, q quits.
Offline:  python pipeline.py --img captures/xxx_raw.png [--laser_on on.png]
Self-test: python pipeline.py --selftest

Scale: --scale_ref (ArUco in frame) or mm_per_px_bed in --calib. Neither: dims skipped, crack length in px only.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from block_detect import detect_blocks, draw_overlay, open_camera
from crack_length import crack_length_mm
from dimensions import draw as draw_dims, load_calib, measure, undistort
from infer import predict_tiled


def run(off, model, predict_fn, calib, laser_on=None, scale_ref=None, height_mm=None, nominal=None, tol=None,
        thresh=0.5, **det_kw):
    """off = laser-OFF frame (BGR). Returns measure()-style dict + per-block crack fields, "dims", "timings".
    Crack model only runs when status == "ok"."""
    t0 = time.perf_counter()
    res = dict(status="no_scale")
    if scale_ref or "mm_per_px_bed" in calib:
        res = measure(off, model, calib, scale_ref=scale_ref, laser_on=laser_on, height_mm=height_mm,
                      nominal=nominal, tol=tol, **det_kw)
    res["dims"] = res["status"] not in ("no_scale", "no_marker")
    if not res["dims"]:
        why = "ArUco marker not found" if res["status"] == "no_marker" else "no scale"
        img = undistort(off, calib)
        res = dict(detect_blocks(img, model, **det_kw), image=img, dims=False,
                   warnings=[f"{why}: dimensions skipped, crack length in px only"])
    res["timings"] = {"detect_dims_s": round(time.perf_counter() - t0, 3)}
    if res["status"] != "ok":
        return res

    Z = res.get("cam_height_mm")
    for b in res["blocks"]:
        t = time.perf_counter()
        prob = predict_tiled(cv2.cvtColor(b["crop"], cv2.COLOR_BGR2RGB), predict_fn)
        b["mask"] = prob > thresh
        h = b.get("H_mm") or height_mm
        mpp = None
        if res["dims"]:   # crack lies on the top face: bed scale * (Z - h) / Z
            mpp = res["mm_per_px_bed"] *((Z - h) / Z if Z and h else 1.0)
        c = crack_length_mm(b["mask"], mpp or 1.0, valid_region=b["interior"])
        b.update(crack_mm=c["length_mm"] if mpp else None, crack_px=c["length_px"], crack_pieces=c["n_pieces"],
                 crack_mm_per_px=mpp)
        res["timings"][f"crack_{b['label']}_s"] = round(time.perf_counter() - t, 3)
    if res["dims"] and not all(Z and (b.get("H_mm") or height_mm) for b in res["blocks"]):
        res["warnings"].append("crack mm uses the bed scale (no camera height or block height): reads too long")
    return res


def draw(res):
    img = res["image"].copy()
    for b in res["blocks"]:
        if "mask" in b:
            x1, y1, x2, y2 = b["box"]
            img[y1:y2, x1:x2][b["mask"] & b["interior"]] = (0, 0, 255)
    vis = draw_dims(img, res) if res["dims"] else draw_overlay(img, res)
    s = max(0.6, vis.shape[1] / 1500.0)
    th = max(2, round(s * 2))
    row = 4 if res["dims"] else 1                                     # below the text the draw function wrote
    for b in res["blocks"]:
        if "crack_px" in b:
            t = f"crack {b['crack_mm']:.1f} mm" if b["crack_mm"] is not None else f"crack {b['crack_px']:.0f} px"
            cv2.putText(vis, f"{t} ({b['crack_pieces']} pcs)", (b["box"][0] + 10, b["box"][1] + int((40 + 35 * row) * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 255), th)
    return vis


def save(res, off, laser_on, out, meta):
    d = os.path.join(out, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(d, exist_ok=True)
    cv2.imwrite(f"{d}/off.png", off)
    if laser_on is not None:
        cv2.imwrite(f"{d}/laser.png", laser_on)
    cv2.imwrite(f"{d}/overlay.png", draw(res))
    keep = ("label", "L_mm", "W_mm", "H_mm", "H_source", "shift_px", "fill", "tolerance", "reasons",
            "crack_mm", "crack_px", "crack_pieces", "crack_mm_per_px")
    blocks = []
    for b in res["blocks"]:
        cv2.imwrite(f"{d}/{b['label']}_crop.png", b["crop"])
        cv2.imwrite(f"{d}/{b['label']}_mask.png", (b["mask"] & b["interior"]).astype(np.uint8) * 255)
        blocks.append(dict({k: b.get(k) for k in keep}, box=[int(v) for v in b["box"]]))
    with open(f"{d}/summary.json", "w") as f:
        json.dump(dict(meta, status=res["status"], message=res["message"], warnings=res["warnings"],
                       timings=res["timings"], blocks=blocks), f, indent=1, default=float)
    for b in blocks:
        print(f"  {b['label']}: L {b['L_mm']}  W {b['W_mm']}  H {b['H_mm']}  {b['tolerance']}  "
              f"crack {b['crack_mm'] if b['crack_mm'] is not None else str(round(b['crack_px'])) + ' px'}"
              f" ({b['crack_pieces']} pcs)")
    for w in res["warnings"]:
        print("  WARNING:", w)
    print(f"timings {res['timings']} -> {d}")


def run_camera(cam, out, meta, **kw):
    """Preview shows the YOLO boxes. SPACE: full pipeline on the laser-OFF frame. L: rerun with laser ON. q quits."""
    cap, frame = open_camera(cam)
    off, ok = None, True
    det_kw = {k: kw[k] for k in ("model", "conf")}
    fx = 1280 / frame.shape[1]
    while ok:
        cv2.imshow("pipeline (SPACE run, L laser frame, q quit)",
                   cv2.resize(draw_overlay(frame, detect_blocks(frame, **det_kw)), None, fx=fx, fy=fx))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("l") and off is None:
            print("press SPACE (laser off) first")
        elif key in (ord(" "), ord("l")):
            if key == ord(" "):
                off = frame.copy()
            on = frame.copy() if key == ord("l") else None
            res = run(off, laser_on=on, **kw)
            if res["status"] == "ok":
                save(res, off, on, out, meta)
            else:
                print(f"not saved: {res['message']}  {res['warnings']}")
            cv2.imshow("result", cv2.resize(draw(res), None, fx=fx, fy=fx))
        ok, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()


def selftest():
    from block_detect import FakeYOLO

    rng = np.random.default_rng(0)
    img = np.clip(rng.normal(60, 4, (900, 1600, 3)), 0, 255).astype(np.uint8)
    boxes = [[150, 300, 450, 600], [650, 300, 950, 600], [1150, 300, 1450, 600]]
    for x1, y1, x2, y2 in boxes:
        img[y1:y2, x1:x2] = np.clip(rng.normal(170, 4, (y2 - y1, x2 - x1, 3)), 0, 255)
    img[449:452, 700:900] = 20                                        # 200 px crack on Middle
    dark = lambda tiles: (tiles[..., 0] < 100).astype(np.float32)     # fake U-Net: dark = crack
    yolo = FakeYOLO(boxes)

    r = run(img, yolo, dark, dict(mm_per_px_bed=0.5))
    assert r["status"] == "ok" and r["dims"], r["status"]
    assert [b["label"] for b in r["blocks"]] == ["Left", "Middle", "Right"]
    left, mid, right = r["blocks"]
    assert abs(mid["crack_mm"] - 199 * 0.5) < 199 * 0.5 * 0.03, mid["crack_mm"]
    assert left["crack_px"] == 0 and right["crack_px"] == 0
    assert abs(mid["L_mm"] - 150) < 2 and abs(mid["W_mm"] - 150) < 2, (mid["L_mm"], mid["W_mm"])
    assert draw(r).shape == img.shape

    r = run(img, yolo, dark, {})                                      # no scale: px only, no dims
    assert not r["dims"] and r["blocks"][1]["crack_mm"] is None and abs(r["blocks"][1]["crack_px"] - 199) < 6
    assert draw(r).shape == img.shape

    r = run(img, FakeYOLO(boxes[:2]), dark, {})                      # wrong count: crack model skipped
    assert r["status"] == "wrong_count" and "crack_px" not in r["blocks"][0]
    print("selftest: ALL PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cam", type=int)
    ap.add_argument("--img", help="laser-OFF photo")
    ap.add_argument("--laser_on", help="same scene, laser ON (for height)")
    ap.add_argument("--weights", default="models/best.pt", help="YOLO block detector")
    ap.add_argument("--ckpt", default="models/unet_efficientnet-b0_best.pth", help="crack U-Net checkpoint")
    ap.add_argument("--backbone", default="efficientnet-b0")
    ap.add_argument("--calib", default="calib.json")
    ap.add_argument("--scale_ref", type=float, help="ArUco marker black-square side in mm (measured)")
    ap.add_argument("--height_mm", type=float, help="assumed block height when there is no laser frame")
    ap.add_argument("--nominal", help="L,W,H mm (from the standard)")
    ap.add_argument("--tol", help="L,W,H tolerance mm (from the standard)")
    ap.add_argument("--conf", type=float, default=0.3, help="YOLO confidence threshold")
    ap.add_argument("--thresh", type=float, default=0.5, help="crack probability threshold")
    ap.add_argument("--edge_frac", type=float, default=0.10, help="ignore this fraction of each box side (walls, shadows)")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = library default)")
    ap.add_argument("--out", default="pipeline_out")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    for p in (a.weights, a.ckpt):
        if not os.path.exists(p):
            ap.error(f"{p} not found: run  python download_models.py")
    from ultralytics import YOLO
    from infer import load_predict_fn
    fn, device = load_predict_fn(a.ckpt, a.backbone, a.threads)
    triple = lambda s: tuple(map(float, s.split(","))) if s else None
    kw = dict(model=YOLO(a.weights), predict_fn=fn, calib=load_calib(a.calib), scale_ref=a.scale_ref,
              height_mm=a.height_mm, nominal=triple(a.nominal), tol=triple(a.tol), thresh=a.thresh,
              conf=a.conf, edge_frac=a.edge_frac)
    meta = dict(weights=os.path.basename(a.weights), ckpt=os.path.basename(a.ckpt), backbone=a.backbone,
                thresh=a.thresh, edge_frac=a.edge_frac, device=device)
    if a.cam is not None:
        return run_camera(a.cam, a.out, meta, **kw)
    if not a.img:
        ap.error("--img or --cam is required")
    off = cv2.imread(a.img)
    assert off is not None, f"could not read {a.img}"
    on = cv2.imread(a.laser_on) if a.laser_on else None
    res = run(off, laser_on=on, **kw)
    if res["status"] != "ok":
        print(f"{res['status']}: {res['message']}  {res['warnings']}")
        raise SystemExit(1)
    save(res, off, on, a.out, meta)


if __name__ == "__main__":
    main()
