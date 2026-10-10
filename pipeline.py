"""
pipeline.py - one capture -> 3 blocks -> dimensions + crack mask + crack length per block (1.1 + 1.2 + 1.3)
-> grade per block (optional, --grader: needs a design trained on the A/B/C dataset).

    detect + crop (block_detect) -> L/W/H (dimensions.measure) -> crack U-Net on each crop (infer)
    -> crack length on the block interior (crack_length) -> grade: Reject if out of tolerance, else the
    classifier (evaluation/designs.py) on the crop -> overlay window + saved record

Live:     python pipeline.py --cam 0          (models/best.pt + models/unet_efficientnet-b0_best.pth by default)
          SPACE runs everything (laser off), L reruns with the current frame as the laser-ON frame, q quits.
          --side_cam 1: second camera, side view, measures height (models/height_detection.pt); SPACE grabs both.
Offline:  python pipeline.py --img captures/xxx_raw.png [--laser_on on.png] [--side_img side.png]
Grade:    python pipeline.py --img ... --grader d1     (models/d1_mobilenet_v3_large.pth from evaluation/train.py)
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
from dimensions import draw as draw_dims, draw_side, load_calib, measure, side_height, side_heights, undistort
from infer import predict_tiled


def run(off, model, predict_fn, calib, laser_on=None, scale_ref=None, height_mm=None, nominal=None, tol=None,
        thresh=0.5, side=None, side_model=None, grade_fn=None, **det_kw):
    """off = laser-OFF frame (BGR). Returns measure()-style dict + per-block crack fields, "dims", "timings".
    Crack model only runs when status == "ok". grade_fn(crop_bgr) -> "A"|"B"|"C": adds b["grade"],
    "Reject" when the block failed the dimension tolerance."""
    t0 = time.perf_counter()
    res = dict(status="no_scale")
    if scale_ref or "mm_per_px_bed" in calib:
        res = measure(off, model, calib, scale_ref=scale_ref, laser_on=laser_on, height_mm=height_mm,
                      nominal=nominal, tol=tol, side=side, side_model=side_model, **det_kw)
    res["dims"] = res["status"] not in ("no_scale", "no_marker")
    if not res["dims"]:
        why = "ArUco marker not found" if res["status"] == "no_marker" else "no scale"
        img = undistort(off, calib)
        res = dict(detect_blocks(img, model, **det_kw), image=img, dims=False,
                   warnings=[f"{why}: dimensions skipped, crack length in px only"])
        if side is not None:                                          # height needs no top-view scale
            res["warnings"] += side_height(res, side, side_model, calib, det_kw.get("conf", 0.5))
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
        if grade_fn:   # same full crop that save() writes as <label>_crop.png, i.e. what the dataset is sorted from
            t = time.perf_counter()
            b["grade"] = "Reject" if b.get("tolerance") == "fail" else grade_fn(b["crop"])
            res["timings"][f"grade_{b['label']}_s"] = round(time.perf_counter() - t, 3)
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
        lines = [f"grade {b['grade']}"] if "grade" in b else []
        if "crack_px" in b:
            t = f"crack {b['crack_mm']:.1f} mm" if b["crack_mm"] is not None else f"crack {b['crack_px']:.0f} px"
            lines.append(f"{t} ({b['crack_pieces']} pcs)")
        if b.get("H_mm") is not None and not res["dims"]:            # dims mode already shows H
            lines.append(f"H {b['H_mm']:.1f} mm")
        elif b.get("H_mm") is None and b.get("side_px") is not None:  # side camera not calibrated yet
            lines.append(f"side {b['side_px']:.1f} px")
        for i, t in enumerate(lines):
            cv2.putText(vis, t, (b["box"][0] + 10, b["box"][1] + int((40 + 35 * (row + i)) * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 255), th)
    return vis


def save(res, off, laser_on, out, meta):
    d = os.path.join(out, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(d, exist_ok=True)
    cv2.imwrite(f"{d}/off.png", off)
    if laser_on is not None:
        cv2.imwrite(f"{d}/laser.png", laser_on)
    if "side" in res:
        cv2.imwrite(f"{d}/side.png", res["side"]["image"])
        cv2.imwrite(f"{d}/side_overlay.png", draw_side(res["side"]["image"], res["side"]))
    cv2.imwrite(f"{d}/overlay.png", draw(res))
    keep = ("label", "L_mm", "W_mm", "H_mm", "H_source", "shift_px", "side_px", "fill", "tolerance", "reasons",
            "crack_mm", "crack_px", "crack_pieces", "crack_mm_per_px", "grade")
    blocks = []
    for b in res["blocks"]:
        cv2.imwrite(f"{d}/{b['label']}_crop.png", b["crop"])
        cv2.imwrite(f"{d}/{b['label']}_mask.png", (b["mask"] & b["interior"]).astype(np.uint8) * 255)
        blocks.append(dict({k: b.get(k) for k in keep}, box=[int(v) for v in b["box"]]))
    with open(f"{d}/summary.json", "w") as f:
        json.dump(dict(meta, status=res["status"], message=res["message"], warnings=res["warnings"],
                       timings=res["timings"], blocks=blocks), f, indent=1, default=float)
    for b in blocks:
        side = f" (side {b['side_px']:.1f} px)" if b["side_px"] is not None else ""
        grade = f" grade {b['grade']} " if b["grade"] is not None else ""
        print(f"  {b['label']}:{grade}  L {b['L_mm']}  W {b['W_mm']}  H {b['H_mm']}{side}  {b['tolerance']}  "
              f"crack {b['crack_mm'] if b['crack_mm'] is not None else str(round(b['crack_px'])) + ' px'}"
              f" ({b['crack_pieces']} pcs)")
    for w in res["warnings"]:
        print("  WARNING:", w)
    print(f"timings {res['timings']} -> {d}")


def run_camera(cam, out, meta, side_cam=None, **kw):
    """Preview shows the YOLO boxes. SPACE: full pipeline on the laser-OFF frame (+ side frame). L: rerun with
    laser ON. q quits. side_cam: second camera, side view, preview shows its measured edges and height in px."""
    cap, frame = open_camera(cam)
    side_cap, side_frame = open_camera(side_cam) if side_cam is not None else (None, None)
    off, side, ok = None, None, True
    det_kw = {k: kw[k] for k in ("model", "conf")}
    fx = 1280 / frame.shape[1]
    while ok:
        cv2.imshow("pipeline (SPACE run, L laser frame, q quit)",
                   cv2.resize(draw_overlay(frame, detect_blocks(frame, **det_kw)), None, fx=fx, fy=fx))
        if side_cap is not None:
            sv = draw_side(side_frame, side_heights(side_frame, kw["side_model"], conf=kw["conf"],
                                                    flip=kw["calib"].get("side_flip", False)))
            cv2.imshow("side camera (height)", cv2.resize(sv, None, fx=960 / sv.shape[1], fy=960 / sv.shape[1]))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("l") and off is None:
            print("press SPACE (laser off) first")
        elif key in (ord(" "), ord("l")):
            if key == ord(" "):
                off = frame.copy()
                side = side_frame.copy() if side_cap is not None else None
            on = frame.copy() if key == ord("l") else None
            res = run(off, laser_on=on, side=side, **kw)
            if res["status"] == "ok":
                save(res, off, on, out, meta)
            else:
                print(f"not saved: {res['message']}  {res['warnings']}")
            cv2.imshow("result", cv2.resize(draw(res), None, fx=fx, fy=fx))
        ok, frame = cap.read()
        if side_cap is not None:
            ok, side_frame = side_cap.read()
    cap.release()
    if side_cap is not None:
        side_cap.release()
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
    assert draw(r).shape == img.shape and "grade" not in mid

    r = run(img, yolo, dark, dict(mm_per_px_bed=0.5), grade_fn=lambda crop: "B",   # 150 mm blocks vs 100 +/- 1
            nominal=(100, 100, 60), tol=(1, 1, 1))
    assert [b["grade"] for b in r["blocks"]] == ["Reject"] * 3, [(b["tolerance"], b["reasons"]) for b in r["blocks"]]
    r = run(img, yolo, dark, dict(mm_per_px_bed=0.5), grade_fn=lambda crop: "B")   # no standard: classifier grades
    assert [b["grade"] for b in r["blocks"]] == ["B"] * 3 and "grade_Middle_s" in r["timings"]
    assert draw(r).shape == img.shape
    graded = r

    r = run(img, yolo, dark, {})                                      # no scale: px only, no dims
    assert not r["dims"] and r["blocks"][1]["crack_mm"] is None and abs(r["blocks"][1]["crack_px"] - 199) < 6
    assert draw(r).shape == img.shape

    r = run(img, FakeYOLO(boxes[:2]), dark, {})                      # wrong count: crack model skipped
    assert r["status"] == "wrong_count" and "crack_px" not in r["blocks"][0]

    side = np.full((800, 1600, 3), 200, np.uint8)                     # side view: faces 250 px tall
    sboxes = [[x, 350, x + 400, 600] for x in (100, 600, 1100)]
    for x1, _, x2, _ in sboxes:
        side[350:600, x1:x2] = np.clip(rng.normal(95, 25, (250, 400, 3)), 0, 255)
    r = run(img, yolo, dark, dict(mm_per_px_bed=0.5, side_coef=[0.2, 0.0]), side=side, side_model=FakeYOLO(sboxes))
    assert [b["H_source"] for b in r["blocks"]] == ["side"] * 3 and all(abs(b["H_mm"] - 50) < 0.5 for b in r["blocks"])
    assert "side" in r and draw(r).shape == img.shape

    r = run(img, yolo, dark, dict(side_coef=[0.2, 0.0]), side=side, side_model=FakeYOLO(sboxes))   # no top scale
    assert not r["dims"] and "side" in r, r["warnings"]
    assert all(b["H_source"] == "side" and abs(b["H_mm"] - 50) < 0.5 for b in r["blocks"]), [b.get("H_mm") for b in r["blocks"]]
    r = run(img, yolo, dark, {}, side=side, side_model=FakeYOLO(sboxes))                             # side not calibrated
    assert all(b.get("H_mm") is None and abs(b["side_px"] - 250) < 3 for b in r["blocks"])
    assert draw(r).shape == img.shape

    import tempfile                                                   # what is saved after a capture
    d = tempfile.mkdtemp()
    save(r, img, None, d, {})
    d = os.path.join(d, os.listdir(d)[0])
    with open(f"{d}/summary.json") as f:
        saved = json.load(f)["blocks"]
    assert all(abs(b["side_px"] - 250) < 3 for b in saved), saved
    assert os.path.exists(f"{d}/side.png") and os.path.exists(f"{d}/side_overlay.png")
    assert all(b["grade"] is None for b in saved)

    d = tempfile.mkdtemp()                                            # grade is saved
    save(graded, img, None, d, {})
    with open(os.path.join(d, os.listdir(d)[0], "summary.json")) as f:
        assert [b["grade"] for b in json.load(f)["blocks"]] == ["B"] * 3
    try:                                                              # the real (untrained) classifier plugs in
        from evaluation.designs import load as load_grader
    except ImportError:
        print("torchvision not installed: real-classifier check skipped")
    else:
        r = run(img, yolo, dark, dict(mm_per_px_bed=0.5), grade_fn=load_grader("d1", None))
        assert all(b["grade"] in ("A", "B", "C") for b in r["blocks"]), [b["grade"] for b in r["blocks"]]
    print("selftest: ALL PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cam", type=int)
    ap.add_argument("--img", help="laser-OFF photo")
    ap.add_argument("--laser_on", help="same scene, laser ON (for height)")
    ap.add_argument("--side_cam", type=int, help="second webcam index, side view (height)")
    ap.add_argument("--side_img", help="side-view photo of the same scene (height)")
    ap.add_argument("--side_weights", default="models/height_detection.pt", help="side-view YOLO")
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
    ap.add_argument("--grader", choices=("d1", "d2", "d3"), help="A/B/C classifier design (evaluation/designs.py)")
    ap.add_argument("--grader_weights", help="default: the design's file in models/ (from evaluation/train.py)")
    ap.add_argument("--out", default="pipeline_out")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    use_side = a.side_cam is not None or a.side_img
    for p in (a.weights, a.ckpt) + ((a.side_weights,) if use_side else ()):
        if not os.path.exists(p):
            ap.error(f"{p} not found: run  python download_models.py")
    if a.grader:   # never grade with an untrained model
        from evaluation.designs import load as load_grader, weights_path
        a.grader_weights = a.grader_weights or weights_path(a.grader)
        if not os.path.exists(a.grader_weights):
            ap.error(f"{a.grader_weights} not found: train it on the A/B/C dataset first "
                     f"(python evaluation/train.py --design {a.grader})")
    from ultralytics import YOLO
    from infer import load_predict_fn
    fn, device = load_predict_fn(a.ckpt, a.backbone, a.threads)
    triple = lambda s: tuple(map(float, s.split(","))) if s else None
    kw = dict(model=YOLO(a.weights), predict_fn=fn, calib=load_calib(a.calib), scale_ref=a.scale_ref,
              height_mm=a.height_mm, nominal=triple(a.nominal), tol=triple(a.tol), thresh=a.thresh,
              conf=a.conf, edge_frac=a.edge_frac, side_model=YOLO(a.side_weights) if use_side else None,
              grade_fn=load_grader(a.grader, a.grader_weights) if a.grader else None)
    meta = dict(weights=os.path.basename(a.weights), ckpt=os.path.basename(a.ckpt), backbone=a.backbone,
                thresh=a.thresh, edge_frac=a.edge_frac, device=device,
                **({"side_weights": os.path.basename(a.side_weights)} if use_side else {}),
                **({"grader": a.grader, "grader_weights": os.path.basename(a.grader_weights)} if a.grader else {}))
    if a.cam is not None:
        return run_camera(a.cam, a.out, meta, side_cam=a.side_cam, **kw)
    if not a.img:
        ap.error("--img or --cam is required")
    off = cv2.imread(a.img)
    assert off is not None, f"could not read {a.img}"
    on = cv2.imread(a.laser_on) if a.laser_on else None
    side = cv2.imread(a.side_img) if a.side_img else None
    assert a.side_img is None or side is not None, f"could not read {a.side_img}"
    res = run(off, laser_on=on, side=side, **kw)
    if res["status"] != "ok":
        print(f"{res['status']}: {res['message']}  {res['warnings']}")
        raise SystemExit(1)
    save(res, off, on, a.out, meta)


if __name__ == "__main__":
    main()
