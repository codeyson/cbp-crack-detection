"""
pipeline.py - one capture -> 3 blocks -> dimensions + weight + crack length + grade per block.

    detect + crop (block_detect) -> L/W/H (dimensions.measure) -> weight (loadcell) -> crack U-Net on each crop
    (infer) -> crack length on the block interior (crack_length) -> grade classifier on each crop
    (evaluation/designs.py, default D1 = trade-off winner) -> overlay window + saved record
    grade: Reject when the L/W/H or weight tolerance check fails, else the classifier's A/B/C.

Live:     python pipeline.py --cam 0          (models/best.pt + models/unet_efficientnet-b0_best.pth by default)
          SPACE runs everything (laser off), L reruns with the current frame as the laser-ON frame, q quits.
          --side_cam 1: second camera, side view, measures height (models/height_detection.pt); SPACE grabs both.
          --design d1|d2|d3|none: grade classifier (models/<design>_*.pth, default d1).
Offline:  python pipeline.py --img captures/xxx_raw.png [--laser_on on.png] [--side_img side.png]
Self-test: python pipeline.py --selftest

Scale: --scale_ref (ArUco in frame) or mm_per_px_bed in --calib. Neither: dims skipped, crack length in px only.
"""
import argparse
import json
import textwrap
import os
import time

import cv2
import numpy as np

from block_detect import detect_blocks, draw_overlay, open_camera
from crack_length import crack_length_mm
from dimensions import draw as draw_dims, draw_side, load_calib, measure, side_height, side_heights, undistort
from infer import predict_tiled


def run(off, model, predict_fn, calib, laser_on=None, scale_ref=None, height_mm=None, nominal=None, tol=None,
        thresh=0.5, side=None, side_model=None, weights=None, nominal_g=None, tol_g=None, classify=None, **det_kw):
    """off = laser-OFF frame (BGR). Returns measure()-style dict + per-block crack fields, "dims", "timings".
    weights: loadcell.read_weights() dict, read at capture time -> weight_g per block, tolerance fail when
    |weight_g - nominal_g| > tol_g. classify: image (BGR) -> "A"|"B"|"C" (designs.load), run on patches of the block
    interior (grade_patches); grade = "Reject" when the tolerance check failed, else the worst patch. Crack model and classifier only run when status == "ok"."""
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
    res["height_mm"] = height_mm                                      # shown as the assumed H in draw_table
    if weights:
        add_weights(res, weights, nominal_g, tol_g)
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
        if classify:   # tolerance is final here: add_weights ran before this loop
            t = time.perf_counter()
            b["class_grade"], b["grade_patches"] = grade_patches(classify, b["crop"], b["interior"])
            b["grade"] = "Reject" if b.get("tolerance") == "fail" else b["class_grade"]
            res["timings"][f"grade_{b['label']}_s"] = round(time.perf_counter() - t, 3)
    if res["dims"] and not all(Z and (b.get("H_mm") or height_mm) for b in res["blocks"]):
        res["warnings"].append("crack mm uses the bed scale (no camera height or block height): reads too long")
    return res


def grade_patches(classify, crop, interior, patch=330):
    """Most common grade over square grayscale patches of the block interior, half-overlapping, a tie goes to the
    worse grade -> (grade, {grade: count}).
    The classifier was trained on 330 px grayscale close-ups of concrete surface, not on whole colour blocks with
    edges and background: a whole crop squashed to 224 px came out B/C on clean Grade A blocks, patches came out A.
    Worst patch wins was tried too: 2 of 3 clean Grade A blocks came out B (a few B patches on speckles and bevels).
    ponytail: majority vote and patch size in px (camera dependent), both untested on cracked blocks from the rig;
    a crack covering few patches can be outvoted. Retrain on rig patches, then revisit the vote."""
    ys, xs = np.nonzero(interior)
    g = cv2.cvtColor(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = g.shape[:2]
    s = min(patch, h, w)
    starts = lambda n: sorted({*range(0, n - s + 1, s // 2), n - s})       # last patch flush with the far edge
    counts = {}
    for y in starts(h):
        for x in starts(w):
            c = classify(g[y:y + s, x:x + s])
            counts[c] = counts.get(c, 0) + 1
    return max(counts, key=lambda c: (counts[c], c)), counts          # "A" < "B" < "C": a tie goes to the worse


def add_weights(res, weights, nominal_g=None, tol_g=None):
    """weight_g per block by label. Weight can only set "fail": a weight pass never upgrades the L/W/H status."""
    if not weights["stable"]:
        res.setdefault("warnings", []).append("load cells not stable")
    for b in res["blocks"]:
        w = b["weight_g"] = weights.get(b["label"])
        if w is None or nominal_g is None or tol_g is None:
            continue
        if abs(w - nominal_g) > tol_g:
            b["tolerance"] = "fail"
            b["reasons"] = (b.get("reasons") or []) + [f"weight {w:.0f} g vs {nominal_g:g} +/- {tol_g:g}"]


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
        lines = [f"Grade {b['grade']}"] if "grade" in b else []
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


def draw_table(res):
    """Results window: one row per block, L/W/H (mm), weight (g), PASS/FAIL, then why values are missing (status,
    warnings). H: measured mm, else the uncalibrated side-camera height in px, else --height_mm marked * (assumed)."""
    cols = [("Block", 20), ("L (mm)", 150), ("W (mm)", 270), ("H (mm)", 390), ("Weight (g)", 510), ("Result", 680),
            ("Grade", 830)]
    notes = ([f"{res['status']}: {res['message']}"] if res["status"] != "ok" else []) + list(res.get("warnings", []))
    if any(h_cell(b, res).endswith("*") for b in res["blocks"]):
        notes.append(f"H * = assumed --height_mm {res['height_mm']:g}, not measured")
    notes = [ln for n in notes for ln in textwrap.wrap(n, 95)]
    img = np.full((70 + 50 * len(res["blocks"]) + 26 * len(notes) + (10 if notes else 0), 960, 3), 255, np.uint8)
    f = lambda v: "-" if v is None else f"{v:.1f}"
    put = lambda t, x, y, col=(0, 0, 0): cv2.putText(img, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    for name, x in cols:
        put(name, x, 40)
    cv2.line(img, (10, 55), (950, 55), (0, 0, 0), 1)
    for i, b in enumerate(res["blocks"]):
        y = 95 + 50 * i
        tol = b.get("tolerance")
        result, col = {"pass": ("PASS", (0, 160, 0)), "fail": ("FAIL", (0, 0, 220)), None: ("-", (0, 0, 0))}.get(
            tol, (tol, (0, 140, 255)))
        for t, (_, x) in zip([b["label"], f(b.get("L_mm")), f(b.get("W_mm")), h_cell(b, res), f(b.get("weight_g"))],
                             cols):
            put(t, x, y)
        put(result, cols[-2][1], y, col)
        g = b.get("grade", "-")
        put(g, cols[-1][1], y, {"A": (0, 160, 0), "B": (0, 140, 255), "C": (0, 90, 255), "Reject": (0, 0, 220)}.get(g, (0, 0, 0)))
    y0 = 70 + 50 * len(res["blocks"])
    for i, n in enumerate(notes):
        cv2.putText(img, n, (20, y0 + 26 * i + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 1)
    return img


def h_cell(b, res):
    if b.get("H_mm") is not None:
        return f"{b['H_mm']:.1f}"
    if b.get("side_px") is not None:                                  # side camera without --fit_side
        return f"{b['side_px']:.0f} px"
    return "-" if res.get("height_mm") is None else f"{res['height_mm']:g}*"


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
            "crack_mm", "crack_px", "crack_pieces", "crack_mm_per_px", "weight_g", "class_grade", "grade_patches", "grade")
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
        wt = f"  {b['weight_g']} g" if b["weight_g"] is not None else ""
        print(f"  {b['label']}: L {b['L_mm']}  W {b['W_mm']}  H {b['H_mm']}{side}{wt}  {b['tolerance']}  "
              f"crack {b['crack_mm'] if b['crack_mm'] is not None else str(round(b['crack_px'])) + ' px'}"
              f" ({b['crack_pieces']} pcs)  grade {b['grade'] or '-'}")
    for w in res["warnings"]:
        print("  WARNING:", w)
    print(f"timings {res['timings']} -> {d}")


def run_camera(cam, out, meta, side_cam=None, scale=None, **kw):
    """Preview shows the YOLO boxes. SPACE: full pipeline on the laser-OFF frame (+ side frame). L: rerun with
    laser ON. q quits. side_cam: second camera, side view, preview shows its measured edges and height in px.
    scale: open load-cell serial port (loadcell.open_scale); preview shows the live weights, SPACE reads them."""
    from loadcell import ScaleError, read_weights

    def weigh():
        try:
            return read_weights(scale), None
        except ScaleError as e:
            print("WARNING:", e)
            return None, str(e)

    live, live_t, last, weights, werr = None, 0.0, 0.0, None, None
    cap, frame = open_camera(cam)
    side_cap, side_frame = open_camera(side_cam) if side_cam is not None else (None, None)
    off, side, ok = None, None, True
    det_kw = {k: kw[k] for k in ("model", "conf")}
    fx = 1280 / frame.shape[1]
    while ok:
        pv = cv2.resize(draw_overlay(frame, detect_blocks(frame, **det_kw)), None, fx=fx, fy=fx)
        if scale is not None:
            if time.monotonic() - last > 0.5:                         # poll, not every frame
                live, _ = weigh()
                last = time.monotonic()
                if live is not None:
                    live_t = last
            t = "load cells: no reading" if live is None else (
                f"L {live['Left']:.1f}  M {live['Middle']:.1f}  R {live['Right']:.1f} g  "
                + ("stable" if live["stable"] else "NOT stable"))
            cv2.putText(pv, t, (10, pv.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 200, 0) if live and live["stable"] else (0, 0, 255), 2)
        cv2.imshow("pipeline (SPACE run, L laser frame, q quit)", pv)
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
                weights, werr = weigh() if scale is not None else (None, None)
                age = time.monotonic() - live_t
                if weights is None and live is not None and age < 2.0:  # one bad read at SPACE: the preview's is fine
                    weights, werr = live, f"weight from the live reading {age:.1f} s before capture ({werr})"
            on = frame.copy() if key == ord("l") else None
            res = run(off, laser_on=on, side=side, weights=weights, **kw)
            if werr:
                res["warnings"].append(werr if weights else f"no weight: {werr}")
            if res["status"] == "ok":
                save(res, off, on, out, meta)
            else:
                print(f"not saved: {res['message']}  {res['warnings']}")
            cv2.imshow("result", cv2.resize(draw(res), None, fx=fx, fy=fx))
            cv2.imshow("results", draw_table(res))
        ok, frame = cap.read()
        if side_cap is not None:
            ok, side_frame = side_cap.read()
    cap.release()
    if side_cap is not None:
        side_cap.release()
    cv2.destroyAllWindows()


def selftest():
    import tempfile
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
    assert draw_table(r).shape[1] >= 600                              # no dims, no weights: dashes
    assert draw(r).shape == img.shape

    r = run(img, FakeYOLO(boxes[:2]), dark, {})                      # wrong count: crack model skipped
    assert r["status"] == "wrong_count" and "crack_px" not in r["blocks"][0]

    w = {"Left": 1000.0, "Middle": 1200.0, "Right": 990.0, "stable": False}
    r = run(img, yolo, dark, dict(mm_per_px_bed=0.5), weights=w, nominal_g=1000, tol_g=50)
    assert [b["weight_g"] for b in r["blocks"]] == [1000.0, 1200.0, 990.0]
    assert r["blocks"][1]["tolerance"] == "fail" and any("weight" in s for s in r["blocks"][1]["reasons"])
    assert r["blocks"][0]["tolerance"] == "no_standard" and "load cells not stable" in r["warnings"]   # no L/W/H standard: weight pass is not PASS
    t = draw_table(r)
    assert t.ndim == 3 and t.shape[1] >= 600
    r = run(img, yolo, dark, dict(mm_per_px_bed=0.5), weights=w, nominal_g=1000, tol_g=50, classify=lambda c: "B")
    assert [b["grade"] for b in r["blocks"]] == ["B", "Reject", "B"]  # weight fail overrides the classifier
    assert r["blocks"][1]["class_grade"] == "B" and draw_table(r).shape[1] >= 900 and draw(r).shape == img.shape
    crop, inner = np.zeros((700, 400, 3), np.uint8), np.zeros((700, 400), bool)
    inner[50:650, 50:350] = True
    crop[600:640, 300:340] = 255                                     # one bright spot in the bottom-right corner
    seen = []
    worst = lambda p: seen.append(p.shape) or ("C" if p.max() else "A")
    assert grade_patches(worst, crop, inner, patch=200) == ("A", {"A": 9, "C": 1})   # 10 patches, majority A
    flip = iter("AB" * 5)                                             # 10 patches, 5 A and 5 B: tie -> worse
    assert grade_patches(lambda p: next(flip), crop, inner, patch=200) == ("B", {"A": 5, "B": 5})
    assert all(sh == (200, 200, 3) for sh in seen)                   # square, gray as 3 channels, edge patch reached
    r = run(img, FakeYOLO(boxes[:2]), dark, {}, weights=w)            # wrong count: no crash
    assert r["status"] == "wrong_count"

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
    assert all(h_cell(b, r) == f"{b['side_px']:.0f} px" for b in r["blocks"])   # table shows the px, not "-"
    t = draw_table(r)                                                # warnings say why: taller than 3 bare rows
    assert t.shape[0] > 70 + 50 * 3 and any("fit_side" in w for w in r["warnings"]), r["warnings"]
    r2 = run(img, yolo, dark, dict(mm_per_px_bed=0.5), height_mm=55)  # no side camera: assumed height, marked *
    assert [h_cell(b, r2) for b in r2["blocks"]] == ["55*"] * 3 and draw_table(r2).ndim == 3
    from dimensions import load_calib
    assert load_calib(os.path.join(tempfile.mkdtemp(), "nope.json"), warn=True) == {}

    d = tempfile.mkdtemp()                                            # what is saved after a capture
    save(r, img, None, d, {})
    d = os.path.join(d, os.listdir(d)[0])
    with open(f"{d}/summary.json") as f:
        saved = json.load(f)["blocks"]
    assert all(abs(b["side_px"] - 250) < 3 for b in saved), saved
    assert "weight_g" in saved[0] and "grade" in saved[0]
    assert os.path.exists(f"{d}/side.png") and os.path.exists(f"{d}/side_overlay.png")
    print("selftest: ALL PASS")


def report_setup(calib, a, use_side, scale):
    """What this run can measure, before the first capture: a missing calibration otherwise only shows as "-"."""
    dims = (f"ON (--scale_ref {a.scale_ref:g} mm marker)" if a.scale_ref else
            f"ON (mm_per_px_bed {calib['mm_per_px_bed']:.4f})" if "mm_per_px_bed" in calib else
            "OFF: no mm_per_px_bed in the calib file and no --scale_ref (dimensions.py --save_calib)")
    if use_side and "side_coef" in calib:
        h = "side camera, calibrated"
    elif "laser_coef" in calib:
        h = "laser (press L after SPACE)"
    else:
        h = ("side camera NOT calibrated: shows side px only (dimensions.py --fit_side)" if use_side else "not measured")
        if a.height_mm is not None:
            h += f"; table shows the assumed --height_mm {a.height_mm:g}*"
    print(f"setup: L/W {dims}\n       H {h}\n       weight {'on ' + a.scale if scale is not None else 'off (no --scale)'}")


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
    ap.add_argument("--design", default="d1", choices=("d1", "d2", "d3", "none"),
                    help="grade classifier (evaluation/designs.py); d1 won the trade-off; none = no grading")
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
    ap.add_argument("--scale", help="load-cell ESP32 serial port, e.g. COM5 (weight per block)")
    ap.add_argument("--nominal_g", type=float, help="block weight from the standard, g")
    ap.add_argument("--tol_g", type=float, help="weight tolerance, g")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    use_side = a.side_cam is not None or a.side_img
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation"))
    import designs
    grader = designs.weights_path(a.design) if a.design != "none" else None
    for p in (a.weights, a.ckpt) + ((a.side_weights,) if use_side else ()) + ((grader,) if grader else ()):
        if not os.path.exists(p):
            ap.error(f"{p} not found: run  python download_models.py")
    scale = None
    if a.scale:
        from loadcell import ScaleError, open_scale, read_weights
        try:
            scale = open_scale(a.scale)
        except ScaleError as e:
            ap.error(str(e))
    from ultralytics import YOLO
    from infer import load_predict_fn
    fn, device = load_predict_fn(a.ckpt, a.backbone, a.threads)
    triple = lambda s: tuple(map(float, s.split(","))) if s else None
    kw = dict(model=YOLO(a.weights), predict_fn=fn, calib=load_calib(a.calib, warn=True), scale_ref=a.scale_ref,
              height_mm=a.height_mm, nominal=triple(a.nominal), tol=triple(a.tol), thresh=a.thresh,
              conf=a.conf, edge_frac=a.edge_frac, side_model=YOLO(a.side_weights) if use_side else None,
              nominal_g=a.nominal_g, tol_g=a.tol_g, classify=designs.load(a.design, grader) if grader else None)
    meta = dict(weights=os.path.basename(a.weights), ckpt=os.path.basename(a.ckpt), backbone=a.backbone,
                thresh=a.thresh, edge_frac=a.edge_frac, device=device, design=a.design,
                **({"side_weights": os.path.basename(a.side_weights)} if use_side else {}))
    report_setup(kw["calib"], a, use_side, scale)
    if a.cam is not None:
        return run_camera(a.cam, a.out, meta, side_cam=a.side_cam, scale=scale, **kw)
    if not a.img:
        ap.error("--img or --cam is required")
    off = cv2.imread(a.img)
    assert off is not None, f"could not read {a.img}"
    on = cv2.imread(a.laser_on) if a.laser_on else None
    side = cv2.imread(a.side_img) if a.side_img else None
    assert a.side_img is None or side is not None, f"could not read {a.side_img}"
    weights = None
    if scale is not None:
        try:
            weights = read_weights(scale)
        except ScaleError as e:
            print("WARNING:", e)
    res = run(off, laser_on=on, side=side, weights=weights, **kw)
    if res["status"] != "ok":
        print(f"{res['status']}: {res['message']}  {res['warnings']}")
        raise SystemExit(1)
    save(res, off, on, a.out, meta)


if __name__ == "__main__":
    main()
