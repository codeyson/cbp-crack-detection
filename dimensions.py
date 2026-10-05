"""
dimensions.py - length / width / height of each block in mm (objective 1.2).

L, W : top camera. Block outline inside each YOLO box, corrected for the side wall a wide lens sees on
       blocks away from the image centre, then pixels -> mm at the block's TOP-FACE plane.
H    : line laser at an angle. On a block top the laser line shifts sideways in proportion to the height.
       Shift (px) is measured against the line on the bed between blocks; shift -> mm comes from a fit on
       objects of known height (--fit_laser).

Scale (mm/px), pick one:
  --scale_ref 50      ArUco marker (DICT_4X4_50, id any) lying on the bed, black square side measured as 50 mm.
                      Works on phone photos. --save_calib stores the scale so later runs need no marker.
  --calib calib.json  stored values: K, dist (from --calibrate_camera), mm_per_px_bed, cam_height_mm, laser_coef

Examples:
  python dimensions.py --make_marker marker.png                       (print it, measure the black square)
  python dimensions.py --img rig.jpg --weights models/block_yolov8n_best.pt --scale_ref 50 --height_mm 60
  python dimensions.py --img off.jpg --laser_on on.jpg --weights ... --calib calib.json
  python dimensions.py --cam 0 --weights ... --calib calib.json       (SPACE measure, L add laser frame, q quit)
  python dimensions.py --calibrate_camera checker_photos/ --board 9x6 --square_mm 25 --calib calib.json
  python dimensions.py --fit_laser laser_samples.csv --calib calib.json   (CSV rows: shift_px,height_mm)
  python dimensions.py --selftest

Height used for the mm scale: laser height if measured, else --height_mm, else none (warned, L/W too large).
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np

from block_detect import detect_blocks, open_camera

ARUCO = cv2.aruco.DICT_4X4_50


# ----------------------------------------------------------------------------------------------- scale
def load_calib(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_calib(path, **upd):
    c = load_calib(path)
    c.update(upd)
    with open(path, "w") as f:
        json.dump(c, f, indent=1)
    print(f"saved {sorted(upd)} -> {path}")


def undistort(img, calib):
    return cv2.undistort(img, np.array(calib["K"]), np.array(calib["dist"])) if "K" in calib else img


def marker_scale(img, side_mm, calib):
    """mm/px on the bed plane from an ArUco marker; also camera height if K is known. None if no marker."""
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO), cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if ids is None:
        return None
    c = corners[0].reshape(4, 2)
    side_px = np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)])
    out = dict(mm_per_px_bed=side_mm / side_px)
    if "K" in calib:                                  # image already undistorted -> dist = 0
        obj = np.array([[0, 0, 0], [side_mm, 0, 0], [side_mm, side_mm, 0], [0, side_mm, 0]], np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj, c.astype(np.float32), np.array(calib["K"]), None)
        if ok:   # camera height above the bed = distance of the camera centre from the marker plane
            R = cv2.Rodrigues(rvec)[0]
            out["cam_height_mm"] = float(abs((-R.T @ tvec)[2, 0]))
    return out


def calibrate_camera(folder, board=(9, 6), square_mm=25.0):
    """Checkerboard photos -> K, dist. board = inner corners (cols, rows)."""
    obj = np.zeros((board[0] * board[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2) * square_mm
    objs, imgs, size = [], [], None
    for p in sorted(glob.glob(os.path.join(folder, "*.*"))):
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        size = g.shape[::-1]
        ok, c = cv2.findChessboardCorners(g, board)
        print(f"{'ok  ' if ok else 'MISS'} {os.path.basename(p)}")
        if ok:
            objs.append(obj)
            imgs.append(cv2.cornerSubPix(g, c, (11, 11), (-1, -1),
                                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)))
    assert len(imgs) >= 5, f"only {len(imgs)} usable checkerboard photos, need 5+ (15 recommended)"
    rms, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, size, None, None)
    print(f"reprojection error {rms:.3f} px (aim < 0.5)")
    return dict(K=K.tolist(), dist=dist.ravel().tolist())


# ------------------------------------------------------------------------------------------------- L, W
def outline_rect(img, box, pad_frac=0.06):
    """Block outline (silhouette) in a padded YOLO box -> (cv2.minAreaRect in full-image px, fill ratio), or None.
    GrabCut seeded with the YOLO box (outside it = background); the piece at the box centre is the block.
    A single colour threshold split real blocks whose colour is close to the background, GrabCut did not."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    p = max(6, int(pad_frac * max(x2 - x1, y2 - y1)))
    X1, Y1, X2, Y2 = max(0, x1 - p), max(0, y1 - p), min(W, x2 + p), min(H, y2 + p)
    roi = img[Y1:Y2, X1:X2]
    m = np.zeros(roi.shape[:2], np.uint8)
    r = (max(1, x1 - X1 - 2), max(1, y1 - Y1 - 2))     # box grown 2 px (YOLO boxes can sit inside the edge)
    r = r + (min(x2 - X1 + 2, X2 - X1 - 1) - r[0], min(y2 - Y1 + 2, Y2 - Y1 - 1) - r[1])
    cv2.grabCut(roi, m, r, None, None, 5, cv2.GC_INIT_WITH_RECT)
    fg = np.isin(m, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n < 2:
        return None
    k = lab[(y1 + y2) // 2 - Y1, (x1 + x2) // 2 - X1] or 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
    cnts, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # ponytail: minAreaRect on the contour is ~1 px accurate; per-side sub-pixel line fits if 0.5 mm is missed
    cnt = max(cnts, key=cv2.contourArea)
    (cx, cy), (w, h), a = cv2.minAreaRect(cnt)
    fill = cv2.contourArea(cnt) / max(w * h, 1)        # 1.0 = clean rectangle; low = merged shadow, chip, bad box
    return ((cx + X1, cy + Y1), (w, h), a), fill


def top_face_mm(rect, centre, mm_per_px_bed, cam_height_mm=None, height_mm=None):
    """Silhouette rect -> (L, W) of the TOP face in mm.
    Pinhole: the top face is the bed footprint scaled by s = Z/(Z-h) about the image centre. A side whose
    outward normal points toward the centre shows the bottom edge (side wall visible): move it out by s.
    ponytail: ignores chamfers and lens distortion left after undistort."""
    (cx, cy), (w, h), a = rect
    Z, hh = cam_height_mm, height_mm or 0.0
    s = Z / (Z - hh) if Z else 1.0
    t = np.radians(a)
    off = np.array([cx, cy]) - np.asarray(centre, float)
    sizes = []
    for n, half in ((np.array([np.cos(t), np.sin(t)]), w / 2), (np.array([-np.sin(t), np.cos(t)]), h / 2)):
        d = [off @ n + half, -(off @ n) + half]       # signed distance of each side from centre along its normal
        sizes.append(sum(x if x >= 0 else s * x for x in d))
    scale = mm_per_px_bed / s                          # mm/px at the top-face plane = bed scale * (Z-h)/Z
    L, W_ = sorted((sizes[0] * scale, sizes[1] * scale), reverse=True)
    return L, W_


# ---------------------------------------------------------------------------------------------------- H
def laser_line(on, off, min_peak=25, win=3):
    """Row of the laser line in every column (sub-pixel), NaN where too weak. The line must run roughly
    left-right across the image. Uses the brightest channel of on - off, so red or green lasers both work."""
    diff = np.clip(on.astype(np.int16) - off.astype(np.int16), 0, None).max(axis=2).astype(np.float32)
    diff = cv2.GaussianBlur(diff, (1, 5), 0)
    peak = diff.argmax(axis=0)
    rows = np.full(diff.shape[1], np.nan)
    for x in np.nonzero(diff[peak, np.arange(diff.shape[1])] >= min_peak)[0]:
        r0, r1 = max(0, peak[x] - win), min(diff.shape[0], peak[x] + win + 1)
        wts = diff[r0:r1, x]
        rows[x] = (np.arange(r0, r1) * wts).sum() / wts.sum()    # centroid around the peak
    return rows


def laser_shifts(rows, boxes, margin_frac=0.15):
    """Median shift (px) of the laser line on each block vs a straight line fitted to the bed points
    (columns outside every box). Signed: the sign is fixed by the laser side and the calibration absorbs it."""
    xs = np.arange(len(rows))
    bed = ~np.isnan(rows)
    for x1, _, x2, _ in boxes:
        bed &= ~((xs >= x1) & (xs < x2))
    if bed.sum() < 10:
        return [None] * len(boxes)
    fit = np.polyfit(xs[bed], rows[bed], 1)
    out = []
    for x1, _, x2, _ in boxes:
        m = int(margin_frac * (x2 - x1))
        cols = xs[x1 + m:x2 - m]
        sh = np.polyval(fit, cols) - rows[x1 + m:x2 - m]
        sh = sh[~np.isnan(sh)]
        out.append(float(np.median(sh)) if len(sh) >= 5 else None)   # median: cracks/chips on the line
    return out


def fit_laser(csv_path):
    """CSV rows shift_px,height_mm (a header line is fine) -> linear coef and residuals."""
    d = np.genfromtxt(csv_path, delimiter=",", skip_header=0, invalid_raise=False)
    d = d[~np.isnan(d).any(axis=1)]
    coef = np.polyfit(d[:, 0], d[:, 1], 1)
    res = d[:, 1] - np.polyval(coef, d[:, 0])
    print(f"height_mm = {coef[0]:.5f} * shift_px + {coef[1]:.3f}   max |residual| {abs(res).max():.3f} mm "
          f"(aim <= 0.3; if worse try separate fits per Left/Middle/Right)")
    return coef.tolist()


# ------------------------------------------------------------------------------------------- pipeline
def check_tolerance(dims, nominal=None, tol=None):
    """dims/nominal/tol: (L, W, H) in mm, H may be None. Returns (status, reasons). status: pass|fail|no_standard."""
    if not nominal or not tol:
        return "no_standard", []
    reasons = [f"{n} {v:.1f} mm vs {nom} +/- {t}" for n, v, nom, t in zip("LWH", dims, nominal, tol)
               if v is not None and abs(v - nom) > t]
    return ("fail" if reasons else "pass"), reasons


def measure(img, model, calib, scale_ref=None, laser_on=None, height_mm=None, nominal=None, tol=None,
            min_fill=0.85, **det_kw):
    """img = laser-OFF frame (BGR). Returns {"status", "message", "warnings", "blocks"}; status from detect_blocks."""
    img = undistort(img, calib)
    sc = dict(calib)
    if scale_ref:
        m = marker_scale(img, scale_ref, calib)
        if m is None:
            return dict(status="no_marker", message="ArUco marker not found.", warnings=[], blocks=[])
        sc.update(m)
    if "mm_per_px_bed" not in sc:
        raise SystemExit("no scale: use --scale_ref or a --calib with mm_per_px_bed (see --save_calib)")
    res = detect_blocks(img, model, **det_kw)
    warn = []
    shifts = [None] * len(res["blocks"])
    if laser_on is not None:
        shifts = laser_shifts(laser_line(undistort(laser_on, calib), img), [b["box"] for b in res["blocks"]])
        if "laser_coef" not in sc:
            warn.append("laser not calibrated (--fit_laser): height not measured, shift_px only")
    if not sc.get("cam_height_mm"):
        warn.append("camera height unknown: no side-wall/height correction, L/W read too large")
    cxy = (np.array(sc["K"])[:2, 2] if "K" in sc else np.array(img.shape[1::-1]) / 2)
    for b, sh in zip(res["blocks"], shifts):
        H = float(np.polyval(sc["laser_coef"], sh)) if sh is not None and "laser_coef" in sc else None
        b.update(shift_px=sh, H_mm=H, H_source="laser" if H is not None else "not measured")
        out = outline_rect(img, b["box"])
        if out is None:
            b.update(L_mm=None, W_mm=None, tolerance="no_outline", reasons=["outline not found"])
            continue
        rect, b["fill"] = out
        b["rect"] = rect
        b["L_mm"], b["W_mm"] = top_face_mm(rect, cxy, sc["mm_per_px_bed"], sc.get("cam_height_mm"),
                                           H if H is not None else height_mm)
        b["tolerance"], b["reasons"] = check_tolerance((b["L_mm"], b["W_mm"], H), nominal, tol)
        if b["fill"] < min_fill:                       # do not trust L/W from a non-rectangular outline
            b["tolerance"], b["reasons"] = "bad_outline", [f"outline fills {b['fill']:.2f} of its rectangle"]
    if height_mm is None and not any(b.get("H_mm") for b in res["blocks"]) and sc.get("cam_height_mm"):
        warn.append("no block height (laser or --height_mm): L/W read too large")
    res.update(warnings=warn, image=img)
    return res


def draw(img, res):
    vis = img.copy()
    s = max(0.6, vis.shape[1] / 1500.0)
    th = max(2, round(s * 2))
    for b in res["blocks"]:
        col = {"pass": (0, 200, 0), "fail": (0, 0, 255)}.get(b.get("tolerance"), (0, 200, 255))
        cv2.rectangle(vis, tuple(b["box"][:2]), tuple(b["box"][2:]), (255, 160, 0), 1)   # YOLO box (thin blue)
        if "rect" in b:
            cv2.drawContours(vis, [cv2.boxPoints(b["rect"]).astype(int)], -1, col, th)
        x1, y1 = b["box"][:2]
        f = lambda v: "-" if v is None else f"{v:.1f}"
        for i, t in enumerate([b["label"], f"L {f(b.get('L_mm'))}  W {f(b.get('W_mm'))}", f"H {f(b.get('H_mm'))} mm",
                               b.get("tolerance", "")]):
            cv2.putText(vis, t, (x1 + 10, y1 + int((40 + 35 * i) * s)), cv2.FONT_HERSHEY_SIMPLEX, s, col, th)
    cv2.putText(vis, res["status"] + "  " + "; ".join(res.get("warnings", [])), (10, int(35 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, s * 0.7, (0, 200, 255), th)
    return vis


def save(res, out, stem):
    os.makedirs(out, exist_ok=True)
    cv2.imwrite(f"{out}/{stem}_dims.png", draw(res["image"], res))
    keep = ("label", "box", "L_mm", "W_mm", "H_mm", "H_source", "shift_px", "fill", "tolerance", "reasons")
    blocks = [{k: (list(map(int, b[k])) if k == "box" else b.get(k)) for k in keep} for b in res["blocks"]]
    with open(f"{out}/{stem}_dims.json", "w") as f:
        json.dump(dict(status=res["status"], message=res["message"], warnings=res["warnings"], blocks=blocks), f, indent=1)
    for b in blocks:
        print(f"  {b['label']}: L {b['L_mm']}  W {b['W_mm']}  H {b['H_mm']} ({b['H_source']}, shift {b['shift_px']})"
              f"  {b['tolerance']} {b['reasons'] or ''}")
    for w in res["warnings"]:
        print("  WARNING:", w)
    print(f"{res['status']}: {res['message'] or 'ok'} -> {out}/{stem}_dims.*")


def run_camera(cam, out, **kw):
    """SPACE: measure with laser OFF. L: (after SPACE) turn the laser on, press L to add height. q quits."""
    cap, frame = open_camera(cam)
    off, ok = None, True
    while ok:
        cv2.imshow("dimensions (SPACE measure, L laser frame, q quit)", cv2.resize(frame, None, fx=1280 / frame.shape[1], fy=1280 / frame.shape[1]))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in (ord(" "), ord("l")):
            if key == ord(" "):
                off = frame.copy()
            elif off is None:
                print("press SPACE (laser off) first")
                ok, frame = cap.read()
                continue
            stem = time.strftime("%Y%m%d_%H%M%S")
            res = measure(off, laser_on=frame if key == ord("l") else None, **kw)
            os.makedirs(out, exist_ok=True)
            cv2.imwrite(f"{out}/{stem}_off.png", off)
            if key == ord("l"):
                cv2.imwrite(f"{out}/{stem}_laser.png", frame)
            save(res, out, stem)
            cv2.imshow("result", cv2.resize(draw(res["image"], res), None, fx=1280 / frame.shape[1], fy=1280 / frame.shape[1]))
        ok, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------------------------- tests
def selftest():
    from types import SimpleNamespace as NS

    class FakeYOLO:
        def __init__(self, boxes): self.boxes = np.array(boxes, np.float32)
        def predict(self, img, conf, verbose):
            return [NS(boxes=NS(xyxy=NS(cpu=lambda: self.boxes)))]

    # 1. Pinhole scene: camera 400 mm above the bed, 0.25 mm/px on the bed (f = 1600 px), blocks 200x100x60 mm.
    #    Each block is drawn as the hull of its bottom and top faces, so off-centre blocks show a side wall.
    Z, mpp, f, hgt = 400.0, 0.25, 1600.0, 60.0
    W_, H_ = 3400, 1200
    c = np.array([W_ / 2, H_ / 2])
    rng = np.random.default_rng(0)
    img = np.clip(rng.normal(60, 4, (H_, W_, 3)), 0, 255).astype(np.uint8)
    boxes = []
    for X in (-220.0, 0.0, 220.0):                                  # block centres (mm from the optical axis)
        corners = np.array([[X - 100, -50], [X + 100, -50], [X + 100, 50], [X - 100, 50]])
        pts = np.vstack([corners * f / Z, corners * f / (Z - hgt)]) + c
        hull = cv2.convexHull(pts.astype(np.float32)).astype(np.int32)
        cv2.fillPoly(img, [hull], (170, 170, 170))
        x, y, w, h = cv2.boundingRect(hull)
        boxes.append([x, y, x + w, y + h])
    img = np.clip(img.astype(np.int16) + rng.normal(0, 4, img.shape).astype(np.int16), 0, 255).astype(np.uint8)
    calib = dict(mm_per_px_bed=mpp, cam_height_mm=Z)
    r = measure(img, FakeYOLO(boxes), calib, height_mm=hgt)
    assert r["status"] == "ok", r["status"]
    for b in r["blocks"]:
        assert abs(b["L_mm"] - 200) < 0.6 and abs(b["W_mm"] - 100) < 0.6, (b["label"], b["L_mm"], b["W_mm"])
    assert all(b["fill"] > 0.95 for b in r["blocks"]), [b["fill"] for b in r["blocks"]]
    naive = top_face_mm(r["blocks"][0]["rect"], c, mpp)              # no correction: side wall counted
    assert naive[0] > 210, naive

    # 2. Laser: bed line slightly tilted; on each block the line moves up by k px per mm of height.
    k = 3.7
    off = np.full((H_, W_, 3), 60, np.uint8)
    on = off.copy()
    true_h = [55.0, 60.0, 80.0]
    rows = 600 + 0.02 * np.arange(W_)
    for (x1, _, x2, _), hh in zip(boxes, true_h):
        rows[x1:x2] -= k * hh
    yy = np.arange(H_)[:, None]
    on[..., 1] = np.clip(60 + 150 * np.exp(-((yy - rows[None, :]) ** 2) / 4), 0, 255).astype(np.uint8)
    sh = laser_shifts(laser_line(on, off), boxes)
    for s_, hh in zip(sh, true_h):
        assert abs(s_ / k - hh) < 0.3, (s_ / k, hh)

    # 3. Laser fit, full measure with the laser frame, tolerance check
    import tempfile
    p = os.path.join(tempfile.mkdtemp(), "s.csv")
    np.savetxt(p, [[k * h, h] for h in (0, 20, 40, 60, 80)], delimiter=",")
    coef = fit_laser(p)
    assert abs(np.polyval(coef, k * 50) - 50) < 1e-6
    on_scene = np.clip(img.astype(np.int16) + (on.astype(np.int16) - off), 0, 255).astype(np.uint8)
    r = measure(img, FakeYOLO(boxes), dict(calib, laser_coef=coef), laser_on=on_scene,
                nominal=(200, 100, 60), tol=(1.6, 1.6, 3.2))
    assert [round(b["H_mm"]) for b in r["blocks"]] == [55, 60, 80], [b["H_mm"] for b in r["blocks"]]
    assert [b["tolerance"] for b in r["blocks"]] == ["fail", "pass", "fail"], [b["reasons"] for b in r["blocks"]]
    assert check_tolerance((200, 100, None), (200, 100, 60), (1, 1, 1)) == ("pass", [])
    assert check_tolerance((200, 100, 60))[0] == "no_standard"
    print("selftest: ALL PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--weights")
    ap.add_argument("--img", help="laser-OFF photo")
    ap.add_argument("--laser_on", help="same scene, laser ON (for height)")
    ap.add_argument("--cam", type=int)
    ap.add_argument("--calib", default="calib.json")
    ap.add_argument("--save_calib", action="store_true", help="store the --scale_ref result in --calib")
    ap.add_argument("--scale_ref", type=float, help="ArUco marker black-square side in mm (measured)")
    ap.add_argument("--height_mm", type=float, help="assumed block height when there is no laser frame")
    ap.add_argument("--nominal", help="L,W,H mm, e.g. 200,100,60 (from the standard)")
    ap.add_argument("--tol", help="L,W,H tolerance mm, e.g. 1.6,1.6,3.2 (from the standard)")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default="dims_out")
    ap.add_argument("--calibrate_camera", metavar="DIR")
    ap.add_argument("--board", default="9x6", help="checkerboard INNER corners, cols x rows")
    ap.add_argument("--square_mm", type=float, default=25.0)
    ap.add_argument("--fit_laser", metavar="CSV")
    ap.add_argument("--make_marker", metavar="PNG")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.make_marker:
        img = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(ARUCO), 0, 600)
        cv2.imwrite(a.make_marker, cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255))
        return print(f"wrote {a.make_marker}: print it flat (no 'fit to page' distortion matters, you measure it), "
                     f"then measure the black square side with calipers -> --scale_ref")
    if a.calibrate_camera:
        return save_calib(a.calib, **calibrate_camera(a.calibrate_camera, tuple(map(int, a.board.split("x"))), a.square_mm))
    if a.fit_laser:
        return save_calib(a.calib, laser_coef=fit_laser(a.fit_laser))
    if not a.weights:
        ap.error("--weights is required")
    from ultralytics import YOLO
    triple = lambda s: tuple(map(float, s.split(","))) if s else None
    calib = load_calib(a.calib)
    kw = dict(model=YOLO(a.weights), calib=calib, scale_ref=a.scale_ref, height_mm=a.height_mm,
              nominal=triple(a.nominal), tol=triple(a.tol), n_expected=a.n, conf=a.conf)
    if a.cam is not None:
        return run_camera(a.cam, a.out, **kw)
    if not a.img:
        ap.error("--img or --cam is required")
    img = cv2.imread(a.img)
    assert img is not None, f"could not read {a.img}"
    on = cv2.imread(a.laser_on) if a.laser_on else None
    res = measure(img, laser_on=on, **kw)
    if a.save_calib and a.scale_ref:
        m = marker_scale(undistort(img, calib), a.scale_ref, calib)
        if m:
            save_calib(a.calib, **m)
    save(res, a.out, os.path.splitext(os.path.basename(a.img))[0])
    raise SystemExit(0 if res["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
