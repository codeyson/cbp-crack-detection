"""
pipeline_ui.py - pipeline.py in one window: both cameras live on top, the results table below.
Same measuring code as pipeline.py (pipeline.run / pipeline.save), same flags, same pipeline_out/ records.

Live:     python pipeline_ui.py --cam 0 --side_cam 1 --calib calib.json --scale COM7 --height_mm 55
Photos:   python pipeline_ui.py --img off.png --side_img side.png --calib calib.json --height_mm 55

SPACE / Capture: run everything on the current frames (top + side + load cells), save if 3 blocks are found.
L / Laser: rerun the last capture with the current top frame as the laser-ON frame.
R / Live: back to the live view after a capture.   Q / Esc: quit.
"""
import argparse
import contextlib
import glob
import io
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import pipeline
from block_detect import detect_blocks, draw_overlay, open_camera
from dimensions import draw_side, load_calib, side_heights

BG, PANEL, FG, MUTED = "#15171c", "#1f232b", "#e8eaed", "#9aa0a6"
GREEN, RED, AMBER, BLUE = "#34c759", "#ff5f57", "#ffb340", "#5ac8fa"


# ------------------------------------------------------------------------------------------------ inputs
class Camera:
    """Reads frames in its own thread; latest() is always the newest frame (no webcam buffer lag)."""
    def __init__(self, index):
        self.cap, self.frame = open_camera(index)
        self.lock, self.alive = threading.Lock(), True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.alive:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = f
            else:
                time.sleep(0.01)

    def latest(self):
        with self.lock:
            return self.frame.copy()

    def close(self):
        self.alive = False
        time.sleep(0.05)
        self.cap.release()


class Still:
    """A photo standing in for a camera (--img / --side_img), for trying the UI without the rig."""
    def __init__(self, path):
        self.frame = cv2.imread(path)
        assert self.frame is not None, f"could not read {path}"

    def latest(self):
        return self.frame.copy()

    def close(self):
        pass


class Scale:
    """Load cells polled every 0.5 s in a thread (shown live). capture() takes a fresh reading and falls back to
    the live one when it is under 2 s old, like pipeline.run_camera."""
    def __init__(self, ser):
        from loadcell import ScaleError, read_weights
        self.ser, self.read, self.Err = ser, read_weights, ScaleError
        self.lock, self.live, self.live_t, self.err, self.alive = threading.Lock(), None, 0.0, None, True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.alive:
            w, e = self._read()
            if w is not None:
                self.live, self.live_t, self.err = w, time.monotonic(), None
            else:
                self.err = e
            time.sleep(0.5)

    def _read(self):
        with self.lock:
            try:
                return self.read(self.ser), None
            except self.Err as e:
                return None, str(e)

    def capture(self):
        w, err = self._read()
        age = time.monotonic() - self.live_t
        if w is None and self.live is not None and age < 2.0:
            return self.live, f"weight from the live reading {age:.1f} s before capture ({err})"
        return w, err

    def close(self):
        self.alive = False


# ------------------------------------------------------------------------------------------------ helpers
def scaled(res, f, px=False):
    """Copy of a detect_blocks/side_heights result with boxes and side edges scaled by f. px: scale the printed
    height in px too (preview frame -> full resolution); off when only shrinking it for display."""
    blocks = []
    for b in res["blocks"]:
        b = dict(b, box=tuple(int(round(v * f)) for v in b["box"]))
        for k in ("top", "bottom") + (("h_px",) if px else ()):
            if k in b:
                b[k] = b[k] * f
        blocks.append(b)
    return dict(res, blocks=blocks)


def fit(img, w, h):
    s = min(w / img.shape[1], h / img.shape[0])
    return cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)


def to_tk(img_bgr):
    return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))


def placeholder(w, h, text):
    img = np.full((h, w, 3), (43, 35, 31), np.uint8)
    cv2.putText(img, text, (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (166, 160, 154), 2)
    return img


def fmt(v, unit=""):
    return "-" if v is None else f"{v:.1f}{unit}"


# ---------------------------------------------------------------------------------------------------- app
class App:
    PREVIEW_W = 960                                   # preview detection runs on frames this wide (speed)

    def __init__(self, root, top, side, scale, kw, meta, out, setup_text):
        self.root, self.top, self.side, self.scale = root, top, side, scale
        self.kw, self.meta, self.out = kw, meta, out
        self.model_lock = threading.Lock()            # YOLO models are shared by the preview and the capture
        self.det, self.side_det = None, None          # latest preview detections (on PREVIEW_W frames)
        self.busy, self.frozen, self.alive = False, False, True
        self.last = None                              # (off, side, weights, werr) of the last capture, for L
        self.saved_dir = None
        self.results = queue.Queue()
        self._build(setup_text)
        threading.Thread(target=self._preview_loop, daemon=True).start()
        self._tick()
        self._poll_results()

    # ---- layout
    def _build(self, setup_text):
        r = self.root
        r.title("CBP block inspection")
        r.configure(bg=BG)
        r.geometry("1500x950")
        r.minsize(1000, 700)
        st = ttk.Style(r)
        st.theme_use("clam")
        st.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=FG, rowheight=34,
                     font=("Segoe UI", 12), borderwidth=0)
        st.configure("Treeview.Heading", background="#2a2f39", foreground=FG, font=("Segoe UI", 11, "bold"),
                     relief="flat")
        st.map("Treeview", background=[("selected", "#2f3644")])

        cams = tk.Frame(r, bg=BG)
        cams.pack(side="top", fill="both", expand=True, padx=12, pady=(12, 6))
        cams.columnconfigure((0, 1), weight=1, uniform="cam")
        cams.rowconfigure(1, weight=1)
        self.cam_title, self.cam_view = [], []
        for i, name in enumerate(("TOP CAMERA  -  L / W, crack, grade", "SIDE CAMERA  -  height")):
            t = tk.Label(cams, text=name, bg=BG, fg=MUTED, font=("Segoe UI", 10, "bold"), anchor="w")
            t.grid(row=0, column=i, sticky="ew", padx=6)
            v = tk.Label(cams, bg=PANEL, bd=0)
            v.grid(row=1, column=i, sticky="nsew", padx=6, pady=(2, 0))
            self.cam_title.append(t)
            self.cam_view.append(v)

        bottom = tk.Frame(r, bg=BG)
        bottom.pack(side="top", fill="x", padx=18, pady=(6, 6))
        left = tk.Frame(bottom, bg=BG)
        left.pack(side="left", fill="both", expand=True)
        cols = [("block", "Block", 110), ("L", "L (mm)", 100), ("W", "W (mm)", 100), ("H", "H (mm)", 110),
                ("wt", "Weight (g)", 120), ("crack", "Crack", 160), ("result", "Result", 140), ("grade", "Grade", 100)]
        self.table = ttk.Treeview(left, columns=[c for c, _, _ in cols], show="headings", height=3, selectmode="none")
        for c, h, w in cols:
            self.table.heading(c, text=h)
            self.table.column(c, width=w, anchor="center")
        for tag, col in (("pass", GREEN), ("fail", RED), ("other", AMBER), ("none", FG)):
            self.table.tag_configure(tag, foreground=col)
        self.table.pack(fill="x")
        self.notes = tk.Text(left, height=5, bg=BG, fg=AMBER, bd=0, font=("Segoe UI", 10), wrap="word",
                             highlightthickness=0)
        self.notes.pack(fill="both", expand=True, pady=(8, 0))
        self._set_notes(["Press SPACE (or Capture) to measure the 3 blocks."])

        right = tk.Frame(bottom, bg=BG)
        right.pack(side="right", fill="y", padx=(18, 0))
        btn = dict(font=("Segoe UI", 12, "bold"), bd=0, padx=16, pady=10, takefocus=0, cursor="hand2",
                   activeforeground=FG)
        self.b_cap = tk.Button(right, text="Capture  [SPACE]", bg="#2563eb", fg="white", activebackground="#1d4ed8",
                               command=self.capture, **btn)
        self.b_cap.pack(fill="x")
        row = tk.Frame(right, bg=BG)
        row.pack(fill="x", pady=(8, 0))
        self.b_laser = tk.Button(row, text="Laser [L]", bg="#374151", fg=FG, activebackground="#4b5563",
                                 command=self.laser, **btn)
        self.b_laser.pack(side="left", expand=True, fill="x")
        self.b_live = tk.Button(row, text="Live [R]", bg="#374151", fg=FG, activebackground="#4b5563",
                                command=self.resume, **btn)
        self.b_live.pack(side="left", expand=True, fill="x", padx=(8, 0))
        self.b_open = tk.Button(right, text="Open last capture folder", bg="#374151", fg=FG,
                                activebackground="#4b5563", command=self.open_folder, **btn)
        self.b_open.pack(fill="x", pady=(8, 0))
        tk.Label(right, text=setup_text, bg=BG, fg=MUTED, font=("Consolas", 9), justify="left",
                 anchor="w").pack(fill="x", pady=(10, 0))

        bar = tk.Frame(r, bg=PANEL)
        bar.pack(side="bottom", fill="x")
        self.weight_lbl = tk.Label(bar, bg=PANEL, fg=MUTED, font=("Segoe UI", 11, "bold"), anchor="w", padx=12, pady=6)
        self.weight_lbl.pack(side="left")
        self.status_lbl = tk.Label(bar, text="Live", bg=PANEL, fg=MUTED, font=("Segoe UI", 11), anchor="e", padx=12)
        self.status_lbl.pack(side="right")

        for keys, fn in ((("<space>",), self.capture), (("l", "L"), self.laser), (("r", "R"), self.resume),
                         (("q", "Q", "<Escape>"), self.quit)):
            for k in keys:
                r.bind(k, lambda e, fn=fn: fn())
        r.protocol("WM_DELETE_WINDOW", self.quit)

    def _set_notes(self, lines):
        self.notes.configure(state="normal")
        self.notes.delete("1.0", "end")
        self.notes.insert("end", "\n".join(lines))
        self.notes.configure(state="disabled")

    def _status(self, text, col=MUTED):
        self.status_lbl.configure(text=text, fg=col)

    # ---- live preview
    def _preview_loop(self):
        """YOLO on downscaled frames in the background; the UI draws the latest boxes on the newest frame."""
        flip = self.kw["calib"].get("side_flip", False)
        while self.alive:
            if self.busy or self.frozen or not self.model_lock.acquire(blocking=False):
                time.sleep(0.05)
                continue
            try:
                f = self.top.latest()
                k = self.PREVIEW_W / f.shape[1]                       # detections stored in full-frame px
                self.det = scaled(detect_blocks(cv2.resize(f, None, fx=k, fy=k), self.kw["model"],
                                                conf=self.kw["conf"]), 1 / k)
                if self.side is not None:
                    s = self.side.latest()
                    k = self.PREVIEW_W / s.shape[1]
                    self.side_det = scaled(side_heights(cv2.resize(s, None, fx=k, fy=k), self.kw["side_model"],
                                                        conf=self.kw["conf"], flip=flip), 1 / k, px=True)
            except Exception as e:                                    # a bad frame must not kill the preview
                print("preview:", e)
            finally:
                self.model_lock.release()
            time.sleep(0.02)

    def _panel_size(self):
        w = max(200, self.cam_view[0].winfo_width())
        h = max(150, self.cam_view[0].winfo_height())
        return w, h

    def _tick(self):
        if not self.alive:
            return
        if not self.frozen:
            w, h = self._panel_size()
            f = self.top.latest()
            small = fit(f, w, h)
            if self.det is not None:
                small = draw_overlay(small, scaled(self.det, small.shape[1] / f.shape[1]))
            self._show(0, small)
            if self.side is not None:
                s = self.side.latest()
                ss = fit(s, w, h)
                if self.side_det is not None:
                    ss = draw_side(ss, scaled(self.side_det, ss.shape[1] / s.shape[1]))
                self._show(1, ss)
            else:
                self._show(1, placeholder(w, h, "no side camera (--side_cam)"))
        if self.scale is not None:
            lv = self.scale.live
            if lv is None:
                self.weight_lbl.configure(text=f"Load cells: no reading  {self.scale.err or ''}", fg=RED)
            else:
                ok = lv["stable"]
                self.weight_lbl.configure(
                    text=f"Load cells   L {lv['Left']:.1f}   M {lv['Middle']:.1f}   R {lv['Right']:.1f} g   "
                         + ("stable" if ok else "NOT stable - wait before capture"), fg=GREEN if ok else AMBER)
        else:
            self.weight_lbl.configure(text="Load cells: off (no --scale)", fg=MUTED)
        self.root.after(33, self._tick)

    def _show(self, i, img):
        im = to_tk(img)
        self.cam_view[i].configure(image=im)
        self.cam_view[i].image = im                                   # keep a reference or Tk drops it

    # ---- capture
    def capture(self):
        if self.busy:
            return
        off = self.top.latest()
        side = self.side.latest() if self.side is not None else None
        weights, werr = self.scale.capture() if self.scale is not None else (None, None)
        self.last = (off, side, weights, werr)
        self._start(off, side, None, weights, werr)

    def laser(self):
        if self.busy:
            return
        if self.last is None:
            self._status("Press SPACE (laser off) first", AMBER)
            return
        off, side, weights, werr = self.last
        self._start(off, side, self.top.latest(), weights, werr)

    def _start(self, off, side, on, weights, werr):
        self.busy, self.frozen = True, True
        self.b_cap.configure(state="disabled", text="Measuring...")
        self._status("Running pipeline: detection, dimensions, height, crack, grade ...", BLUE)
        threading.Thread(target=self._job, args=(off, side, on, weights, werr), daemon=True).start()

    def _job(self, off, side, on, weights, werr):
        try:
            with self.model_lock:
                res = pipeline.run(off, laser_on=on, side=side, weights=weights, **self.kw)
            if werr:
                res["warnings"].append(werr if weights else f"no weight: {werr}")
            saved = None
            if res["status"] == "ok":
                pipeline.save(res, off, on, self.out, self.meta)      # same files + results.csv as pipeline.py
                dirs = [d for d in glob.glob(os.path.join(self.out, "*")) if os.path.isdir(d)]
                saved = max(dirs, key=os.path.getmtime) if dirs else None
            else:
                print(f"not saved: {res['message']}  {res['warnings']}")
            self.results.put((res, saved, None))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.results.put((None, None, e))

    def _poll_results(self):
        try:
            res, saved, err = self.results.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_results)
            return
        self.busy = False
        self.b_cap.configure(state="normal", text="Capture  [SPACE]")
        if err is not None:
            self.frozen = False
            self._status(f"Error: {err}", RED)
        else:
            self._show_result(res, saved)
        self.root.after(100, self._poll_results)

    def _show_result(self, res, saved):
        w, h = self._panel_size()
        self._show(0, fit(pipeline.draw(res), w, h))
        if "side" in res:
            self._show(1, fit(draw_side(res["side"]["image"], res["side"]), w, h))
        for t in self.cam_title:
            t.configure(fg=BLUE)
        self.cam_title[0].configure(text="TOP CAMERA  -  CAPTURED  (R = back to live)")
        self.cam_title[1].configure(text="SIDE CAMERA  -  CAPTURED")

        self.table.delete(*self.table.get_children())
        for b in res["blocks"]:
            tol = b.get("tolerance")
            result = {"pass": "PASS", "fail": "FAIL", None: "-"}.get(tol, tol)
            if "crack_px" in b:
                crack = (f"{b['crack_mm']:.1f} mm" if b.get("crack_mm") is not None else f"{b['crack_px']:.0f} px") \
                    + f" ({b['crack_pieces']})"
            else:
                crack = "-"
            grade = b.get("grade", "-")
            tag = "fail" if tol == "fail" or grade == "Reject" else "pass" if tol == "pass" else \
                "none" if tol is None else "other"
            self.table.insert("", "end", tags=(tag,), values=(
                b["label"], fmt(b.get("L_mm")), fmt(b.get("W_mm")), pipeline.h_cell(b, res), fmt(b.get("weight_g")),
                crack, result, grade))

        notes = ([f"{res['status']}: {res['message']}  (not saved)"] if res["status"] != "ok" else []) \
            + list(res.get("warnings", []))
        if any(pipeline.h_cell(b, res).endswith("*") for b in res["blocks"]):
            notes.append(f"H * = assumed --height_mm {res['height_mm']:g}, not measured")
        self._set_notes(notes or ["No warnings."])
        if saved:
            self.saved_dir = saved
            self._status(f"Saved -> {os.path.join(os.path.basename(self.out), os.path.basename(saved))}"
                         f"   ({sum(res['timings'].values()):.1f} s)", GREEN)
        else:
            self._status(f"Not saved: {res['message']}", RED)

    def resume(self):
        if self.busy:
            return
        self.frozen = False
        self.cam_title[0].configure(text="TOP CAMERA  -  L / W, crack, grade", fg=MUTED)
        self.cam_title[1].configure(text="SIDE CAMERA  -  height", fg=MUTED)
        self._status("Live")

    def open_folder(self):
        d = self.saved_dir or self.out
        if not os.path.isdir(d):
            self._status("Nothing saved yet", AMBER)
        elif os.name == "nt":
            os.startfile(os.path.abspath(d))
        else:
            self._status(os.path.abspath(d))

    def quit(self):
        self.alive = False
        for c in (self.top, self.side, self.scale):
            if c is not None:
                c.close()
        self.root.destroy()


# --------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="pipeline.py with one window: cameras on top, results below")
    ap.add_argument("--cam", type=int, help="top webcam index")
    ap.add_argument("--img", help="top photo instead of a camera (try the UI without the rig)")
    ap.add_argument("--side_cam", type=int, help="side webcam index (height)")
    ap.add_argument("--side_img", help="side photo instead of a camera")
    ap.add_argument("--side_weights", default="models/height_detection.pt")
    ap.add_argument("--weights", default="models/best.pt", help="YOLO block detector")
    ap.add_argument("--ckpt", default="models/unet_efficientnet-b0_best.pth", help="crack U-Net checkpoint")
    ap.add_argument("--backbone", default="efficientnet-b0")
    ap.add_argument("--design", default="d1", choices=("d1", "d2", "d3", "none"))
    ap.add_argument("--calib", default="calib.json")
    ap.add_argument("--scale_ref", type=float, help="ArUco marker black-square side in mm (measured)")
    ap.add_argument("--height_mm", type=float, help="assumed block height when height is not measured")
    ap.add_argument("--nominal", help="L,W,H mm (from the standard)")
    ap.add_argument("--tol", help="L,W,H tolerance mm (from the standard)")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--edge_frac", type=float, default=0.10)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="pipeline_out")
    ap.add_argument("--scale", help="load-cell ESP32 serial port, e.g. COM7")
    ap.add_argument("--nominal_g", type=float)
    ap.add_argument("--tol_g", type=float)
    a = ap.parse_args()
    if a.cam is None and not a.img:
        ap.error("--cam or --img is required")
    use_side = a.side_cam is not None or a.side_img

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation"))
    import designs
    grader = designs.weights_path(a.design) if a.design != "none" else None
    for p in (a.weights, a.ckpt) + ((a.side_weights,) if use_side else ()) + ((grader,) if grader else ()):
        if not os.path.exists(p):
            ap.error(f"{p} not found: run  python download_models.py")
    ser = None
    if a.scale:
        from loadcell import ScaleError, open_scale
        try:
            ser = open_scale(a.scale)
        except ScaleError as e:
            ap.error(str(e))
    from ultralytics import YOLO
    from infer import load_predict_fn
    print("loading models ...")
    fn, device = load_predict_fn(a.ckpt, a.backbone, a.threads)
    triple = lambda s: tuple(map(float, s.split(","))) if s else None
    kw = dict(model=YOLO(a.weights), predict_fn=fn, calib=load_calib(a.calib, warn=True), scale_ref=a.scale_ref,
              height_mm=a.height_mm, nominal=triple(a.nominal), tol=triple(a.tol), thresh=a.thresh,
              conf=a.conf, edge_frac=a.edge_frac, side_model=YOLO(a.side_weights) if use_side else None,
              nominal_g=a.nominal_g, tol_g=a.tol_g, classify=designs.load(a.design, grader) if grader else None)
    meta = dict(weights=os.path.basename(a.weights), ckpt=os.path.basename(a.ckpt), backbone=a.backbone,
                thresh=a.thresh, edge_frac=a.edge_frac, device=device, design=a.design, ui=True,
                **({"side_weights": os.path.basename(a.side_weights)} if use_side else {}))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pipeline.report_setup(kw["calib"], a, use_side, ser)
    print(buf.getvalue(), end="")

    top = Camera(a.cam) if a.cam is not None else Still(a.img)
    side = Camera(a.side_cam) if a.side_cam is not None else Still(a.side_img) if a.side_img else None
    scale = Scale(ser) if ser is not None else None
    root = tk.Tk()
    App(root, top, side, scale, kw, meta, a.out, buf.getvalue().strip())
    root.mainloop()


if __name__ == "__main__":
    main()