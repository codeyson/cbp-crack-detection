# Commands

Run every command from the repo folder with the venv active.

## Setup (once)

Needs Python 3.11 and git. On Windows:

```bash
git clone https://github.com/codeyson/cbp-crack-detection.git
cd cbp-crack-detection
python -m venv venv
venv\Scriptsctivate
pip install -r requirements.txt
python download_models.py
```

On macOS/Linux activate with `source venv/bin/activate`. `download_models.py` puts the weights from the GitHub Release into `models/`:
- `models/best.pt`: block detector, the default everywhere
- `models/STABLE_block_yolov8n_best.pt`: older block detector, use with `--weights models/STABLE_block_yolov8n_best.pt`
- `models/unet_efficientnet-b0_best.pth`: crack model

Every script uses these paths by default, so `--weights` and `--ckpt` are only needed for other files.

Updating later: `git pull`, then `pip install -r requirements.txt` again if `requirements.txt` changed.

## Self-tests (no weights, no camera needed)

```bash
python block_detect.py --selftest
python crack_length.py
python infer.py --selftest
python dimensions.py --selftest
python pipeline.py --selftest
```

Each prints `ALL PASS` (or the passing checks). Run them after any code change.

## 1.1 Block detection: `block_detect.py`

Live webcam. SPACE saves the frame and crops (only when exactly 3 blocks are found), q quits:

```bash
python block_detect.py --cam 0 --conf 0.3 --out captures
```

One photo:

```bash
python block_detect.py --img rig.jpg --out blocks_out
```

Score against a Roboflow YOLOv8 export (pass = all blocks found with IoU >= 0.8):

```bash
python block_detect.py --yolo_dir "CBP Block Detection.v3i.yolov8"
```

Options: `--conf` (YOLO confidence, default 0.5), `--erode_px` (interior shrink, default 6), `--edge_frac` (interior shrink per side as a fraction of the box, default 0.10; keeps side walls and shadows out of the crack measurement).
Output per photo: `*_overlay.png`, `*_<Left|Middle|Right>_crop.png`, `*_summary.json` (+ `*_raw.png` in webcam mode).

## 1.2 Dimensions: `dimensions.py`

### One-time setup

Make an ArUco marker, print it, and measure the black square side with calipers (in mm):

```bash
python dimensions.py --make_marker marker.png
```

Camera calibration: take about 15 photos of a printed checkerboard at different angles, put them in one folder. `--board` is the number of INNER corners (cols x rows), `--square_mm` the measured square size:

```bash
python dimensions.py --calibrate_camera checker_photos/ --board 9x6 --square_mm 25 --calib calib.json
```

Store the rig scale (marker lying on the bed, camera in its final position). Writes `mm_per_px_bed` (and `cam_height_mm` if the camera is calibrated) into `calib.json`, so later runs need no marker:

```bash
python dimensions.py --img marker_on_bed.jpg --scale_ref 50 --save_calib --calib calib.json
```

Laser height calibration: measure objects of known height (calipers), note each block's printed `shift`, and write a CSV with rows `shift_px,height_mm`. Then:

```bash
python dimensions.py --fit_laser laser_samples.csv --calib calib.json
```

Aim for max residual <= 0.3 mm. Redo calibration whenever the camera or laser moves.

Side-camera height (replaces the laser). Mount the second camera level with the blocks, lens at about half the block
height (above the tops it sees a strip of the top face), facing the long side of the row so all 3 blocks stand side
by side, plain backdrop behind. Needs `models/height_detection.pt`. Calibrate with blocks measured by calipers:
photograph them on the rig, print each block's `side_px`, and write a CSV with rows `side_px,height_mm`:

```bash
python dimensions.py --side_img side.jpg --conf 0.3
```

```bash
python dimensions.py --fit_side side_samples.csv --calib calib.json
```

Use at least 6 blocks of different heights (or the same blocks raised on shims of known thickness). If the side
camera looks at the row from the back, so its left is the top camera's right, add `"side_flip": true` to `calib.json`.
Redo the fit whenever the side camera moves.

### Measuring

Photo with the marker in frame, assumed block height (no laser yet):

```bash
python dimensions.py --img photo.jpg --scale_ref 50 --height_mm 60
```

Rig photo using `calib.json`, laser OFF + laser ON frames (measures height):

```bash
python dimensions.py --img off.jpg --laser_on on.jpg --calib calib.json
```

Live webcam: use `pipeline.py --cam` (below).

Pass/fail against the standard (fill in the real values, there are no defaults):

```bash
python dimensions.py --img photo.jpg --calib calib.json --nominal 200,100,60 --tol 1.6,1.6,3.2
```

Options: `--conf`, `--out` (default `dims_out`).
Output: `*_dims.png` (outline, L/W/H, pass/fail) and `*_dims.json`. `bad_outline` = outline not rectangular enough, numbers not trusted.

## 1.3 Crack segmentation: `infer.py`

Run on a block crop (e.g. from `block_detect.py`):

```bash
python infer.py --img captures/<stamp>_Left_crop.png --out results --threads 4
```

Crack length in mm: use `pipeline.py` (below).
Options: `--tile 256`, `--overlap 64`, `--batch 8`, `--thresh 0.5`.

Note: the model is DeepCrack-pretrained only. It is not reliable on pavers until fine-tuned.

## End-to-end pipeline: `pipeline.py` (start here)

Detect + crop, dimensions, crack mask and crack length for all 3 blocks in one go. No grading yet.

Two cameras, top + side (height from the side camera, after `--fit_side`). SPACE grabs both, q quits:

```bash
python pipeline.py --cam 0 --side_cam 1 --calib calib.json
```

Offline with a side photo: `python pipeline.py --img top.png --side_img side.png --calib calib.json`.
If the second camera does not open, plug it into a different USB port (two 2K cameras can overload one port).

Live webcam with the laser instead. SPACE runs everything (laser off), then turn the laser on and press L to rerun with height, q quits:

```bash
python pipeline.py --cam 0 --calib calib.json
```

Offline on a saved frame (same code as live):

```bash
python pipeline.py --img captures/<stamp>_raw.png
```

Scale: `--scale_ref 50` (ArUco in frame) or `mm_per_px_bed` in `--calib`. With neither, dimensions are skipped and crack length is in px.
Also takes `--conf` (default 0.3 here), `--edge_frac` (default 0.10), `--weights`, `--ckpt`, `--laser_on`, `--side_weights`, `--height_mm`, `--nominal`, `--tol`, `--thresh`, `--threads`, `--backbone`.
Output: `pipeline_out/<timestamp>/` with `off.png`, `laser.png`, `side.png`, `side_overlay.png`, `overlay.png`, `<Label>_crop.png`, `<Label>_mask.png`, `summary.json` (values, warnings, per-step timings, model names). Nothing is saved unless exactly 3 blocks are found.

## Training (Google Colab, not on this PC)

- `crack_pretrain_unet.ipynb`: crack U-Net. Set `BACKBONE`, re-run the model cell before every training run.
- `train_block_yolov8.ipynb`: YOLOv8 block detector on a T4.
