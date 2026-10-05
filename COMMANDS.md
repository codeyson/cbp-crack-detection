# Commands

Run every command from the repo folder with the venv active.

## Setup (once)

```bash
python -m venv venv
venv\Scripts\activate
```

Install torch + torchvision (CPU build) first, using the command from https://pytorch.org for your OS. Then:

```bash
pip install -r requirements.txt
```

Weights are not in git. Put them in `models/`:
- `models/block_yolov8n_best.pt` (block detector, from `train_block_yolov8.ipynb`)
- `models/unet_efficientnet-b0_best.pth` (crack model, from `crack_pretrain_unet.ipynb`)

## Self-tests (no weights, no camera needed)

```bash
python block_detect.py --selftest
python crack_length.py
python infer.py --selftest
python dimensions.py --selftest
```

Each prints `ALL PASS` (or the passing checks). Run them after any code change.

## 1.1 Block detection: `block_detect.py`

Live webcam. SPACE saves the frame and crops (only when exactly 3 blocks are found), q quits:

```bash
python block_detect.py --cam 0 --weights models/block_yolov8n_best.pt --conf 0.3 --out captures
```

One photo:

```bash
python block_detect.py --img rig.jpg --weights models/block_yolov8n_best.pt --out blocks_out
```

Score against a Roboflow YOLOv8 export (pass = all blocks found with IoU >= 0.8):

```bash
python block_detect.py --yolo_dir "CBP Block Detection.v3i.yolov8" --weights models/block_yolov8n_best.pt
```

Options: `--conf` (YOLO confidence, default 0.5), `--n` (expected blocks, default 3), `--erode_px` (interior shrink, default 6).
Output per photo: `*_overlay.png`, `*_<Left|Middle|Right>_crop.png`, `*_interior.png`, `*_summary.json` (+ `*_raw.png` in webcam mode).

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
python dimensions.py --img marker_on_bed.jpg --weights models/block_yolov8n_best.pt --scale_ref 50 --save_calib --calib calib.json
```

Laser height calibration: measure objects of known height (calipers), note each block's printed `shift`, and write a CSV with rows `shift_px,height_mm`. Then:

```bash
python dimensions.py --fit_laser laser_samples.csv --calib calib.json
```

Aim for max residual <= 0.3 mm. Redo calibration whenever the camera or laser moves.

### Measuring

Photo with the marker in frame, assumed block height (no laser yet):

```bash
python dimensions.py --img photo.jpg --weights models/block_yolov8n_best.pt --scale_ref 50 --height_mm 60
```

Rig photo using `calib.json`, laser OFF + laser ON frames (measures height):

```bash
python dimensions.py --img off.jpg --laser_on on.jpg --weights models/block_yolov8n_best.pt --calib calib.json
```

Live webcam. SPACE measures (laser off), then turn the laser on and press L to add height, q quits:

```bash
python dimensions.py --cam 0 --weights models/block_yolov8n_best.pt --calib calib.json
```

Pass/fail against the standard (fill in the real values, there are no defaults):

```bash
python dimensions.py --img photo.jpg --weights models/block_yolov8n_best.pt --calib calib.json --nominal 200,100,60 --tol 1.6,1.6,3.2
```

Options: `--conf`, `--n`, `--out` (default `dims_out`).
Output: `*_dims.png` (outline, L/W/H, pass/fail) and `*_dims.json`. `bad_outline` = outline not rectangular enough, numbers not trusted.

## 1.3 Crack segmentation: `infer.py`

Run on a block crop from `block_detect.py`:

```bash
python infer.py --img captures/20261005_155850_Left_crop.png --ckpt models/unet_efficientnet-b0_best.pth --backbone efficientnet-b0 --out results --threads 4
```

Add `--mm_per_px 0.1` to also print crack length in mm (use the real scale from calibration).
Options: `--tile 256`, `--overlap 64`, `--batch 8`, `--thresh 0.5`.

Note: the model is DeepCrack-pretrained only. It is not reliable on pavers until fine-tuned.

## Training (Google Colab, not on this PC)

- `crack_pretrain_unet.ipynb`: crack U-Net. Set `BACKBONE`, re-run the model cell before every training run.
- `train_block_yolov8.ipynb`: YOLOv8 block detector on a T4.
