# CBP Crack Detection

Software that checks concrete paving blocks for quality using a camera, a laser and load cells, and sorts each block into Grade A, B or C.

## Why this project exists

Concrete block pavers (CBPs) are usually checked by hand: someone measures them, weighs them and looks for cracks. That is slow and depends on who is checking. Some tests also destroy the block.

This project is the software side of a thesis:

> *Design of a Deep Learning-Based Quality Classification System for Non-Destructive Concrete Block Pavement Grading with Optimized Formula Mixture Recommendation*

The goal is a rig that grades blocks quickly and the same way every time, without damaging them.

## How it works

Three blocks are placed on the rig at a time. Then:

1. **Take a photo.** A camera above the rig takes one picture of all three blocks.
2. **Find the blocks.** A trained detector (YOLOv8) checks that there are exactly three blocks and labels them Left, Middle and Right. If it does not see three, it asks the user to re-place them.
3. **Cut out each block.** Each block is cropped into its own image. The edges are trimmed so they are not mistaken for cracks.
4. **Measure size and weight.** Length and width come from the photo, using a printed marker or a saved calibration to turn pixels into millimetres. Height comes from a laser line that shifts when it hits a taller object. Weight comes from one load cell under each block.
5. **Find cracks.** A crack model (U-Net) marks every crack pixel on each block. The software then measures the total crack length in millimetres.
6. **Grade the block.**
   - If the size or weight is outside the allowed tolerance, the block is **rejected**.
   - Otherwise, a deep-learning classifier looks at the block crop and decides **Grade A, B or C** (designs compared in `evaluation/`).
   - Chipped, spalled or shattered blocks are **Grade C**.

## Current status

| Part | Status |
|---|---|
| Finding the 3 blocks and cropping them | Working |
| Measuring length, width and height | Working, needs calibration on the real rig |
| Crack detection and crack length | Working, but **not yet reliable on pavers** (see below) |
| Weight from load cells | Not built yet |
| Grading (A/B/C/Reject) | Step ready in `pipeline.py --grader`. Waiting on the A/B/C dataset to train the classifier |
| One-button run of the whole process (`pipeline.py`) | Working (grades once a classifier is trained) |
| Mix formula recommendation | Out of scope for this repo |

## Important things to know

- **The crack model is not trained on pavers yet.** It was trained on a public crack dataset (DeepCrack). It must be fine-tuned on photos from this rig before its results can be trusted.
- **Calibrate whenever the setup changes.** If the camera or laser moves, redo the calibration, or the measurements will be wrong.
- **Use one camera.** Training and real use should use the same camera and lighting. Mixing cameras makes the model less accurate.
- **Agree what counts as Grade A, B and C before sorting the dataset.** Changing it afterward means re-labelling everything.
- **Model files are not in git.** `python download_models.py` fetches them from the GitHub Release into `models/`.
- **Only rectangular blocks are supported.**

## What is in this folder

| File | What it does |
|---|---|
| `pipeline.py` | Runs everything on one capture: blocks, size, cracks. Start here |
| `download_models.py` | Downloads the trained models into `models/` |
| `block_detect.py` | Finds the 3 blocks and crops them |
| `dimensions.py` | Measures length, width and height; calibration tools |
| `infer.py` | Runs the crack model on a block image |
| `crack_length.py` | Turns a crack mask into a length |
| `train_block_yolov8.ipynb` | Trains the block detector (Google Colab) |
| `crack_pretrain_unet.ipynb` | Trains the crack model (Google Colab) |
| `evaluation/` | Compares 3 classifier designs against the thesis design constraints. Guide: [evaluation/README.md](evaluation/README.md) |
| `marker.png` | ArUco marker to print for the mm scale |
| `COMMANDS.md` | Setup steps and every command to run |

## Getting started

Needs Python 3.11 and git. On Windows:

```bash
git clone https://github.com/codeyson/cbp-crack-detection.git
cd cbp-crack-detection
python -m venv venv
venv\Scriptsctivate
pip install -r requirements.txt
python download_models.py
python pipeline.py --cam 0
```

A window shows the camera with a box around each block. Press SPACE to run the whole check, q to quit. Results are saved in `pipeline_out/`.

- No camera at hand: `python pipeline.py --img some_photo.png`
- Wrong camera: try `--cam 1`
- Lengths in mm need a scale: print `marker.png`, measure its black square with calipers, place it beside the blocks at block-top height, add `--scale_ref <mm>`. Details in "1.2 Dimensions" in [COMMANDS.md](COMMANDS.md).
- Already cloned: `git pull`, then `pip install -r requirements.txt` if it changed.

Check the install without a camera:

```bash
python pipeline.py --selftest
```

It should end with `ALL PASS`. All commands and calibration steps are in [COMMANDS.md](COMMANDS.md).

## Credits

The crack model was pretrained on DeepCrack (Liu et al., 2019, CC BY 4.0).

## License

MIT. See [LICENSE](LICENSE).
