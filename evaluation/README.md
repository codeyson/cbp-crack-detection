# Design Constraint Evaluation

This folder compares 3 alternative designs of the grading classifier against the thesis design constraints. The rest of the repo (block detection, dimensions, U-Net) is not touched.

Each design is a deep-learning image classifier. It takes one block crop and outputs grade **A, B or C**. Reject is decided separately by the dimension and weight checks, which are the same for every design and are not part of this comparison.

| Design | Model | Size | Architecture family | Built for |
|---|---|---|---|---|
| D1 | MobileNetV3-Large | 5.5M params | Depthwise-separable, NAS-designed | Speed on mobile CPUs |
| D2 | EfficientNet-B0 | 5.3M params | Compound-scaled MBConv | Accuracy per parameter |
| D3 | ResNet18 | 11.7M params | Residual blocks | Classic, simple baseline |

All three start from ImageNet weights and are fine-tuned on our crops (transfer learning). The code is in [designs.py](designs.py).

Why these three:
1. **Same weight class.** All are lightweight, 5–12M parameters, and all ran within about 1.5× of each other on real crops. A heavy model such as ResNet50 (25.6M, about 4× slower) was left out, because it would lose speed and memory by default.
2. **Different architecture families.** Each one was designed with a different goal, so each has a likely strength.
3. **One framework.** All three come from torchvision and use the same training and prediction code. Only the architecture changes, so the framework cannot cause a difference.
4. **Practical.** All are common in concrete-crack classification studies, have ImageNet weights, and run on a CPU MiniPC.

## How each constraint is measured

| Constraint | Metric | How | Winner | Script |
|---|---|---|---|---|
| Reliability | Misclassification rate (%) | `(1 - accuracy_score) x 100` on the test set | lowest | `evaluate.py` |
| Performance | Average inference time (s) | `time.perf_counter()` around every prediction | lowest | `evaluate.py` |
| Efficiency | Average memory usage (%) | `psutil` `memory_percent()` after every prediction, averaged over 50 runs | lowest | `evaluate.py` |
| Manufacturability | Training time (s) | `time.perf_counter()` before and after training | lowest | `train.py` |
| Sustainability | Maintainability Index | `radon` on the design's source code: `designs.py` plus its torchvision architecture file, weighted by lines of code | highest | `evaluate.py` |

Rules that keep it fair:
- Same dataset split, epochs, image size (224), batch size, and seed for every design.
- Training runs in Colab on the same GPU type. The GPU name is saved with the result.
- Downloading the ImageNet weights is not counted in training time.
- Final inference runs on the MiniPC (the rig computer), one design per run, with other apps closed. The first 5 predictions (warm-up) and reading the image from disk are not timed.
- Radon measures the code a future researcher would read to change a design: our `designs.py` (the same for all three) plus that model's torchvision file (`mobilenetv3.py`, `efficientnet.py` or `resnet.py`).

Outputs go to `evaluation/results/`:
- `training_time.csv`
- `evaluation.csv`
- `confusion_<design>.csv`

Each `evaluate.py --design dX` call does **50 runs**, each in a fresh process with one inspection cycle over all the images. The 50 rows share one batch ID. `tradeoff.py` averages the newest batch of each design and ignores older batches. A run takes about 10 s on 36 crops, so one design takes about 8–9 minutes. Use `--runs 1` for a quick check.

## What can be done without the A/B/C dataset

| Constraint | Without dataset? |
|---|---|
| Sustainability (MI) | Yes, final. Radon only reads code |
| Performance, Efficiency | Yes, provisional. Speed and memory depend on the model's size, not on what it learned. Re-run on the trained models for the final table |
| Manufacturability | Rehearsal only. Training time depends on dataset size |
| Reliability | No. Needs verified A/B/C labels |

## Step by step

Run all commands from the repo folder, with the venv active.

### Part A: Now (no dataset needed)

1. Install the extras:

   ```bash
   pip install -r requirements.txt -r evaluation/requirements.txt
   ```

2. Check that everything works. This creates fake images, trains each design for 1 epoch, and evaluates it. It takes a few minutes and must end with `selftest: ALL PASS`. The numbers it prints mean nothing.

   ```bash
   python evaluation/evaluate.py --selftest
   ```

3. The Maintainability Index needs no separate step: every `evaluate.py` run records it in `maintainability_index`, and it is already the final value.

4. Get provisional inference time and memory. These use real block crops at 224 px with untrained models; speed and memory depend on the model's size, not on what it learned.
   - Put real crops in a folder named `crops`: copy only the `*_crop.png` files from `captures`, because that folder also holds raw frames and overlays.
   - Run once per design:

   ```bash
   python evaluation/evaluate.py --design d1 --img_dir crops
   ```

   Repeat with `--design d2` and `--design d3`.

   These rows go to `evaluation/results/provisional/`, never into the final table. Every row records the `device` (computer name) it ran on. View the table and the trade-off analysis:

   ```bash
   python evaluation/tradeoff.py --results evaluation/results/provisional
   ```

   Numbers from the dev laptop are only a preview. The final ones come from the MiniPC in step 11.

### Part B: Build the dataset

5. Agree on the A/B/C crack thresholds with the adviser. **Do this before labeling.** Changing them later means relabeling everything.
6. Capture blocks on the rig. Each SPACE press saves one crop per block. Aim for **at least 100 crops per grade**, under the same camera and lighting that will be used in the final system.

   ```bash
   python block_detect.py --cam 0 --conf 0.3 --out captures
   ```

7. Sort the `*_crop.png` files by hand into `raw/A`, `raw/B`, `raw/C`. Have two people label when possible, and settle disagreements together.
8. Split the crops into train, validation and test sets (70/15/15 per grade, fixed seed):

   ```bash
   python evaluation/make_split.py raw dataset
   ```

   From now on, **never use `dataset/test` for anything except step 11.**

### Part C: Final evaluation

9. Train in Colab:
   - Upload `dataset/` to Google Drive at `MyDrive/cbp/dataset`.
   - Open `evaluation/train_designs.ipynb` in Colab and pick the T4 GPU.
   - Run all cells. It trains D1, D2 and D3 with the same settings.
   - It downloads `designs_out.zip`, which holds the weights and `training_time.csv`. This is the **final** training time.
10. Set up the MiniPC, the computer that runs the rig. Do this once:
    - Install Python 3.11 and git.
    - Clone the repo, create the venv, and install everything (same steps as the root README):

    ```bash
    pip install -r requirements.txt -r evaluation/requirements.txt
    ```

    - Unzip `designs_out.zip` in the repo folder. The weights land in `models/` and `training_time.csv` in `evaluation/results/`.
    - Copy `dataset/test` from the laptop to `dataset/test` on the MiniPC. Only the test split is needed.
11. On the MiniPC, plug in the power, close other apps, and run each design. These are the **final** misclassification, inference time and memory values:

    ```bash
    python evaluation/evaluate.py --design d1
    ```

    Repeat with `--design d2` and `--design d3`. Run step 12 on the MiniPC, or copy `evaluation/results/` back to the laptop first.
12. Run the trade-off analysis for Chapter 4. It prints the table with the winner per constraint and saves the charts. Run it again whenever the results or `constraints.json` change:

    ```bash
    python evaluation/tradeoff.py
    ```

    It also works in Part A. Any constraint that doesn't have a value for every design yet is skipped.

## Trade-off analysis (`tradeoff.py`)

The constraints live in [constraints.json](constraints.json), not in code. Each one has:
- a name
- the metric
- the results CSV and column it comes from
- whether `lower` or `higher` is better
- a weight

Add, remove or reweight constraints there. Weights are relative: `1,1,1,1,1` and `0.2` each give the same result.

| Step | Formula | Output |
|---|---|---|
| Normalized score (0–10) | lower is better: `best / value x 10`; higher is better: `value / best x 10` | `radar.png`, `tradeoff.csv` |
| Overall score | `sum(weight x score)`, with the weights scaled to sum to 1 | `tradeoff.csv`, rows in rank order |
| Pareto | a design is optimal unless another design is at least as good on every constraint and better on one | `pareto_optimal` column |
| Sensitivity | each constraint's weight is swept from 0 to 1, and the others keep their ratios | `sensitivity.png`, printed flip weights |
| Run consistency | one polar chart per constraint that varies between runs: each angle is one run, the radius is that run's value | `runs.png` |

How to read the output:
- **All designs are Pareto-optimal** means each one wins somewhere. None is simply worse, so the fight is close.
- **A small overall margin** also means the fight is close.
- **In `sensitivity.png`**, a red line marks the weight where the winner changes.
  - A red line close to the gray line (the current weight) means the winner is fragile.
  - "winner never changes" means the winner holds no matter how important that constraint is.
- **In `runs.png`**, lines that stay apart all the way around mean the ranking holds in every run. Lines that cross mean the designs are too close to separate on that constraint.
  - Only inference time and memory change from run to run.
  - The Maintainability Index, misclassification (with fixed trained weights) and training time are the same every run, so they are not drawn.

## Notes for the paper

- Each torchvision architecture file holds the whole model family. For example, `resnet.py` has ResNet18 through ResNet152, and `efficientnet.py` has B0–B7 and V2. Say this when you report the Maintainability Index.
- All three designs use identical augmentation (random resized crop and flips), optimizer (Adam, lr 0.001), epochs, image size and batch size.
- `confusion_<design>.csv` shows which grades get mixed up. Panels usually ask for it.
- Report the hardware with the results:
  - the CPU and total RAM (saved in `evaluation.csv`)
  - the GPU (saved in `training_time.csv`)
