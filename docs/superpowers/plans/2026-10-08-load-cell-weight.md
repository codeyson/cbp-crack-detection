# Load-Cell Weight + Results Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read 3 load cells (one per block position) from an ESP32-S3 over USB serial, attach `weight_g` to each block in `pipeline.py` with a pass/fail tolerance check, and show L/W/H/weight/result in an OpenCV table window.

**Architecture:** Firmware answers a line-based request/response protocol. `loadcell.py` is the only module that talks serial. `pipeline.run()` takes an already-read weights dict, so it stays testable without hardware; `run_camera()`/`main()` do the reading.

**Tech Stack:** Arduino (ESP32 core, `bogde/HX711`, `Preferences`), Python 3.11, `pyserial`, OpenCV.

**Spec:** `docs/superpowers/specs/2026-10-08-load-cell-weight-design.md`

## Global Constraints

- Tests follow the repo pattern: an `assert`-based `selftest()` per module, run with `python <module>.py --selftest`, ending in `print("selftest: ALL PASS")`. No pytest.
- Protocol: 115200 baud, `W` -> `W,<left_g>,<middle_g>,<right_g>,<stable>`, `T` -> `OK,T`, `C,<i>,<known_g>` -> `OK,C,<i>,<factor>`, else `ERR,<line>`; boot prints `READY`.
- Pins: Left DT 4 / SCK 5, Middle DT 6 / SCK 7, Right DT 15 / SCK 16. `WINDOW` 10, `STABLE_G` 2.0.
- Weight dict keys: `"Left"`, `"Middle"`, `"Right"`, `"stable"` (match `block_detect` labels).
- Only new dependency: `pyserial` (pin the version installed in the venv).

## Review Focus

- ESP32 resets when the port opens: first command must not be lost. `open_scale` waits for `READY` (timeout ~3 s) and drains input. Test: fake serial that emits `READY` before answering.
- A reply line of a different command (stale, e.g. `OK,T` when reading `W`): `read_weights` raises `ScaleError`, not a parse crash. Test in Task 2.
- Blocks present but detection not "ok" (wrong count): weights must not be attached or crash. Test in Task 3.
- Detection labels are `Block 1..n` (not Left/Middle/Right) -> no weight match: `weight_g` stays `None`, no crash. Covered by the wrong-count test.
- Serial unplugged mid-session: `read_weights` raises `ScaleError` (timeout or `serial.SerialException` wrapped); live loop prints a warning and continues. Test: timeout case in Task 2.

---

### Task 1: Firmware

**Files:**
- Create: `firmware/loadcells/loadcells.ino`

**Interfaces:**
- Produces: the serial protocol in Global Constraints.

- [ ] **Step 1: Write the sketch.** Header comment = the pin/wiring table and load-cell colours from the spec, the Arduino IDE settings (`bogde/HX711` library, *USB CDC On Boot: Enabled*). Per cell: `HX711`, float ring[`WINDOW`], tare offset (long), factor (float, default 1.0). `setup()`: `Serial.begin(115200)`, `begin(DT, SCK)` each, load `off0..2`/`fac0..2` from `Preferences` namespace `"cells"`, `set_offset`/`set_scale`, print `READY`. `loop()`: for each cell `if (is_ready()) ring push get_units(1)`; read a line when available and dispatch. `T` -> `tare(10)` each, save offsets. `C,i,g` -> `set_scale(get_value(10) / g)`, save factor. Means printed with 1 decimal.
- [ ] **Step 2: Verify.** No compiler in this repo: check by reading against the protocol table. On the rig (manual, owner: hardware team): Serial Monitor at 115200, Newline line ending: `T`, then `C,0,<known>` with a known mass on Left, repeat for 1 and 2, then `W` matches a kitchen scale within a few grams.
- [ ] **Step 3: Commit** `feat: ESP32-S3 firmware for 3 load cells`

### Task 2: `loadcell.py`

**Files:**
- Create: `loadcell.py`
- Modify: `requirements.txt` (add `pyserial==<installed version>`)

**Interfaces:**
- Produces: `class ScaleError(Exception)`; `open_scale(port: str, baud: int = 115200) -> serial.Serial`; `read_weights(ser) -> dict` (`{"Left": float, "Middle": float, "Right": float, "stable": bool}`); `tare(ser) -> None`; `calibrate(ser, i: int, known_g: float) -> float` (factor). All raise `ScaleError` on timeout, `ERR`, wrong reply, or `serial.SerialException`.

- [ ] **Step 1: Write `selftest()`** with a `FakeSerial` (`write(bytes)` queues the reply per protocol, `readline()` returns it or `b""` for timeout, `reset_input_buffer()`):

```python
s = FakeSerial({"W": "W,1000.5,998.0,1003.2,1"})
assert read_weights(s) == {"Left": 1000.5, "Middle": 998.0, "Right": 1003.2, "stable": True}
assert read_weights(FakeSerial({"W": "W,1,2,3,0"}))["stable"] is False
tare(FakeSerial({"T": "OK,T"}))
assert calibrate(FakeSerial({"C,0,1000": "OK,C,0,421.7"}), 0, 1000) == 421.7
for bad in ({"W": "ERR,W"}, {"W": "OK,T"}, {}):          # error, stale reply, timeout
    try: read_weights(FakeSerial(bad)); assert False, bad
    except ScaleError: pass
```

- [ ] **Step 2:** `python loadcell.py --selftest` -> fails (functions missing).
- [ ] **Step 3: Implement.** `_ask(ser, cmd)`: `reset_input_buffer()`, write `cmd + "\n"`, `readline()` with the serial timeout (1 s), decode/strip; empty -> `ScaleError("no reply ...")`. `open_scale`: `serial.Serial(port, baud, timeout=1)`, read lines up to 3 s until `READY` (ok if never seen: board may not reset on open), port error -> `ScaleError` naming the port. CLI (`argparse`): `--port`, `--tare`, `--cal I --known G`, `--selftest`; no action flag -> print `read_weights` once.
- [ ] **Step 4:** `pip install pyserial`, then `python loadcell.py --selftest` -> `selftest: ALL PASS`.
- [ ] **Step 5: Commit** `feat: loadcell.py serial client for the 3 load cells`

### Task 3: Weight in `pipeline.run()` and the saved record

**Files:**
- Modify: `pipeline.py` (`run`, `save`, `selftest`)

**Interfaces:**
- Consumes: weights dict from Task 2 (passed in, `run` never touches serial).
- Produces: `run(..., weights=None, nominal_g=None, tol_g=None)`; each block gains `weight_g: float | None`; `"weight_g"` in `save()`'s `keep` and in the console line.

- [ ] **Step 1: Add to `selftest()`:**

```python
w = {"Left": 1000.0, "Middle": 1200.0, "Right": 990.0, "stable": False}
r = run(img, yolo, dark, dict(mm_per_px_bed=0.5), weights=w, nominal_g=1000, tol_g=50)
assert [b["weight_g"] for b in r["blocks"]] == [1000.0, 1200.0, 990.0]
assert r["blocks"][1]["tolerance"] == "fail" and any("weight" in s for s in r["blocks"][1]["reasons"])
assert r["blocks"][0]["tolerance"] != "fail" and "load cells not stable" in r["warnings"]
r = run(img, FakeYOLO(boxes[:2]), dark, {}, weights=w)          # wrong count: no crash
assert r["status"] == "wrong_count"
```

  And after the existing `save()` check: `assert "weight_g" in saved[0]`.
- [ ] **Step 2:** `python pipeline.py --selftest` -> fails (`unexpected keyword 'weights'`).
- [ ] **Step 3: Implement** in `run()`, right after the `res["timings"]` line, before the `status != "ok"` return: if `weights`, warn when not stable, then per block `weight_g = weights.get(label)`; when `nominal_g`/`tol_g` and weight not None and off by more than `tol_g`: `tolerance = "fail"`, append `f"weight {w:.0f} g vs {nominal_g} +/- {tol_g}"` to `reasons` (create list if absent); else if `tolerance` missing or `"no_standard"` and a weight standard is given: `"pass"`. A weight pass never overrides `"fail"`.
- [ ] **Step 4:** `python pipeline.py --selftest` -> `selftest: ALL PASS`.
- [ ] **Step 5: Commit** `feat: block weight from load cells with tolerance check`

### Task 4: Results table, live weights, CLI flags, docs

**Files:**
- Modify: `pipeline.py` (`draw_table`, `run_camera`, `main`, `selftest`), `README.md`, `COMMANDS.md`

**Interfaces:**
- Consumes: `open_scale`, `read_weights`, `ScaleError` (Task 2); `run(weights=...)` (Task 3).
- Produces: `draw_table(res) -> np.ndarray` (BGR image).

- [ ] **Step 1: Add to `selftest()`:** `t = draw_table(r); assert t.ndim == 3 and t.shape[1] >= 600` for an ok result with weights, and for the no-scale result (all dims `None`).
- [ ] **Step 2:** selftest fails (`draw_table` missing).
- [ ] **Step 3: Implement `draw_table`.** White canvas, header row `Block | L (mm) | W (mm) | H (mm) | Weight (g) | Result`, one row per block, values 1 decimal or `-`. Result text: `PASS` green (0,160,0), `FAIL` red (0,0,220), any other `tolerance` value amber (0,140,255), `-` when absent.
- [ ] **Step 4: Wire it.** `main()`: flags `--scale` (port), `--nominal_g`, `--tol_g` (float); when `--scale`, `open_scale` at startup (`ScaleError` -> `ap.error(...)`), pass `scale=ser` to `run_camera`, and read once for `--img` runs; pass `nominal_g`/`tol_g` in `kw`. `run_camera(..., scale=None)`: every ~0.5 s (`time.monotonic`) refresh `live = read_weights(scale)`, put `L 1000.5  M 998.0  R 1003.2 g  stable` on the preview (red text when not stable); at SPACE use a fresh `read_weights` (`ScaleError` -> print warning, `weights=None`); show `cv2.imshow("results", draw_table(res))` after each run. Add `weight_g` to the `meta`-independent console print.
- [ ] **Step 5:** `python pipeline.py --selftest` and `python loadcell.py --selftest` -> both `ALL PASS`.
- [ ] **Step 6: Docs.** README status row "Weight from load cells" -> "Working, needs calibration on the real rig"; file table gets `loadcell.py` and `firmware/loadcells/`. `COMMANDS.md`: a "Load cells" section with flash steps, `--tare`, `--cal`, and `python pipeline.py --cam 0 --scale COM5 --nominal_g <g> --tol_g <g>`.
- [ ] **Step 7: Commit** `feat: results table and live load-cell weights in the pipeline`
