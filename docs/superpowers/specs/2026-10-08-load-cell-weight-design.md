# Load-cell weight + results UI (objective 1.2)

Date: 2026-10-08. Status: approved design, awaiting spec review.

## Goal

Objective 1.2 needs each block's **weight** next to its L/W/H. The rig has **3 load cells, one under each block
position** (Left, Middle, Right; decision D3). Weight is a pass/fail tolerance check like L/W/H (D7, D19).
The operator also needs a simple on-screen table of L, W, H, weight and pass/fail per block.

## Hardware

- ESP32-S3 dev board, 3x HX711 amplifier boards, 3 load cells. USB serial (wired) to the mini PC.
- Every HX711: VCC -> 3V3, GND -> GND. 3V3, not 5V: the ESP32-S3 pins are not 5 V tolerant.

| Position | Index | HX711 DT (DOUT) | HX711 SCK |
|---|---|---|---|
| Left | 0 | GPIO 4 | GPIO 5 |
| Middle | 1 | GPIO 6 | GPIO 7 |
| Right | 2 | GPIO 15 | GPIO 16 |

Load cell -> HX711 (usual colours, check the datasheet): Red E+, Black E-, White A-, Green A+.
Reading goes negative under load: swap A+ and A-.

## Firmware: `firmware/loadcells/loadcells.ino`

- Arduino IDE, ESP32 core, library **HX711 by Bogdan Necula** (`bogde/HX711`), `Preferences` (built in).
- Native USB port: Tools -> USB CDC On Boot: Enabled.
- Loop: read each HX711 when ready (gain 128, channel A) into a per-cell ring of the last `WINDOW` (10) readings.
- Calibration factors (counts per gram) and tare offsets are stored in flash with `Preferences`, so they survive reboot.
- Serial protocol, 115200 baud, one command per line, one reply line per command:

| Command | Reply | Meaning |
|---|---|---|
| `W` | `W,<left_g>,<middle_g>,<right_g>,<stable>` | mean of each ring, grams, 1 decimal; stable = 1 when every ring's max-min < `STABLE_G` (2.0 g) |
| `T` | `OK,T` | tare all 3 cells (empty platforms), saved |
| `C,<i>,<known_g>` | `OK,C,<i>,<factor>` | calibrate cell i with a known mass on it, saved |
| anything else | `ERR,<line>` | |

- On boot prints `READY` once. `WINDOW` and `STABLE_G` are `#define`s to tune on the rig.
- Request/response, not streaming: the PC never parses stale buffered lines.

## PC side: `loadcell.py`

- Dependency: `pyserial` (added to `requirements.txt`).
- `open_scale(port, baud=115200)` -> serial object (waits out the ESP32 reset on open, drains `READY`).
- `read_weights(ser)` -> `{"Left": g, "Middle": g, "Right": g, "stable": bool}`. Keys match `block_detect` labels.
- `tare(ser)`, `calibrate(ser, i, known_g)`: send the command, check the `OK` reply, raise on `ERR`/timeout.
- CLI: `python loadcell.py --port COM5` (print readings), `--tare`, `--cal 0 --known 1000`, `--selftest`.
- Self-test uses a fake serial object that answers the protocol; no hardware needed.

## Pipeline integration: `pipeline.py`

- New flags: `--scale COM5`, `--nominal_g`, `--tol_g`.
- `run(..., weights=None, nominal_g=None, tol_g=None)`: when `weights` is given, each block gets `weight_g`.
  If `nominal_g` and `tol_g` are set and `|weight_g - nominal_g| > tol_g`: `tolerance = "fail"` and a reason
  `"weight <g> g vs <nominal> +/- <tol>"` is appended. A weight pass never overrides an L/W/H fail.
  If `weights["stable"]` is false: warning `"load cells not stable"`, values are still used.
- Live (`--cam` + `--scale`): weights are read at SPACE time, together with the frame.
- Offline (`--img`): no weight unless `--scale` is given.
- `weight_g` is saved per block in `summary.json` and printed in the console line.

## Results UI

- A `results` window drawn with OpenCV (already installed, no new UI library), shown after each SPACE run.
- One row per block (Left, Middle, Right). Columns: L (mm), W (mm), H (mm), Weight (g), Result.
  Result is PASS (green), FAIL (red), or the raw status (amber) when there is no standard. Missing values show `-`.
- Function `draw_table(res)` in `pipeline.py` returns the image; the self-test checks it renders.
- Live preview also shows the current 3 weights and the stable flag, polled about every 0.5 s, not every frame.

## Error handling

- Serial port missing or not answering at startup: exit with a clear message naming the port.
- Read timeout or bad reply during a run: warning, weights skipped for that capture, pipeline continues.

## Testing

- `python loadcell.py --selftest`: protocol parsing, tare/calibrate replies, error reply, timeout.
- `python pipeline.py --selftest`: weight attached by label, weight fail sets `tolerance = "fail"`, `draw_table` renders.
- Firmware: manual on the rig (tare, calibrate with a known mass, check `W` against a kitchen scale).

## Out of scope

WiFi, a GUI framework, temperature compensation, the weight standard values themselves (team supplies
`--nominal_g` / `--tol_g`).
