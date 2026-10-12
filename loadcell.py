"""
loadcell.py - read the 3 load cells (Left, Middle, Right) from the ESP32-S3 over USB serial (objective 1.2 weight).
Firmware: firmware/loadcells/hx711_monitor.ino. It streams "DATA,<left_kg>,<middle_kg>,<right_kg>" every 250 ms
("NA" for a cell with no fresh reading). Tare and calibration are the OFFSETS / CALIBRATION_FACTORS in the sketch.

    python loadcell.py --port COM5                      print the 3 weights (about 1 s of readings)
    python loadcell.py --selftest                       no hardware needed
"""
import argparse
import math

import serial

LABELS = ("Left", "Middle", "Right")
WINDOW = 4        # readings kept for the stability check: 1 s at the sketch's 250 ms
STABLE_G = 2.0    # stable when every cell's last WINDOW readings span less than this (grams)


class ScaleError(Exception):
    pass


def open_scale(port, baud=115200):
    """Opens the port and checks that a DATA line arrives (opening the port may reset the ESP32: a few tries)."""
    try:
        ser = serial.Serial(port, baud, timeout=1)
    except serial.SerialException as e:
        raise ScaleError(f"load cells: cannot open {port}: {e}") from e
    for i in range(3):
        try:
            read_weights(ser)
            return ser
        except ScaleError as e:
            err = e
    raise ScaleError(f"load cells: {port} opened but does not stream like the ESP32 ({err})")


def _parse(line):
    """'DATA,kg,kg,kg' -> [g, g, g]; None for any other line (boot messages)."""
    r = line.split(",")
    if r[0] != "DATA" or len(r) != 4:
        return None
    if "NA" in r:
        raise ScaleError(f"load cells: no reading from {LABELS[r.index('NA') - 1]} (check that HX711's wiring)")
    try:
        g = [round(float(v) * 1000, 1) for v in r[1:]]   # sketch prints kg to 3 decimals = 1 g
    except ValueError:
        g = [math.nan]
    if not all(map(math.isfinite, g)):
        raise ScaleError(f"load cells: bad line: {line}")
    return g


def read_weights(ser):
    """Takes every DATA line buffered since the last call (waits for one if none).
    -> {"Left": g, "Middle": g, "Right": g, "stable": bool}, newest reading; stable over the last WINDOW lines."""
    hist = ser.__dict__.setdefault("hist", [])
    new, err = False, None
    try:
        for _ in range(1000):                    # bounded: a port spewing non-DATA lines must not hang us
            if (new or err) and not ser.in_waiting:
                break
            line = ser.readline().decode(errors="replace").strip()
            if not line:
                break
            try:
                g = _parse(line)
            except ScaleError as e:              # one NA/bad line (boot, missed read): only fatal if it is the latest
                err, new = e, False
                hist.clear()
                continue
            if g:
                hist.append(g)
                new = True
    except serial.SerialException as e:
        raise ScaleError(f"load cells: serial error: {e}") from e
    if not new:
        raise err or ScaleError("load cells: no DATA line (unplugged? wrong port? sketch not running?)")
    del hist[:-WINDOW]
    stable = len(hist) == WINDOW and all(max(c) - min(c) < STABLE_G for c in zip(*hist))
    return dict(zip(LABELS, hist[-1]), stable=stable)


class FakeSerial:
    """Hands out the given lines one per readline(), then times out (b"")."""
    def __init__(self, lines):
        self.lines = [ln.encode() for ln in lines]

    @property
    def in_waiting(self):
        return len(self.lines)

    def readline(self):
        return self.lines.pop(0) + b"\n" if self.lines else b""


def selftest():
    s = FakeSerial(["boot", "DATA,1.000,0.998,1.003"])
    assert read_weights(s) == {"Left": 1000.0, "Middle": 998.0, "Right": 1003.0, "stable": False}   # 1 reading
    s.lines = [b"DATA,1.000,0.998,1.003"] * 3
    assert read_weights(s)["stable"] is True                           # 4 readings within 2 g
    s.lines = [b"DATA,1.010,0.998,1.003"]
    w = read_weights(s)                                                # Left moved 10 g: not stable, newest value
    assert w["Left"] == 1010.0 and w["stable"] is False, w
    assert read_weights(FakeSerial(["DATA,1,NA,1", "DATA,1,1,1"]))["Left"] == 1000.0   # NA then good: fine
    for bad, msg in ((["DATA,1,NA,1"], "Middle"), (["DATA,1,1,1", "DATA,1,1,NA"], "Right"), (["DATA,x,1,1"], "bad line"), (["DATA,nan,1,1"], "bad line"),
                     ([], "no DATA"), (["hello"] * 5, "no DATA")):
        try:
            read_weights(FakeSerial(bad))
            assert False, bad
        except ScaleError as e:
            assert msg in str(e), (bad, e)
    real = serial.Serial                                               # port opens but nothing streams
    serial.Serial = lambda *a, **k: FakeSerial([])
    try:
        open_scale("COM42")
        assert False, "silent port accepted"
    except ScaleError as e:
        assert "COM42" in str(e), e
    finally:
        serial.Serial = real
    print("selftest: ALL PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--port", help="ESP32 serial port, e.g. COM5 or /dev/ttyACM0")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.port:
        ap.error("--port is required")
    try:
        ser = open_scale(a.port)
        for _ in range(WINDOW):                  # fill the stability window
            w = read_weights(ser)
        print(w)
    except ScaleError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
