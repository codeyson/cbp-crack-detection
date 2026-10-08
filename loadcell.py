"""
loadcell.py - read the 3 load cells (Left, Middle, Right) from the ESP32-S3 over USB serial (objective 1.2 weight).
Firmware and wiring: firmware/loadcells/loadcells.ino.

    python loadcell.py --port COM5                      print the 3 weights once
    python loadcell.py --port COM5 --tare               tare all 3 (platforms empty)
    python loadcell.py --port COM5 --cal 0 --known 1000 calibrate cell 0 (Left) with a 1000 g mass on it
    python loadcell.py --selftest                       no hardware needed
"""
import argparse
import math
import time

import serial

LABELS = ("Left", "Middle", "Right")


class ScaleError(Exception):
    pass


def open_scale(port, baud=115200):
    """Opens the port and waits up to 3 s for the ESP32's READY (opening the port may reset it)."""
    try:
        ser = serial.Serial(port, baud, timeout=1)
    except serial.SerialException as e:
        raise ScaleError(f"load cells: cannot open {port}: {e}") from e
    end = time.monotonic() + 3
    while time.monotonic() < end and ser.readline().strip() != b"READY":
        pass                                     # no READY is fine: some boards do not reset on open
    try:
        read_weights(ser)                        # something on this port must speak the protocol
    except ScaleError as e:
        raise ScaleError(f"load cells: {port} opened but does not answer like the ESP32 ({e})") from None
    return ser


def _ask(ser, cmd, tries=1):
    """tries: readline attempts (1 s timeout each). Tare/calibrate average 10 readings per cell at 10 Hz: seconds."""
    try:
        ser.reset_input_buffer()                 # drop anything stale
        ser.write((cmd + "\n").encode())
        reply = ""
        for _ in range(tries):
            reply = ser.readline().decode(errors="replace").strip()
            if reply:
                break
    except serial.SerialException as e:
        raise ScaleError(f"load cells: serial error: {e}") from e
    if not reply:
        raise ScaleError(f"load cells: no reply to {cmd!r} (unplugged? wrong port?)")
    return reply.split(",")


def read_weights(ser):
    """-> {"Left": g, "Middle": g, "Right": g, "stable": bool}"""
    r = _ask(ser, "W")
    try:
        g = list(map(float, r[1:4]))
        if r[0] == "W" and len(r) == 5 and all(map(math.isfinite, g)):   # nan/inf: bad calibration or wiring
            return dict(zip(LABELS, g), stable=r[4] == "1")
    except ValueError:
        pass
    raise ScaleError(f"load cells: bad reply to 'W': {','.join(r)}")


def tare(ser):
    r = _ask(ser, "T", tries=8)
    if r != ["OK", "T"]:
        raise ScaleError(f"load cells: tare failed: {','.join(r)}")


def calibrate(ser, i, known_g):
    """Known mass of known_g grams on cell i (0 Left, 1 Middle, 2 Right). Returns the saved factor (counts per g)."""
    r = _ask(ser, f"C,{i},{known_g:g}", tries=4)
    if r[:3] != ["OK", "C", str(i)] or len(r) != 4:
        raise ScaleError(f"load cells: calibrate failed: {','.join(r)}")
    return float(r[3])


class FakeSerial:
    """Answers the firmware protocol from a dict {command: reply}. A command not in the dict times out.
    slow: number of 1 s readline timeouts before the reply arrives (tare/calibrate take seconds on the ESP32)."""
    def __init__(self, replies, slow=0):
        self.replies, self.out, self.slow = replies, [], slow

    def reset_input_buffer(self):
        self.out = []

    def write(self, data):
        cmd = data.decode().strip()
        if cmd in self.replies:
            self.out.append(self.replies[cmd] + "\n")

    def readline(self):
        if self.slow:
            self.slow -= 1
            return b""
        return self.out.pop(0).encode() if self.out else b""


def selftest():
    s = FakeSerial({"W": "W,1000.5,998.0,1003.2,1"})
    assert read_weights(s) == {"Left": 1000.5, "Middle": 998.0, "Right": 1003.2, "stable": True}
    assert read_weights(FakeSerial({"W": "W,1,2,3,0"}))["stable"] is False
    tare(FakeSerial({"T": "OK,T"}, slow=4))                            # tare(10) x 3 cells ~3 s at 10 Hz
    assert calibrate(FakeSerial({"C,0,1000": "OK,C,0,421.7"}, slow=2), 0, 1000) == 421.7
    real = serial.Serial                                               # port opens but nothing answers
    serial.Serial = lambda *a, **k: FakeSerial({})
    try:
        open_scale("COM42")
        assert False, "silent port accepted"
    except ScaleError as e:
        assert "COM42" in str(e), e
    finally:
        serial.Serial = real
    for bad in ({"W": "ERR,W"}, {"W": "OK,T"}, {}, {"W": "W,x,2,3,1"}, {"W": "W,nan,2,3,1"}, {"W": "W,1,inf,3,1"}):
        try:
            read_weights(FakeSerial(bad))
            assert False, bad
        except ScaleError:
            pass
    print("selftest: ALL PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--port", help="ESP32 serial port, e.g. COM5 or /dev/ttyACM0")
    ap.add_argument("--tare", action="store_true", help="tare all 3 cells (platforms empty)")
    ap.add_argument("--cal", type=int, choices=(0, 1, 2), help="calibrate this cell: 0 Left, 1 Middle, 2 Right")
    ap.add_argument("--known", type=float, help="known mass on that cell, grams")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.port:
        ap.error("--port is required")
    if a.cal is not None and not a.known:
        ap.error("--cal needs --known")
    try:
        ser = open_scale(a.port)
        if a.tare:
            tare(ser)
            print("tared")
        elif a.cal is not None:
            print(f"cell {a.cal} factor {calibrate(ser, a.cal, a.known)}")
        else:
            print(read_weights(ser))
    except ScaleError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
