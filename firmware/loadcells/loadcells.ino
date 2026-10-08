/*
  loadcells.ino - ESP32-S3 + 3x HX711, one load cell under each block position. Talks to loadcell.py over USB serial.

  Arduino IDE: board "ESP32S3 Dev Module" (ESP32 core by Espressif), library "HX711" by Bogdan Necula (bogde).
  Using the board's native USB port: Tools -> USB CDC On Boot: Enabled.

  Wiring. Every HX711: VCC -> 3V3, GND -> GND (3V3, not 5V: ESP32-S3 pins are not 5 V tolerant).

    Position  Index  HX711 DT (DOUT)  HX711 SCK
    Left      0      GPIO 4           GPIO 5
    Middle    1      GPIO 6           GPIO 7
    Right     2      GPIO 15          GPIO 16

  Load cell -> HX711 (usual colours, check the datasheet): Red E+, Black E-, White A-, Green A+.
  Reading goes negative under load: swap A+ and A-.

  Serial protocol, 115200 baud, one command per line (Serial Monitor: "Newline"), one reply line each:
    W            -> W,<left_g>,<middle_g>,<right_g>,<stable 0|1>
    T            -> OK,T                 tare all 3 (platforms empty), saved to flash
    C,<i>,<g>    -> OK,C,<i>,<factor>    calibrate cell i with a known mass of g grams on it, saved to flash
    other        -> ERR,<line>           (T/C also reply ERR,T,<i> / ERR,C,<i> ... when an HX711 does not answer)
  Prints READY once at boot.

  First use: T with all platforms empty, then C,0,<g> with a known mass on Left, C,1,<g> on Middle, C,2,<g> on Right.
*/
#include <HX711.h>
#include <Preferences.h>

#define WINDOW 10      // readings averaged per cell
#define STABLE_G 2.0   // stable when every cell's last WINDOW readings span less than this (grams)

const int DT_PIN[3] = {4, 6, 15};   // not "SCK": the ESP32 core already defines SCK (SPI pin)
const int SCK_PIN[3] = {5, 7, 16};

HX711 cell[3];
float ring[3][WINDOW];
int n[3], pos[3];      // readings in the ring (up to WINDOW), next slot
Preferences prefs;
String line;

void clearRing(int i) { n[i] = 0; pos[i] = 0; }

float mean(int i) {
  float s = 0;
  for (int k = 0; k < n[i]; k++) s += ring[i][k];
  return n[i] ? s / n[i] : 0;
}

bool stable(int i) {
  if (n[i] < WINDOW) return false;
  float lo = ring[i][0], hi = ring[i][0];
  for (int k = 1; k < WINDOW; k++) { lo = min(lo, ring[i][k]); hi = max(hi, ring[i][k]); }
  return hi - lo < STABLE_G;
}

void handle(String cmd) {
  char key[8];
  if (cmd == "W") {
    bool st = stable(0) && stable(1) && stable(2);
    Serial.printf("W,%.1f,%.1f,%.1f,%d\n", mean(0), mean(1), mean(2), st ? 1 : 0);
  } else if (cmd == "T") {
    for (int i = 0; i < 3; i++)        // tare blocks forever on a missing HX711: check all first
      if (!cell[i].wait_ready_timeout(1000)) { Serial.printf("ERR,T,%d not ready (check wiring)\n", i); return; }
    for (int i = 0; i < 3; i++) {
      cell[i].tare(10);
      snprintf(key, sizeof key, "off%d", i);
      prefs.putLong(key, cell[i].get_offset());
      clearRing(i);
    }
    Serial.println("OK,T");
  } else if (cmd.startsWith("C,")) {
    int i = cmd.substring(2, 3).toInt();
    float g = cmd.substring(4).toFloat();
    if (cmd.length() < 5 || cmd[3] != ',' || i < 0 || i > 2 || g <= 0) { Serial.println("ERR," + cmd); return; }
    if (!cell[i].wait_ready_timeout(1000)) { Serial.printf("ERR,C,%d not ready (check wiring)\n", i); return; }
    float factor = cell[i].get_value(10) / g;   // counts per gram, tare offset already removed
    if (fabs(factor) < 0.001) { Serial.printf("ERR,C,%d no load seen (mass on the cell? tared first?)\n", i); return; }
    cell[i].set_scale(factor);
    snprintf(key, sizeof key, "fac%d", i);
    prefs.putFloat(key, factor);
    clearRing(i);
    Serial.printf("OK,C,%d,%.4f\n", i, factor);
  } else {
    Serial.println("ERR," + cmd);
  }
}

void setup() {
  Serial.begin(115200);
  prefs.begin("cells", false);
  char key[8];
  for (int i = 0; i < 3; i++) {
    cell[i].begin(DT_PIN[i], SCK_PIN[i]);   // gain 128, channel A
    snprintf(key, sizeof key, "off%d", i);
    cell[i].set_offset(prefs.getLong(key, 0));
    snprintf(key, sizeof key, "fac%d", i);
    cell[i].set_scale(prefs.getFloat(key, 1.0));
    clearRing(i);
  }
  Serial.println("READY");
}

void loop() {
  for (int i = 0; i < 3; i++) {
    if (cell[i].is_ready()) {
      ring[i][pos[i]] = cell[i].get_units(1);
      pos[i] = (pos[i] + 1) % WINDOW;
      if (n[i] < WINDOW) n[i]++;
    }
  }
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      line.trim();
      if (line.length()) handle(line);
      line = "";
    } else if (line.length() < 32) {
      line += c;
    }
  }
}
