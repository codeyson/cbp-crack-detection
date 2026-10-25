#include <Arduino.h>
#include "HX711.h"

const uint8_t SENSOR_COUNT = 3;

const uint8_t DT_PINS[SENSOR_COUNT] = {5, 7, 16};
const uint8_t SCK_PINS[SENSOR_COUNT] = {4, 6, 15};

HX711 scales[SENSOR_COUNT];

// Unloaded offsets
const double OFFSETS[SENSOR_COUNT] = {
  -157693.250,  // Sensor 1
  -215584.800,  // Sensor 2
  -270299.550   // Sensor 3
};

// Raw counts per kilogram
const double CALIBRATION_FACTORS[SENSOR_COUNT] = {
  -414196.339,  // Sensor 1
  -415572.225,  // Sensor 2
  -414115.997   // Sensor 3
};

long latestRaw[SENSOR_COUNT] = {0, 0, 0};
bool hasReading[SENSOR_COUNT] = {false, false, false};

unsigned long lastReadingMs[SENSOR_COUNT] = {0, 0, 0};
unsigned long lastReportMs = 0;

const unsigned long REPORT_INTERVAL_MS = 250;
const unsigned long STALE_TIMEOUT_MS = 1500;

void setup() {
  Serial0.begin(115200);
  delay(1500);

  for (uint8_t i = 0; i < SENSOR_COUNT; i++) {
    scales[i].begin(DT_PINS[i], SCK_PINS[i]);
  }
}

void loop() {
  for (uint8_t i = 0; i < SENSOR_COUNT; i++) {
    if (scales[i].is_ready()) {
      latestRaw[i] = scales[i].read();
      lastReadingMs[i] = millis();
      hasReading[i] = true;
    }
  }

  unsigned long now = millis();

  if (now - lastReportMs >= REPORT_INTERVAL_MS) {
    lastReportMs = now;

    Serial0.print("DATA");

    for (uint8_t i = 0; i < SENSOR_COUNT; i++) {
      Serial0.print(",");

      bool recentReading =
          hasReading[i] &&
          (now - lastReadingMs[i] <= STALE_TIMEOUT_MS);

      if (recentReading) {
        double weightKg =
            ((double)latestRaw[i] - OFFSETS[i]) /
            CALIBRATION_FACTORS[i];

        Serial0.print(weightKg, 3);
      } else {
        Serial0.print("NA");
      }
    }

    Serial0.println();
  }

  delay(1);
}
