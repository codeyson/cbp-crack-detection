#include <Arduino.h>
#include "HX711.h"
const uint8_t SENSOR_COUNT = 3;
const uint8_t DT_PINS[3] = {5,7,16};
const uint8_t SCK_PINS[3] = {4,6,15};
HX711 scales[3];
const double OFFSETS[3] = {-161669.900,-213026.000,-273475.000};
const double CALIBRATION_FACTORS[3] = {-430110.900,-431443.000,-430185.800};
long latestRaw[3] = {0,0,0};
bool hasReading[3] = {false,false,false};
unsigned long lastReadingMs[3] = {0,0,0};
unsigned long lastReportMs = 0;
const unsigned long REPORT_INTERVAL_MS = 250;
const unsigned long STALE_TIMEOUT_MS = 1500;
void setup() {
  Serial.begin(115200);
  for (uint8_t i=0;i<SENSOR_COUNT;i++) scales[i].begin(DT_PINS[i],SCK_PINS[i]);
}
void loop() {
  for (uint8_t i=0;i<SENSOR_COUNT;i++) {
    if (scales[i].is_ready()) {
      latestRaw[i]=scales[i].read();
      lastReadingMs[i]=millis();
      hasReading[i]=true;
    }
  }
  unsigned long now=millis();
  if (now-lastReportMs>=REPORT_INTERVAL_MS) {
    lastReportMs=now;
    Serial.print("DATA");
    for (uint8_t i=0;i<SENSOR_COUNT;i++) {
      Serial.print(",");
      if (hasReading[i] && now-lastReadingMs[i]<=STALE_TIMEOUT_MS) {
        double weightKg=((double)latestRaw[i]-OFFSETS[i])/CALIBRATION_FACTORS[i];
        Serial.print(weightKg,3);
      } else Serial.print("NA");
    }
    Serial.println();
  }
  delay(1);
}
