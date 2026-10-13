#include <Arduino.h>

void setup() {
  Serial0.begin(115200);
  delay(1500);
}

void loop() {
  Serial0.println("CH343 UART0 serial working");
  delay(1000);
}