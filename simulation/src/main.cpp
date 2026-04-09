#include <Arduino.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <WiFi.h>
#include <HTTPClient.h>

#include "wokwi_test_vectors.h"
#include "gas_model.h"
#include "scaler_params.h"
#include "labels.h"

// ── Pins ─────────────────────────────────────────────
#define PIN_LED_RED     18   // Hazardous gas
#define PIN_LED_GREEN   19   // Safe / normal
#define PIN_LED_YELLOW  21   // Sensor fault
#define PIN_BUZZER      22   // Sounds only on hazardous gas
#define PIN_LED_BLUE    23   // Heartbeat — setup complete indicator

// ── WiFi & ThingSpeak ─────────────────────────────────
const char* WIFI_SSID  = "Wokwi-GUEST";
const char* WIFI_PASS  = "";
const char* TS_API_KEY = "SHE39S6B1CV71J88";
const char* TS_URL     = "https://api.thingspeak.com/update";

// ── Class logic ───────────────────────────────────────
// Verify these against your labels.h print from training
const int FAULT_CLASS_IDX   = 10;              // Sensor_Fault
const int HAZARD_CLASSES[]  = {2, 5, 6};       // Ammonia_10000, CO_1000, CO_4000
const int N_HAZARD          = 3;

// ── TFLite ────────────────────────────────────────────
const int TENSOR_ARENA_SIZE = 70 * 1024;
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

static tflite::MicroErrorReporter error_reporter;
static tflite::AllOpsResolver     resolver;
const  tflite::Model*             tflite_model = nullptr;
static tflite::MicroInterpreter*  interpreter  = nullptr;
TfLiteTensor* input_tensor  = nullptr;
TfLiteTensor* output_tensor = nullptr;

// ── Serial input buffer ───────────────────────────────
String inputBuffer = "";

// ─────────────────────────────────────────────────────
bool isHazardous(int idx) {
  for (int i = 0; i < N_HAZARD; i++)
    if (idx == HAZARD_CLASSES[i]) return true;
  return false;
}

void setOutputs(bool red, bool green, bool yellow, bool buzzer) {
  digitalWrite(PIN_LED_RED,    red    ? HIGH : LOW);
  digitalWrite(PIN_LED_GREEN,  green  ? HIGH : LOW);
  digitalWrite(PIN_LED_YELLOW, yellow ? HIGH : LOW);
  digitalWrite(PIN_BUZZER,     buzzer ? HIGH : LOW);
}

void allOff() { setOutputs(false, false, false, false); }
void setupFailed() {
  allOff();
  digitalWrite(PIN_LED_RED,    HIGH);
  digitalWrite(PIN_LED_GREEN,  HIGH);
  digitalWrite(PIN_LED_YELLOW, HIGH);
  while (1);
}

// ─────────────────────────────────────────────────────
void pushThingSpeak(int pred_idx, float conf,
                    bool is_fault,
                    float runner_conf, int runner_idx) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[TS] WiFi not connected — skipped");
    return;
  }
  HTTPClient http;
  String url = String(TS_URL)
    + "?api_key=" + TS_API_KEY
    + "&field1="  + pred_idx
    + "&field2="  + String(conf, 4)
    + "&field3="  + (is_fault ? 1 : 0)
    + "&field4="  + String(runner_conf, 4)
    + "&field5="  + runner_idx;

  http.begin(url);
  int code = http.GET();
  Serial.printf("[TS] HTTP %d | Entry: %s\n",
                code, http.getString().c_str());
  http.end();
}

// ─────────────────────────────────────────────────────
void runInference(float* raw_features) {
  // 1. Scale: StandardScaler using scaler_params.h
  float in_scale   = input_tensor->params.scale;
  int   in_zero_pt = input_tensor->params.zero_point;

  for (int i = 0; i < N_FEATURES; i++) {
    float scaled = (raw_features[i] - SCALER_MEAN[i]) / SCALER_STD[i];
    // Quantize float → int8
    int32_t q = (int32_t)(scaled / in_scale + in_zero_pt);
    // Clamp to int8 range
    q = q < -128 ? -128 : (q > 127 ? 127 : q);
    input_tensor->data.int8[i] = (int8_t)q;
  }

  // 2. Run
  TfLiteStatus status = interpreter->Invoke();
  if (status != kTfLiteOk) {
    Serial.println("[ERROR] Inference failed");
    return;
  }

  // 3. Dequantize output scores
  float out_scale   = output_tensor->params.scale;
  int   out_zero_pt = output_tensor->params.zero_point;

  float scores[N_CLASSES];
  for (int i = 0; i < N_CLASSES; i++)
    scores[i] = (output_tensor->data.int8[i] - out_zero_pt) * out_scale;

  // 4. Find top-1 and top-2
  int top1 = 0;
  for (int i = 1; i < N_CLASSES; i++)
    if (scores[i] > scores[top1]) top1 = i;

  int top2 = (top1 == 0) ? 1 : 0;
  for (int i = 0; i < N_CLASSES; i++)
    if (i != top1 && scores[i] > scores[top2]) top2 = i;

  bool is_fault  = (top1 == FAULT_CLASS_IDX);
  bool is_hazard = isHazardous(top1);

  // 5. Print result
  Serial.println("\n─────────────────────────────────────");
  Serial.printf("Predicted : %s\n",          CLASS_LABELS[top1]);
  Serial.printf("Confidence: %.1f%%\n",       scores[top1] * 100.0f);
  Serial.printf("Runner-up : %s (%.1f%%)\n",  CLASS_LABELS[top2], scores[top2] * 100.0f);

  // 6. Branch logic — 3 outcomes
  if (is_fault) {
    setOutputs(false, false, true, false);
    Serial.println("ACTION    : SENSOR FAULT — alarm suppressed, maintenance LED on");
    Serial.println("LEDs      : [ ] RED  [ ] GREEN  [Y] YELLOW  [ ] BUZZER");
  } else if (is_hazard) {
    setOutputs(true, false, false, true);
    Serial.printf ("ACTION    : HAZARDOUS GAS (%s) — EVACUATE!\n", CLASS_LABELS[top1]);
    Serial.println("LEDs      : [R] RED  [ ] GREEN  [ ] YELLOW  [!] BUZZER");
  } else {
    setOutputs(false, true, false, false);
    Serial.printf ("ACTION    : Safe (%s) — monitoring\n", CLASS_LABELS[top1]);
    Serial.println("LEDs      : [ ] RED  [G] GREEN  [ ] YELLOW  [ ] BUZZER");
  }

  // 7. Push to ThingSpeak
  pushThingSpeak(top1, scores[top1], is_fault, scores[top2], top2);
}

// ─────────────────────────────────────────────────────
void parseAndRun(String& line) {
  float features[N_FEATURES];
  int   idx   = 0;
  int   start = 0;

  for (int i = 0; i <= (int)line.length() && idx < N_FEATURES; i++) {
    if (i == (int)line.length() || line[i] == ',') {
      features[idx++] = line.substring(start, i).toFloat();
      start = i + 1;
    }
  }

  if (idx == N_FEATURES) {
    runInference(features);
  } else {
    Serial.printf("[ERROR] Expected %d features, received %d\n",
                  N_FEATURES, idx);
  }
}

// ─────────────────────────────────────────────────────
void setup() {
  Serial.setRxBufferSize(4096);  // 288 floats * ~10 chars = ~2900 bytes
  Serial.begin(115200);
  delay(500);

  pinMode(PIN_LED_RED,    OUTPUT);
  pinMode(PIN_LED_GREEN,  OUTPUT);
  pinMode(PIN_LED_YELLOW, OUTPUT);
  pinMode(PIN_BUZZER,     OUTPUT);
  allOff();
  pinMode(PIN_LED_BLUE, OUTPUT);

  // Startup blink — visual confirm
  for (int i = 0; i < 3; i++) {
    setOutputs(true, true, true, false); delay(150);
    allOff();                            delay(150);
  }

  // Load TFLite model
  Serial.println("\n[TFLite] Loading model...");
  tflite_model = tflite::GetModel(gas_model_data);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("[ERROR] Schema version mismatch — recompile model");
    setupFailed();
  }

  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver,
      tensor_arena, TENSOR_ARENA_SIZE,
      &error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("[ERROR] AllocateTensors failed — increase TENSOR_ARENA_SIZE");
    setupFailed();
  }

  input_tensor  = interpreter->input(0);
  output_tensor = interpreter->output(0);
  Serial.printf("[TFLite] Ready | Input [1x%d] int8 | Output [1x%d] int8\n",
                input_tensor->dims->data[1],
                output_tensor->dims->data[1]);

  // WiFi
  Serial.printf("[WiFi] Connecting to %s", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  int tries = 0;
  while (WiFi.status() != WL_CONNECTED && tries++ < 20) {
    delay(500); Serial.print(".");
  }
  Serial.println(WiFi.status() == WL_CONNECTED
    ? "\n[WiFi] Connected"
    : "\n[WiFi] Failed — ThingSpeak posts will be skipped");

  Serial.println("\n[READY] Waiting for 288 comma-separated sensor values...");
  digitalWrite(PIN_LED_YELLOW, HIGH);
}

// ─────────────────────────────────────────────────────
// void loop() {
//   // Heartbeat blink — visual indicator that setup is done
//   digitalWrite(PIN_LED_BLUE, HIGH); delay(300);
//   digitalWrite(PIN_LED_BLUE, LOW);  delay(300);

//   while (Serial.available()) {
//     char c = (char)Serial.read();
//     if (c == '\n') {
//       inputBuffer.trim();
//       if (inputBuffer.length() > 0)
//         parseAndRun(inputBuffer);
//       inputBuffer = "";
//     } else if (c != '\r') {
//       inputBuffer += c;
//     }
//   }
// }

// ── Stratified 5-sample selection from 20 test vectors ──
// Indices by category (from wokwi_test_vectors.h true labels):
// Hazardous: check TRUE_LABELS for Ammonia_10000, CO_1000, CO_4000
// Fault:     check TRUE_LABELS for Sensor_Fault
// Safe:      everything else

void loop() {
  static int pick = 0;

  if (pick < N_TEST_VECTORS){
    int test_idx = pick;

    // Reset all LEDs
    allOff();
    delay(1000);

    Serial.println("\n═════════════════════════════════════");
    Serial.printf("TEST [%d/5] | Vector #%d | True: %s\n",
                  pick + 1, test_idx, TRUE_LABELS[test_idx]);
    Serial.println("═════════════════════════════════════");

    float in_scale   = input_tensor->params.scale;
    int   in_zero_pt = input_tensor->params.zero_point;

    for (int i = 0; i < N_FEATURES; i++) {
      // 1. Get the RAW value (which now has 288 features)
      float raw_val = TEST_VECTORS[test_idx][i];
      
      // 2. Standardize it (z = (x - mean) / std_dev)
      // Change SCALER_SCALE to SCALER_STD if that is what your file uses!
      float scaled_val = (raw_val - SCALER_MEAN[i]) / SCALER_STD[i];
      
      // 3. Quantize it for Int8
      int32_t q = (int32_t)(scaled_val / in_scale + in_zero_pt);
      q = q < -128 ? -128 : (q > 127 ? 127 : q);
      
      // 4. Feed it to the model
      input_tensor->data.int8[i] = (int8_t)q;
    }

    TfLiteStatus status = interpreter->Invoke();
    if (status != kTfLiteOk) {
      Serial.println("[ERROR] Inference failed");
    } else {
      float out_scale   = output_tensor->params.scale;
      int   out_zero_pt = output_tensor->params.zero_point;

      float scores[N_CLASSES];
      for (int i = 0; i < N_CLASSES; i++)
        scores[i] = (output_tensor->data.int8[i] - out_zero_pt) * out_scale;

      int top1 = 0;
      for (int i = 1; i < N_CLASSES; i++)
        if (scores[i] > scores[top1]) top1 = i;

      int top2 = (top1 == 0) ? 1 : 0;
      for (int i = 0; i < N_CLASSES; i++)
        if (i != top1 && scores[i] > scores[top2]) top2 = i;

      bool is_fault  = (top1 == FAULT_CLASS_IDX);
      bool is_hazard = isHazardous(top1);

      Serial.printf("Predicted : %s\n",         CLASS_LABELS[top1]);
      Serial.printf("Confidence: %.1f%%\n",      scores[top1] * 100.0f);
      Serial.printf("Runner-up : %s (%.1f%%)\n", CLASS_LABELS[top2], scores[top2] * 100.0f);

      if (is_fault) {
        setOutputs(false, false, true, false);
        Serial.println("ACTION    : SENSOR FAULT — alarm suppressed");
        Serial.println("LEDs      : [ ] RED  [ ] GREEN  [Y] YELLOW");
      } else if (is_hazard) {
        setOutputs(true, false, false, true);
        Serial.printf("ACTION    : HAZARDOUS GAS (%s) — EVACUATE!\n", CLASS_LABELS[top1]);
        Serial.println("LEDs      : [R] RED  [ ] GREEN  [ ] YELLOW  [!] BUZZER");
      } else {
        setOutputs(false, true, false, false);
        Serial.printf("ACTION    : Safe (%s) — monitoring\n", CLASS_LABELS[top1]);
        Serial.println("LEDs      : [ ] RED  [G] GREEN  [ ] YELLOW");
      }

      pushThingSpeak(top1, scores[top1], is_fault, scores[top2], top2);
    }

    pick++;
    delay(15000);  // 15s gap for ThingSpeak free tier

  } else {
    // All 5 done — heartbeat only
    allOff();
    digitalWrite(PIN_LED_BLUE, HIGH); delay(300);
    digitalWrite(PIN_LED_BLUE, LOW);  delay(300);
  }
}