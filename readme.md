---

# Edge AI Gas Classifier with Sensor Fault Detection — Complete System Documentation

---

## System Overview

This project implements a TinyML gas classification system on an ESP32 microcontroller (simulated via Wokwi) that classifies 11 industrial chemicals and detects sensor hardware faults as a 12th class. The core innovation is distinguishing a real hazardous gas event from a broken sensor — preventing false alarms and unnecessary evacuations.

The pipeline has three stages: **training** (Python/TensorFlow), **test vector generation** (Python), and **edge inference** (ESP32/TFLite Micro via Wokwi simulation).

---

## Stage 1 — Model Training (`train_1.py`)

### Dataset Preparation
The base dataset is `chemicals_in_wind_tunnel.csv` containing MOx (Metal Oxide) sensor readings across 288 sensor feature columns for 11 chemical classes. The smallest class (CO_1000) has 450 samples.

A 12th synthetic class `Sensor_Fault` is created by:
- Selecting `per_class = FAULT_SAMPLE_COUNT / 11` rows from each of the 11 existing classes
- Corrupting 2–50% of their sensor columns with extreme values — either very low (0.00–0.05) or very high (0.92–0.99) using uniform random draws
- The number of broken sensors per row follows an exponential distribution (scale=10, minimum 2), capped at 50% of total sensors
- These corrupted rows are relabelled as `Sensor_Fault`, producing a final dataset of 12 classes saved as `chemicals_in_wind_tunnel_fault.csv`

### Preprocessing
- Features and labels are separated; labels are integer-encoded alphabetically using `LabelEncoder` (producing indices 0–11)
- A stratified 80/20 train/test split is applied to preserve class proportions
- `StandardScaler` is fit **only on training data** and applied to both train and test sets — this prevents data leakage
- The scaler's mean and standard deviation arrays are exported as `scaler_params.h` for use on the ESP32
- Class weights are computed using `compute_class_weight='balanced'` to handle class imbalance during training

### Model Architecture
```
Input (288 features)
    ↓
Dense(64, relu)
BatchNormalization
Dropout(0.3)
    ↓
Dense(32, relu)
BatchNormalization
Dropout(0.2)
    ↓
Dense(16, relu)
    ↓
Dense(12, softmax)
```
Trained with Adam optimizer, categorical crossentropy loss, EarlyStopping (patience=10) on validation loss.

### TFLite Conversion
The trained Keras model is converted to TFLite with **full int8 quantization**:
- A representative dataset of 200 random training samples is used to calibrate quantization ranges
- Both input and output types are set to `int8`
- The quantized model is exported as a C byte array in `gas_model.h` with `alignas(8)` and `PROGMEM` for ESP32 flash storage

### Output Header Files
| File | Contents |
|------|----------|
| `gas_model.h` | TFLite model as `gas_model_data[]` byte array (174KB) |
| `scaler_params.h` | `SCALER_MEAN[288]` and `SCALER_STD[288]` float arrays |
| `labels.h` | `CLASS_LABELS[12]` string array, `N_CLASSES=12` |

### Label Index Mapping
| Index | Class | Category |
|-------|-------|----------|
| 0 | Acetaldehyde_500 | Safe |
| 1 | Acetone_2500 | Safe |
| 2 | Ammonia_10000 | **Hazardous** |
| 3 | Benzene_200 | Safe |
| 4 | Butanol_100 | Safe |
| 5 | CO_1000 | **Hazardous** |
| 6 | CO_4000 | **Hazardous** |
| 7 | Ethylene_500 | Safe |
| 8 | Methane_1000 | Safe |
| 9 | Methanol_200 | Safe |
| 10 | Sensor_Fault | **Fault** |
| 11 | Toluene_200 | Safe |

---

## Stage 2 — Test Vector Generation (`test_vector_generator.py`)

This script generates a C header file containing 5 stratified raw sensor samples from the fault-injected CSV for use in the ESP32 simulation.

### Process
1. Loads `chemicals_in_wind_tunnel_fault.csv` and drops the index column
2. Separates features (`X`) and labels (`y`)
3. Buckets all row indices into three categories:
   - **Fault**: rows where `y == 'Sensor_Fault'`
   - **Hazardous**: rows where `y` is one of `Ammonia_10000`, `CO_1000`, `CO_4000`
   - **Safe**: all remaining rows
4. Performs stratified random sampling using `numpy.random.default_rng()`:
   - 2 samples from Fault pool
   - 1 sample from Hazardous pool
   - 2 samples from Safe pool
5. Shuffles the 5 selected samples to randomize presentation order
6. Exports **raw unscaled** values as a C float array to `include/wokwi_test_vectors.h`

### Key Design Decision
The vectors are stored **raw** (unscaled). Scaling happens inside the ESP32 firmware using `scaler_params.h`. This mirrors exactly what would happen in a real deployment where raw sensor readings arrive and must be preprocessed on-device.

### Output — `wokwi_test_vectors.h`
```cpp
#define N_TEST_VECTORS 5
#define N_FEATURES     288

static const char* TRUE_LABELS[5] = { ... };
static const float TEST_VECTORS[5][288] PROGMEM = { ... };
```

---

## Stage 3 — ESP32 Edge Inference (`main.cpp`)

### Hardware Configuration (`diagram.json`)
The simulated circuit consists of an ESP32 DevKit V1 with 5 output components, each connected through a 220Ω current-limiting resistor:

| Pin | Component | Color | Purpose |
|-----|-----------|-------|---------|
| D18 | LED | Red | Hazardous gas detected |
| D19 | LED | Green | Safe gas — normal monitoring |
| D21 | LED | Yellow | Sensor fault — maintenance required |
| D22 | Buzzer | — | Audible alert for hazardous gas only |
| D23 | LED | Blue | Heartbeat — confirms setup complete |

All LED cathodes and buzzer ground connect to ESP32 GND.

### Firmware Walkthrough

#### Global State
```cpp
const int TENSOR_ARENA_SIZE = 70 * 1024;
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
```
A 70KB static buffer allocated in RAM serves as the TFLite Micro working memory for tensor allocation, intermediate activations, and scratch space during inference.

```cpp
static tflite::MicroErrorReporter error_reporter;
static tflite::AllOpsResolver     resolver;
const  tflite::Model*             tflite_model = nullptr;
static tflite::MicroInterpreter*  interpreter  = nullptr;
TfLiteTensor* input_tensor  = nullptr;
TfLiteTensor* output_tensor = nullptr;
```
TFLite Micro objects are declared statically to avoid heap fragmentation. `AllOpsResolver` registers all supported ops — the model only uses FullyConnected, Softmax, BatchNorm, and Quantize ops at runtime.

---

#### `isHazardous(int idx)`
Iterates over the 3 hazardous class indices `{2, 5, 6}` and returns true if the predicted index matches any of them. Used to branch between the three output states.

---

#### `setOutputs(bool red, bool green, bool yellow, bool buzzer)`
Writes HIGH or LOW to all four output pins in a single call. Ensures consistent LED state — no pin is accidentally left on from a previous inference cycle.

---

#### `allOff()`
Convenience wrapper that calls `setOutputs(false, false, false, false)`. Used at startup, between test vectors, and inside `setupFailed()`.

---

#### `setupFailed()`
Called when any critical setup step fails (schema mismatch or tensor allocation failure). Turns all three indicator LEDs on simultaneously as a visual error code and halts execution with `while(1)`. The blue heartbeat LED deliberately stays off — its absence distinguishes a setup failure from normal operation.

---

#### `pushThingSpeak(...)`
Checks WiFi connection status first. If connected, constructs a GET request URL with 5 query parameters and sends it to `api.thingspeak.com/update`:
- `field1` — predicted class index
- `field2` — top-1 confidence (4 decimal places)
- `field3` — fault flag (1 if Sensor_Fault, 0 otherwise)
- `field4` — runner-up confidence
- `field5` — runner-up class index

Prints the HTTP response code and entry ID to serial. ThingSpeak free tier enforces a minimum 15-second interval between updates — hence the `delay(15000)` in `loop()`.

---

#### `runInference(float* raw_features)`
Full inference pipeline for live serial input mode (currently commented out in `loop()` but preserved for future use):

1. Retrieves input tensor quantization parameters (`scale`, `zero_point`)
2. For each of 288 features: applies StandardScaler normalization using `SCALER_MEAN[i]` and `SCALER_STD[i]`, then quantizes to int8 using `q = (int32_t)(scaled / in_scale + in_zero_pt)`, clamped to [-128, 127]
3. Calls `interpreter->Invoke()`
4. Dequantizes output scores: `scores[i] = (int8_output - zero_point) * scale`
5. Finds top-1 and top-2 predictions by linear scan
6. Branches into fault/hazard/safe logic, drives LEDs, prints result, pushes to ThingSpeak

---

#### `parseAndRun(String& line)`
Parses a comma-separated string of 288 float values received over Serial into a float array and calls `runInference()`. Designed for the live serial input mode where a Python sender streams CSV rows over TCP. Currently preserved but not active in the test vector demo.

---

#### `setup()`
Executes once on boot in this order:

1. Sets Serial RX buffer to 4096 bytes (accommodates ~2900 bytes for 288 floats at ~10 chars each) and begins at 115200 baud
2. Configures all 5 pins as OUTPUT
3. Runs a **startup blink** — all three indicator LEDs flash 3 times simultaneously as visual confirmation that the firmware has started and GPIO is working
4. Loads the TFLite model from `gas_model_data[]` in flash using `tflite::GetModel()` and validates the flatbuffer schema version — calls `setupFailed()` on mismatch
5. Instantiates `MicroInterpreter` with the model, resolver, tensor arena, and error reporter — calls `setupFailed()` if `AllocateTensors()` fails
6. Retrieves pointers to input and output tensors and prints their dimensions to serial
7. Attempts WiFi connection to `Wokwi-GUEST` (open network provided by Wokwi simulator) with 20 retries at 500ms intervals — ThingSpeak posts are skipped gracefully if this fails
8. Prints `[READY]` to serial and turns the Yellow LED on as a steady ready indicator

---

#### `loop()`
The main inference loop iterates through all `N_TEST_VECTORS` (5) test vectors sequentially:

**For each vector:**
1. Calls `allOff()` and waits 1 second — clears previous LED state visually
2. Prints a header with vector index and true label to serial
3. Retrieves input tensor quantization parameters
4. For each of 288 features:
   - Reads raw value from `TEST_VECTORS[test_idx][i]` stored in flash
   - Applies StandardScaler: `scaled = (raw - SCALER_MEAN[i]) / SCALER_STD[i]`
   - Quantizes to int8: `q = (int32_t)(scaled / in_scale + in_zero_pt)`, clamped to [-128, 127]
   - Writes to `input_tensor->data.int8[i]`
5. Calls `interpreter->Invoke()` to run inference
6. Dequantizes all 12 output scores
7. Finds top-1 and top-2 by linear scan
8. Branches on outcome:

| Outcome | Red | Green | Yellow | Buzzer | Serial Action |
|---------|-----|-------|--------|--------|---------------|
| Sensor_Fault | OFF | OFF | ON | OFF | SENSOR FAULT — alarm suppressed |
| Hazardous gas | ON | OFF | OFF | ON | HAZARDOUS GAS — EVACUATE! |
| Safe gas | OFF | ON | OFF | OFF | Safe — monitoring |

9. Calls `pushThingSpeak()` to log to cloud
10. Increments `pick`, waits 15 seconds before next vector

**After all 5 vectors:** enters idle mode — all LEDs off, blue LED blinks continuously as a heartbeat confirming the firmware is still running.

---

## Complete Data Flow

```
chemicals_in_wind_tunnel.csv
        ↓
train_1.py
  → Fault injection → 12-class dataset
  → StandardScaler fit on train set
  → ANN training (64→32→16→12)
  → TFLite int8 quantization
  → gas_model.h, scaler_params.h, labels.h
        ↓
test_vector_generator.py
  → Stratified sampling (2 fault, 1 hazard, 2 safe)
  → Raw values exported
  → wokwi_test_vectors.h
        ↓
ESP32 (Wokwi Simulation)
  → Boot, load model, connect WiFi
  → For each of 5 test vectors:
       Raw value → StandardScaler → int8 quantize
       → TFLite inference → 12 softmax scores
       → Top-1 prediction → LED/buzzer output
       → ThingSpeak cloud log
```

---

## ThingSpeak Cloud Integration

Each inference result is pushed to ThingSpeak channel `3330882` with a 15-second minimum interval enforced by the free tier. The 5 fields allow post-hoc analysis of prediction confidence, fault detection rate, and runner-up class distribution directly from the ThingSpeak dashboard.

## PlatformIO Build Configuration (`platformio.ini`)

The firmware is built using PlatformIO, a professional embedded development framework that manages toolchains, libraries, and build configurations. The project targets the `esp32dev` environment with the following settings:

**Platform and Board:** `espressif32` platform with `esp32dev` board definition — this maps to the ESP32 DevKit V1 which matches the Wokwi simulation target. The framework is Arduino, providing the familiar `setup()`/`loop()` programming model on top of ESP-IDF.

**TFLite Library:** The TensorFlow Lite for Microcontrollers library is sourced directly from `tanakamasayuki/Arduino_TensorFlowLite_ESP32` on GitHub. This is a community-maintained port of TFLite Micro specifically optimized for the ESP32's Xtensa LX6 architecture, providing the `MicroInterpreter`, `AllOpsResolver`, and int8 quantization support used throughout the firmware.

**Build Flags:** Two compiler flags are applied — `-DCORE_DEBUG_LEVEL=0` suppresses all ESP-IDF internal debug output over serial, keeping the serial monitor clean and showing only application-level messages. `-O2` enables level-2 compiler optimization, reducing inference latency and binary size by allowing the compiler to perform instruction reordering, loop unrolling, and inlining — important for a compute-heavy TFLite workload on a microcontroller.

**Monitor Speed:** Set to 115200 baud, matching the `Serial.begin(115200)` call in firmware, ensuring the serial monitor correctly decodes inference results and debug messages without framing errors.