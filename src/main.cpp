#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <math.h>

// ---- TFLite Micro (TensorFlowLite_ESP32) ----
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ---- Model header (you provide) ----
#include "model_data_135020744.h"  // must define: gesture_model_int8_APOGEE_tflite + gesture_model_int8_APOGEE_tflite_len

// ---- Wi-Fi Wokwi ----
static const char* WIFI_SSID = "Wokwi-GUEST";
static const char* WIFI_PASS = "";

// ---- MQTT ----
static const char* MQTT_HOST = "test.mosquitto.org"; // ca marche bien maintenant
static const uint16_t MQTT_PORT = 1883;
static const char* MQTT_TOPIC = "tinyml/tp2/135020744/acc";  // replace APOGEE with your ID

// ---- LEDs (class 0..3) ----
static const int LED0 = 23;
static const int LED1 = 21;
static const int LED2 = 19;
static const int LED3 = 18;

// ---- Model window: 64 samples, 3 axes ----
static constexpr int N = 64;
static constexpr int D = 3;
static constexpr int kFrameSize = N * D;  // 192

// Ring buffer for raw accel
static float ring_f[kFrameSize];
static int ring_idx = 0;       // next write position (0..63)
static bool window_full = false;

// ---- Quantization (read from model tensor params) ----
static float INPUT_SCALE = 1.0f;
static int INPUT_ZERO_POINT = 0;

// ---- TFLM globals ----
static tflite::AllOpsResolver resolver;
static const tflite::Model* model = nullptr;

static tflite::MicroErrorReporter micro_error_reporter;
static tflite::ErrorReporter* error_reporter = &micro_error_reporter;

static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;

// ✅ Tensor arena: allocate dynamically to avoid .dram0.bss overflow
// Start at 60KB. If AllocateTensors fails, increase to 70KB or 80KB.
static constexpr size_t kTensorArenaSize = 60 * 1024;
static uint8_t* tensor_arena = nullptr;

// MQTT
WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);

// ---------------------------------- Paramètre N-temporel ----------------------------------
static const int n_factor = 2; // 1, 2, 3, 4
static int mqtt_count = 0;
static float acc_sum_x = 0, acc_sum_y = 0, acc_sum_z = 0;

// ---------------- Helpers ----------------
static void setOneHotLED(int cls) {
  digitalWrite(LED0, (cls == 0) ? HIGH : LOW);
  digitalWrite(LED1, (cls == 1) ? HIGH : LOW);
  digitalWrite(LED2, (cls == 2) ? HIGH : LOW);
  digitalWrite(LED3, (cls == 3) ? HIGH : LOW);
}

static int argmax_tensor(const TfLiteTensor* out) {
  int best_i = 0;
  int n = out->dims->data[out->dims->size - 1];

  if (out->type == kTfLiteInt8) {
    int8_t best = out->data.int8[0];
    for (int i = 1; i < n; i++) {
      int8_t v = out->data.int8[i];
      if (v > best) { best = v; best_i = i; }
    }
    return best_i;
  }

  if (out->type == kTfLiteUInt8) {
    uint8_t best = out->data.uint8[0];
    for (int i = 1; i < n; i++) {
      uint8_t v = out->data.uint8[i];
      if (v > best) { best = v; best_i = i; }
    }
    return best_i;
  }

  float best = out->data.f[0];
  for (int i = 1; i < n; i++) {
    float v = out->data.f[i];
    if (v > best) { best = v; best_i = i; }
  }
  return best_i;
}

static int8_t quantize(float x) {
  int q = (int)lroundf(x / INPUT_SCALE) + INPUT_ZERO_POINT;
  if (q < -128) q = -128;
  if (q > 127) q = 127;
  return (int8_t)q;
}

static void wifiConnect() {
  Serial.print("[WiFi] Connecting to ");
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("[WiFi] OK IP=");
  Serial.println(WiFi.localIP());
}

static void mqttConnect() {
  mqttClient.setServer(MQTT_HOST, MQTT_PORT);

  while (!mqttClient.connected()) {
    String clientId = String("wokwi-esp32-") + String((uint32_t)ESP.getEfuseMac(), HEX);
    Serial.print("[MQTT] Connecting as ");
    Serial.println(clientId);

    if (mqttClient.connect(clientId.c_str())) {
      Serial.println("[MQTT] OK");
      mqttClient.subscribe(MQTT_TOPIC);
      Serial.print("[MQTT] Subscribed: ");
      Serial.println(MQTT_TOPIC);
    } else {
      Serial.print("[MQTT] Failed rc=");
      Serial.println(mqttClient.state());
      delay(1000);
    }
  }
}

static void tflmSetup() {
  // ✅ Use the symbol from model_data_APOGEE.h
  model = tflite::GetModel(gesture_model_int8_135020744_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("[TFLM] Schema version mismatch");
    while (true) delay(100);
  }

  // ✅ Allocate arena dynamically (prefer PSRAM if present)
  tensor_arena = (uint8_t*)ps_malloc(kTensorArenaSize);
  if (!tensor_arena) {
    Serial.println("[TFLM] PSRAM alloc failed, trying malloc...");
    tensor_arena = (uint8_t*)malloc(kTensorArenaSize);
  }
  if (!tensor_arena) {
    Serial.println("[TFLM] ERROR: tensor_arena alloc failed");
    while (true) delay(100);
  }

  interpreter = new tflite::MicroInterpreter(
    model,
    resolver,
    tensor_arena,
    kTensorArenaSize,
    error_reporter,
    /*resource_vars=*/nullptr,
    /*profiler=*/nullptr
  );

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("[TFLM] AllocateTensors failed -> increase kTensorArenaSize (e.g. 70KB/80KB)");
    while (true) delay(100);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("[TFLM] Model loaded");
  Serial.print("[TFLM] Input type="); Serial.println(input->type);
  Serial.print("[TFLM] Input bytes="); Serial.println(input->bytes);
  Serial.print("[TFLM] Output type="); Serial.println(output->type);

  // Must match TP requirement: int8 model
  if (input->type != kTfLiteInt8) {
    Serial.println("[TFLM] ERROR: expected int8 input model");
    while (true) delay(100);
  }
  if (input->bytes < (size_t)kFrameSize) {
    Serial.println("[TFLM] ERROR: input tensor too small for 64x3");
    while (true) delay(100);
  }

  INPUT_SCALE = input->params.scale;
  INPUT_ZERO_POINT = input->params.zero_point;

  Serial.print("[TFLM] Input scale="); Serial.println(INPUT_SCALE, 8);
  Serial.print("[TFLM] Input zero_point="); Serial.println(INPUT_ZERO_POINT);
}

static void push_sample(float x, float y, float z) {
  ring_f[ring_idx * 3 + 0] = x;
  ring_f[ring_idx * 3 + 1] = y;
  ring_f[ring_idx * 3 + 2] = z;

  ring_idx++;
  if (ring_idx >= N) {
    ring_idx = 0;
    window_full = true;
  }
}

static void run_inference_if_ready() {
  if (!window_full) return;

  // Z-score on the 64-sample window per axis (Part 1 style)
  float mean[3] = {0, 0, 0};
  float var[3]  = {0, 0, 0};

  // chronological order: oldest sample is ring_idx
  for (int i = 0; i < N; i++) {
    int idx = (ring_idx + i) % N;
    mean[0] += ring_f[idx * 3 + 0];
    mean[1] += ring_f[idx * 3 + 1];
    mean[2] += ring_f[idx * 3 + 2];
  }
  mean[0] /= N; mean[1] /= N; mean[2] /= N;

  for (int i = 0; i < N; i++) {
    int idx = (ring_idx + i) % N;
    float x = ring_f[idx * 3 + 0] - mean[0];
    float y = ring_f[idx * 3 + 1] - mean[1];
    float z = ring_f[idx * 3 + 2] - mean[2];
    var[0] += x * x;
    var[1] += y * y;
    var[2] += z * z;
  }
  var[0] /= N; var[1] /= N; var[2] /= N;

  float stdv[3] = {
    sqrtf(var[0]) + 1e-6f,
    sqrtf(var[1]) + 1e-6f,
    sqrtf(var[2]) + 1e-6f
  };

  // Fill model input: 64x3 -> 192 int8
  int out_i = 0;
  for (int i = 0; i < N; i++) {
    int idx = (ring_idx + i) % N;

    float x = (ring_f[idx * 3 + 0] - mean[0]) / stdv[0];
    float y = (ring_f[idx * 3 + 1] - mean[1]) / stdv[1];
    float z = (ring_f[idx * 3 + 2] - mean[2]) / stdv[2];

    input->data.int8[out_i++] = quantize(x);
    input->data.int8[out_i++] = quantize(y);
    input->data.int8[out_i++] = quantize(z);
  }

  uint32_t t0 = millis();
  TfLiteStatus st = interpreter->Invoke();
  uint32_t t1 = millis();

  if (st != kTfLiteOk) {
    Serial.println("[TFLM] Invoke failed");
    return;
  }

  int cls = argmax_tensor(output);
  setOneHotLED(cls);

  Serial.print("[PRED] class=");
  Serial.print(cls);
  Serial.print(" | dt=");
  Serial.print(t1 - t0);
  Serial.println(" ms");
}

static void mqttCallback(char* topic, byte* payload, unsigned int length) {
  StaticJsonDocument<128> doc;
  DeserializationError err = deserializeJson(doc, payload, length);
  if (err) {
    Serial.print("[RX] JSON parse error: ");
    Serial.println(err.f_str());
    return;
  }

  if (!doc.containsKey("x") || !doc.containsKey("y") || !doc.containsKey("z")) {
    Serial.println("[RX] Missing keys x/y/z");
    return;
  }
// 1. Accumilation des donnes
  acc_sum_x += doc["x"].as<float>();
  acc_sum_y += doc["y"].as<float>();
  acc_sum_z += doc["z"].as<float>();
  mqtt_count++;

  // float x = doc["x"].as<float>();
  // float y = doc["y"].as<float>();
  // float z = doc["z"].as<float>();

  // 2. une fois quee on a n echantillons, on calcule la moyenne et on pousse au modèle
  if (mqtt_count >= n_factor) {
    float avg_x = acc_sum_x / n_factor;
    float avg_y = acc_sum_y / n_factor;
    float avg_z = acc_sum_z / n_factor;

    push_sample(avg_x, avg_y, avg_z);
    run_inference_if_ready();

    // Réinitialisation pour le prochain groupe de n
    acc_sum_x = 0; acc_sum_y = 0; acc_sum_z = 0;
    mqtt_count = 0;
  }

  // push_sample(x, y, z);
  // run_inference_if_ready();
}

// ---------------- Arduino entry points ----------------
void setup() {
  Serial.begin(115200);

  pinMode(LED0, OUTPUT);
  pinMode(LED1, OUTPUT);
  pinMode(LED2, OUTPUT);
  pinMode(LED3, OUTPUT);
  setOneHotLED(-1);

  Serial.println("[SYS] Booting...");

  wifiConnect();

  mqttClient.setCallback(mqttCallback);
  mqttConnect();

  tflmSetup();

  Serial.print("[SYS] Ready, waiting MQTT JSON on: ");
  Serial.println(MQTT_TOPIC);
}

void loop() {
  if (!mqttClient.connected()) mqttConnect();
  mqttClient.loop();
}
