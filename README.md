# Watch Gesture Detection — TinyML on ESP32

> **On-device gesture recognition** for a simulated smartwatch using a quantized neural network, IMU sensor data, and MQTT-based IoT communication.

---

## 🧠 What This Project Does

This project implements a complete **embedded AI pipeline** that detects wrist/watch gestures in real time — entirely on a microcontroller, with no cloud inference needed.

A quantized TinyML model runs directly on an **ESP32** (simulated via Wokwi), reading accelerometer/gyroscope sensor data and classifying movement gestures. Detected gestures are published over an **MQTT broker**, enabling integration with other IoT systems.

```
IMU Sensor Data (accel/gyro)
        │
        ▼
   ESP32 (Wokwi)
        │
   TinyML Inference  ◄── Quantized Neural Network (int8)
        │
        ▼
   Gesture Label
        │
        ▼
   MQTT Broker  ──► Any subscriber (dashboard, actuator, app...)
```

---

## 🏗️ Architecture & Technical Stack

| Component | Technology |
|-----------|------------|
| Microcontroller | ESP32 (simulated on Wokwi) |
| Development framework | PlatformIO |
| Inference engine | TensorFlow Lite Micro |
| Model optimization | INT8 quantization |
| Sensor interface | IMU (accelerometer + gyroscope) |
| Communication | MQTT (IoT publish/subscribe) |
| Language | C / C++ |

---

## 📁 Project Structure

```
watch-gestures-detection/
├── src/              # Main firmware source code (C/C++)
├── include/          # Header files
├── lib/              # External libraries (TFLite Micro, MQTT client)
├── test/             # Unit tests
├── diagram.json      # Wokwi circuit diagram (ESP32 + sensors)
├── wokwi.toml        # Wokwi simulation config
└── platformio.ini    # PlatformIO build configuration
```

---

## 🔬 Key Concepts Applied

**TinyML & Model Quantization**
The neural network is converted to TensorFlow Lite format and quantized to INT8, drastically reducing memory footprint to fit within the ESP32's ~512KB RAM. This is the core challenge of embedded AI: making inference work under extreme hardware constraints.

**Edge AI / On-Device Inference**
No data is sent to the cloud for classification. The entire inference pipeline runs locally on the microcontroller — important for latency, privacy, and offline operation.

**MQTT for IoT Integration**
The MQTT protocol is used to send and receive sensor readings and publish gesture predictions, making the system composable with other IoT devices or dashboards.

**Wokwi Hardware Simulation**
The full circuit (ESP32 + IMU sensor wiring) is defined in `diagram.json` and simulated via Wokwi, enabling development and testing without physical hardware.

---

## 🚀 Running the Simulation

### Prerequisites
- [PlatformIO](https://platformio.org/) (VS Code extension or CLI)
- [Wokwi VS Code extension](https://docs.wokwi.com/vscode/getting-started)
- An MQTT broker (e.g., [Mosquitto](https://mosquitto.org/) locally, or test broker `broker.hivemq.com`)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/brahim77777/watch-gestures-detection.git
   cd watch-gestures-detection
   ```

2. Open in VS Code with PlatformIO installed.

3. Build the firmware:
   ```bash
   pio run
   ```

4. Open `diagram.json` with the Wokwi extension to start the simulation.

5. Monitor MQTT output:
   ```bash
   mosquitto_sub -h broker.hivemq.com -t "gestures/#"
   ```

---

## 📊 Gestures Detected

| Gesture | Description |
|---------|-------------|
| *(e.g., shake)* | Rapid lateral movement |
| *(e.g., twist)* | Wrist rotation gesture |
| *(e.g., still)* | No movement / idle state |

> *Update this table with the actual gesture classes from your trained model.*

---

## 💡 What I Learned

- How to convert and quantize a Keras/TensorFlow model to TFLite Micro format
- How to deploy and run inference on a microcontroller using C/C++
- The memory constraints of embedded systems and how quantization (float32 → int8) addresses them
- MQTT protocol architecture and IoT publish/subscribe patterns
- PlatformIO build system and embedded project structure
- Hardware simulation with Wokwi for rapid prototyping

---

## 🔗 Related Work

- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
- [Wokwi ESP32 Simulator](https://wokwi.com)
- [PlatformIO](https://platformio.org/)

---

## 👤 Author

**Brahim Bazi** — Master's student in Embedded Artificial Intelligence  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Brahim_Bazi-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/brahim-bazi-9b49b235a/)
