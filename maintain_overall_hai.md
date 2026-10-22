# 🛡️ ASTRA MAINTAIN
### AI-Powered Predictive Maintenance & Coupled Physics Digital Twin for Indian Defence Assets

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?logo=pytorch&logoColor=white)
![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react&logoColor=black)
![Three.js](https://img.shields.io/badge/Three.js-R3F-black?logo=threedotjs&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4+-38B2AC?logo=tailwind-css&logoColor=white)
![Status](https://img.shields.io/badge/Phase-Phase%201%20Active-success)

> **ASTRA MAINTAIN** is a closed-loop **Coupled Physics Digital Twin**, **Temporal Deep Learning RUL (Remaining Useful Life) Predictor**, and **Prescriptive Maintenance Decision Engine** built to predict mechanical and electrical degradation in mission-critical Indian defence assets before they fail in the field.

---

## 📖 Table of Contents
1. [The Problem: Why Traditional Defence Maintenance Fails](#-1-the-problem-why-traditional-defence-maintenance-fails)
2. [What is ASTRA MAINTAIN? (Explained Simply)](#-2-what-is-astra-maintain-explained-simply)
3. [End-to-End System Workflow & Architecture](#-3-end-to-end-system-workflow--architecture)
4. [Target Indian Defence Equipment Profiles & Environments](#-4-target-indian-defence-equipment-profiles--environments)
5. [The 7 Coupled Telemetry Sensors & Physics Engine](#-5-the-7-coupled-telemetry-sensors--physics-engine)
6. [AI & Deep Learning Architecture (TCN + Bi-GRU + Attention)](#-6-ai--deep-learning-architecture-tcn--bi-gru--attention)
7. [Complete Repository Folder Structure](#-7-complete-repository-folder-structure)
8. [API Contract & Sample JSON Payloads](#-8-api-contract--sample-json-payloads)
9. [The 3-Phase Master Roadmap](#-9-the-3-phase-master-roadmap)
10. [Research Sources, Datasets & iDEX Anchors](#-10-research-sources-datasets--idex-anchors)
11. [Installation & Local Setup Guide](#-11-installation--local-setup-guide)
12. [Team Ownership & Contributors](#-12-team-ownership--contributors)

---

## 🚨 1. The Problem: Why Traditional Defence Maintenance Fails

In military aviation, naval operations, and autonomous UAV squadrons, keeping machines mission-ready runs into a classic two-way trap:

### 1. Reactive Maintenance ("Fix It Too Late")
Waiting for a cockpit warning light to flash or a single sensor to cross a static red-line threshold. By the time an engine crosses a critical overheating or vibration limit, physical damage has already occurred—leading to aborted sorties, grounded aircraft, or catastrophic in-flight failure.

### 2. Preventive / Time-Based Maintenance ("Fix It Too Early")
Replacing expensive components on a rigid timer (e.g., every 300 or 500 flight hours) regardless of actual wear.
* **The 89% vs. 11% Reality:** Foundational aerospace reliability studies (U.S. DoD CBM+ and Nowlan & Heap RCM benchmarks) prove that **only 11% of aircraft components fail purely due to age**, while **89% fail due to complex operational stresses**.
* **The Cost:** Maintenance, Repair, and Overhaul (MRO) accounts for **15%–20%** of direct operating costs and **60%–70%** of the total lifecycle cost of a military weapon system. Fixed timers waste crores of rupees in healthy spare parts and lock mission-capable aircraft inside hangars.

### 3. Why Fixed Timers Break Down in Indian Operational Theatres
Two identical engines never degrade at the same speed in India:
* An aircraft or UAV starting up at **Leh Airbase (3,256m altitude, -25°C)** faces 35% thinner air, thickened lubricants, battery voltage sag, and massive cold-start torque loads.
* The exact same engine flying out of **Jaisalmer / Thar Desert (45°C+)** ingests fine silica dust that erodes compressor blades and clogs cooling channels.
* An engine operating out of **Kochi / INS Hansa** breathes humid, salt-laden maritime air that accelerates hot-section sulfidation corrosion and electrical insulation leakage.

### 4. What Happens Physically Inside the Machine (The Chain Reaction)
Failures do not happen out of nowhere—they start as coupled micro-trends:
1. A microscopic crack (**spall**) forms on a high-speed turbine bearing race.
2. Mechanical friction increases $\rightarrow$ **Vibration ($V_{\text{rms}}$) creeps up**.
3. Extra friction generates localized heat $\rightarrow$ **Exhaust Gas Temperature ($T_{\text{EGT}}$) rises**.
4. Rotor alignment shifts slightly, causing seals to leak $\rightarrow$ **Discharge Pressure ($P_3$) drops**.
5. The starter motor or core has to fight higher drag to maintain **RPM**, drawing **+20% more Current ($I$)** and dragging down **Bus Voltage ($V_{\text{bus}}$)**.

Standard threshold alarms miss this chain reaction until the bearing seizes. **ASTRA MAINTAIN catches the correlation across all sensors hours before failure.**

---

## 💡 2. What is ASTRA MAINTAIN? (Explained Simply)

Since a student engineering team does not have a ₹50-crore military jet engine sitting in the lab to break on purpose, **ASTRA MAINTAIN** builds three connected software systems:

### Part A: The "Video Game Clone" (Coupled Physics Digital Twin)
A living, physics-consistent software simulation of the asset and its subsystems (Compressor, Combustor, Turbine Bearings, Gearbox, Starter-Generator). Instead of plotting random numbers, all 7 sensor streams are **mathematically coupled**. If you inject a bearing fault or desert dust fouling, the physics engine ripples that wear across vibration, temperature, pressure, RPM, current, and voltage in real time.

### Part B: The "Time-Traveling Doctor" (Temporal AI Predictor)
A normal thermometer only beeps *after* you hit a 100°F fever. Our Temporal Deep Learning model watches a **40-tick rolling window** of all 7 sensors simultaneously, learning the *slope and multi-sensor shape* of degradation. It outputs:
1. **Asset Health Index (HI):** `0% – 100%` (e.g., `Engine Health: 72%`)
2. **Failure Risk Probability:** `0% – 100%` (e.g., `Failure Risk: 78%`)
3. **Bounded Remaining Useful Life (RUL):** Instead of guessing a fake single decimal like `5.32 hours`, it uses Quantile Regression to output an honest uncertainty window: **`Estimated RUL: 4–7 hours`**.

### Part C: The "Squadron Commander's Brain" (Prescriptive Maintenance Engine)
Predicting a number is useless if the ground crew doesn't know *what* to fix. Our decision engine reads the AI's sensor attribution weights and generates plain-English maintenance cards:
> **`ALERT: HIGH PRIORITY`**  
> *Observed Trend: Rising vibration (+38%) + Rising EGT (+34%) + Declining pressure (-28%).*  
> *Action: Inspect High-Pressure Turbine Bearing & Stage-1 Seal before next flight window.*

---

## ⚙️ 3. End-to-End System Workflow & Architecture

```text
+-----------------------------------------------------------------------------------+
|                        INDIAN DEFENCE ASSET & THEATRE CONFIG                      |
|   [Asset: Fighter GTSE | Do-228 | MALE UAV]  x  [Theatre: Leh | Thar | Kochi]     |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 1. COUPLED PHYSICS DIGITAL TWIN ENGINE (Python / NumPy / SciPy)                   |
|                                                                                   |
|  Hidden Component Health Vector: H(t) = [h_comp, h_turb, h_bearing, h_elec]       |
|  <-- Live Operator Fault Injection (Bearing Spall / HPC Sand Fouling / Voltage)   |
|                                                                                   |
|  Outputs 7 Coupled Channels @ 1Hz-10Hz + Exact Simulated Ground-Truth RUL_true    |
|  [Temp (C), Pressure (kPa), Vibration (mm/s), RPM, Voltage (V), Current (A), Load]|
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 2. TEMPORAL PREPROCESSOR & SLIDING WINDOW BUFFER                                  |
|                                                                                   |
|  Rolling Buffer (W = 40 ticks) -> Load-Normalized Z-Score                         |
|  Feature Extraction per Sensor: [Raw_Norm, Rolling_Mean_10, Slope_dX/dt]          |
|  Produces Tensor Shape: (Batch, 40, 21)                                           |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 3. MULTI-HEAD TEMPORAL AI ENGINE (PyTorch TCN / Bi-GRU + Self-Attention)          |
|  (With Deterministic EWMA + Mahalanobis Fallback if GPU/Weights Unavailable)      |
|                                                                                   |
|  +--> Head 1: Health Index (%) & Failure Risk Probability (%)                     |
|  +--> Head 2: Quantile RUL Bounds [q_10 (Low), q_50 (Median), q_90 (High)]        |
|  +--> Head 3: Sensor Attention Weights (Root-Cause Attribution %)                 |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 4. PRESCRIPTIVE DECISION ENGINE & FASTAPI STREAMING SERVER                        |
|                                                                                   |
|  Maps Top Degrading Sensors -> Actionable Maintenance Work Order & Priority Score |
|  Streams Telemetry + AI Predictions + Ground-Truth Validation via REST/WebSockets |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 5. REACT + TAILWIND + THREE.JS OPERATOR COMMAND CONSOLE                           |
|                                                                                   |
|  - Live 7-Sensor Synchronized Telemetry Charts                                    |
|  - Interactive Chaos / Fault-Injection Sandbox Panel                              |
|  - True Simulated RUL vs. AI Predicted RUL Convergence Fan Chart                  |
|  - 3D Subsystem X-Ray Twin & Plain-English Maintenance Alert Cards                |
+-----------------------------------------------------------------------------------+