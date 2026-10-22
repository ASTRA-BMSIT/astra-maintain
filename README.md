[astra_maintain.md](https://github.com/user-attachments/files/33822769/astra_maintain.md)
# ASTRA MAINTAIN
### AI-Powered Predictive Maintenance & Coupled Physics Digital Twin for Indian Defence Assets

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?logo=pytorch&logoColor=white)
![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react&logoColor=black)
![Three.js](https://img.shields.io/badge/Three.js-R3F-black?logo=threedotjs&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4+-38B2AC?logo=tailwind-css&logoColor=white)
![Status](https://img.shields.io/badge/Phase-Phase%201%20Active-success)

> ASTRA MAINTAIN is a closed-loop Coupled Physics Digital Twin, Temporal Deep Learning RUL (Remaining Useful Life) Predictor, and Prescriptive Maintenance Decision Engine built to predict mechanical and electrical degradation in mission-critical Indian defence assets before they fail in the field.

---

## Table of Contents
1. [The Problem: Why Traditional Defence Maintenance Fails](#1-the-problem-why-traditional-defence-maintenance-fails)
2. [What is ASTRA MAINTAIN? (Overview)](#2-what-is-astra-maintain-overview)
3. [End-to-End System Workflow & Architecture](#3-end-to-end-system-workflow--architecture)
4. [Target Indian Defence Equipment Profiles & Environments](#4-target-indian-defence-equipment-profiles--environments)
5. [The 7 Coupled Telemetry Sensors & Physics Engine](#5-the-7-coupled-telemetry-sensors--physics-engine)
6. [AI & Deep Learning Architecture (TCN + Bi-GRU + Attention)](#6-ai--deep-learning-architecture-tcn--bi-gru--attention)
7. [Complete Repository Folder Structure](#7-complete-repository-folder-structure)
8. [API Contract & Sample JSON Payloads](#8-api-contract--sample-json-payloads)
9. [The 3-Phase Master Roadmap](#9-the-3-phase-master-roadmap)
10. [Research Sources, Datasets & iDEX Anchors](#10-research-sources-datasets--idex-anchors)
11. [Installation & Local Setup Guide](#11-installation--local-setup-guide)
12. [Team Ownership & Contributors](#12-team-ownership--contributors)

---

## 1. The Problem: Why Traditional Defence Maintenance Fails

In military aviation, naval operations, and autonomous UAV squadrons, keeping machines mission-ready runs into a classic two-way trap:

### Reactive Maintenance ("Fix It Too Late")
Waiting for a cockpit warning light to flash or a single sensor to cross a static red-line threshold. By the time an engine crosses a critical overheating or vibration limit, physical damage has already occurred, leading to aborted sorties, grounded aircraft, or catastrophic in-flight failure.

### Preventive / Time-Based Maintenance ("Fix It Too Early")
Replacing expensive components on a rigid timer (e.g., every 300 or 500 flight hours) regardless of actual wear.
* The 89% vs. 11% Reality: Foundational aerospace reliability studies (U.S. DoD CBM+ and Nowlan & Heap RCM benchmarks) prove that only 11% of aircraft components fail purely due to age, while 89% fail due to complex operational stresses.
* The Cost: Maintenance, Repair, and Overhaul (MRO) accounts for 15% to 20% of direct operating costs and 60% to 70% of the total lifecycle cost of a military weapon system. Fixed timers waste crores of rupees in healthy spare parts and lock mission-capable aircraft inside hangars.

### Why Fixed Timers Break Down in Indian Operational Theatres
Two identical engines never degrade at the same speed in India:
* An aircraft or UAV starting up at Leh Airbase (3,256m altitude, -25C) faces 35% thinner air, thickened lubricants, battery voltage sag, and massive cold-start torque loads.
* The exact same engine flying out of Jaisalmer or the Thar Desert (45C+) ingests fine silica dust that erodes compressor blades and clogs cooling channels.
* An engine operating out of Kochi or INS Hansa breathes humid, salt-laden maritime air that accelerates hot-section sulfidation corrosion and electrical insulation leakage.

### What Happens Physically Inside the Machine (The Chain Reaction)
Failures do not happen out of nowhere; they start as coupled micro-trends:
1. A microscopic crack (spall) forms on a high-speed turbine bearing race.
2. Mechanical friction increases -> Vibration (V_rms) creeps up.
3. Extra friction generates localized heat -> Exhaust Gas Temperature (EGT) rises.
4. Rotor alignment shifts slightly, causing seals to leak -> Discharge Pressure (P3) drops.
5. The starter motor or core has to fight higher drag to maintain RPM, drawing +20% more Current (I) and dragging down Bus Voltage (V_bus).

Standard threshold alarms miss this chain reaction until the bearing seizes. ASTRA MAINTAIN catches the correlation across all sensors hours before failure.

---

## 2. What is ASTRA MAINTAIN? (Overview)

Since a student engineering team does not have a 50-crore military jet engine sitting in the lab to break on purpose, ASTRA MAINTAIN builds three connected software systems:

### Part A: The "Video Game Clone" (Coupled Physics Digital Twin)
A living, physics-consistent software simulation of the asset and its subsystems (Compressor, Combustor, Turbine Bearings, Gearbox, Starter-Generator). Instead of plotting random numbers, all 7 sensor streams are mathematically coupled. If you inject a bearing fault or desert dust fouling, the physics engine ripples that wear across vibration, temperature, pressure, RPM, current, and voltage in real time.

### Part B: The "Time-Traveling Doctor" (Temporal AI Predictor)
A normal thermometer only beeps after you hit a fever. Our Temporal Deep Learning model watches a 40-tick rolling window of all 7 sensors simultaneously, learning the slope and multi-sensor shape of degradation. It outputs:
1. Asset Health Index (HI): 0% to 100% (e.g., "Engine Health: 72%")
2. Failure Risk Probability: 0% to 100% (e.g., "Failure Risk: 78%")
3. Bounded Remaining Useful Life (RUL): Instead of guessing a fake single decimal like "5.32 hours", it uses Quantile Regression to output an honest uncertainty window: "Estimated RUL: 4 to 7 hours".

### Part C: The "Squadron Commander's Brain" (Prescriptive Maintenance Engine)
Predicting a number is useless if the ground crew does not know what to fix. Our decision engine reads the AI's sensor attribution weights and generates plain-English maintenance cards:
> [ALERT: HIGH PRIORITY] 
> Observed Trend: Rising vibration (+38%) + Rising EGT (+34%) + Declining pressure (-28%). 
> Action: Inspect High-Pressure Turbine Bearing and Stage-1 Seal before next flight window.

---

## 3. End-to-End System Workflow & Architecture

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
```

---

## 4. Target Indian Defence Equipment Profiles & Environments

Our simulator is anchored directly in official iDEX DRISHTI problem statements and supports 4 Indian Defence Equipment Profiles:

| Profile ID | Asset Name | Real-World Indian Context | Primary Failure Modes Simulated |
| :--- | :--- | :--- | :--- |
| GTSE_FIGHTER | Fighter Jet Gas Turbine Starter Engine (GTSE) | Cranks main turbofan spools on LCA Tejas / Su-30MKI from 0 to 40,000+ RPM and >650C in under 45 seconds. Matches iDEX Challenge #2567. | High-speed bearing spall, starter clutch slip, turbine wheel thermal fatigue. |
| DO228_TURBOPROP | HAL Dornier Do-228 Turboprop Engine | Flown by IAF, Indian Navy, and Coast Guard for maritime patrol and transport. Matches iDEX Challenge #2531. | Compressor salt pitting/fouling, reduction gearbox harmonic vibration, oil pressure drop. |
| MALE_UAV | Autonomous Military UAV (TAPAS-BH type) | High-altitude ISR and logistics drones operating along northern borders. | BLDC motor bearing wear, ESC MOSFET thermal runaway, Li-Po/Li-Ion cold voltage sag. |
| RADAR_APU | Mobile Radar & SAM Launcher APU | Continuous field power generators for mobile radar and missile batteries. | Alternator winding insulation breakdown, cooling jacket blockage, fuel pressure drop. |

### Indian Theatre Environmental Modifiers
* LEH_HIGH_ALT (3,200m to 5,000m, -25C): Air density drops by approx 35%. Cold-thickened oil increases startup mechanical drag (+25% current and vibration spike), and battery internal resistance rises (-15% voltage sag).
* THAR_DESERT (45C+, High Silica Dust): Accelerates High-Pressure Compressor (HPC) blade erosion (2.5x faster pressure ratio decay and higher baseline EGT).
* KOCHI_MARITIME (32C, 85%+ Humidity, Salt Fog): Accelerates turbine hot-section corrosion and electrical harness current leakage.

---

## 5. The 7 Coupled Telemetry Sensors & Physics Engine

Every simulation tick generates 7 coupled channels alongside hidden subsystem health states H(t) = [h_comp, h_turb, h_bearing, h_elec] ranging from 0 to 1:

| # | Sensor Channel | Unit | Normal Range (GTSE) | Critical Threshold | Coupled Degradation Behavior |
| :- | :--- | :--- | :--- | :--- | :--- |
| 1 | Temperature (T) | C | 520 - 610 C | > 680 C | Rises (+15% to +30%) as compressor efficiency drops or bearing friction generates heat. |
| 2 | Pressure (P) | kPa | 380 - 620 kPa | < 310 kPa | Declines (-10% to -25%) as compressor blade tip clearance widens or seals leak. |
| 3 | Vibration (V_rms) | mm/s | 1.2 - 3.5 mm/s | > 8.0 mm/s | Spikes quadratically relative to bearing wear and RPM when bearing spall develops. |
| 4 | Rotational Speed | RPM | 28,000 - 42,000 | Hunting +/- 1,500 | Develops micro-jitter and slower spool-up response under degraded fuel/ESC control. |
| 5 | Bus Voltage (V) | V DC | 27.5 - 28.5 V | < 24.5 V | Sags under load as electrical health degrades or current spikes. |
| 6 | Current Draw (I) | Amps | 45 - 95 A | > 140 A | Climbs (+15% to +40%) to overcome rising mechanical bearing drag at target RPM. |
| 7 | Operational Load | % | 0 - 100 % | Context Input | Tells the AI whether high temp/RPM is due to 100% mission throttle or internal wear. |

---

## 6. AI & Deep Learning Architecture (TCN + Bi-GRU + Attention)

### Why We Use TCN (Primary) & Bi-GRU + Self-Attention (Secondary)
1. Temporal Convolutional Network (TCN): Uses causal dilated convolutions and residual connections. Unlike vanilla LSTMs that process ticks sequentially and suffer from vanishing gradients, TCNs process the entire 40-tick window in parallel, capturing both high-frequency vibration spikes and slow thermal creep with fast, stable training.
2. Temporal & Sensor Self-Attention: Defence engineers will never ground an aircraft based on a black-box number. Our attention layer outputs feature attribution weights (e.g., Vibration: 42%, Temperature: 35%, Pressure: 23%) so the operator sees why the RUL dropped.
3. Piecewise-Linear RUL Target Cap (RUL_max = 125): Prevents the model from overfitting to noisy sensor fluctuations when an engine is 100% healthy, focusing network capacity on the active degradation phase.
4. Quantile Regression Loss (Uncertainty Bounds): Instead of Mean Squared Error (MSE), the RUL head is trained with Pinball / Quantile Loss for percentiles (0.10, 0.50, 0.90). This directly gives the operator a reliable interval like "Estimated RUL: 4 to 7 hours".
5. NASA PHM Asymmetric Scoring Metric: In defence, predicting a failure too late is catastrophic, while predicting slightly too early is safe. We benchmark our models using both RMSE and the NASA Asymmetric Scoring Function, which penalizes late predictions exponentially harder than early ones.
6. Zero-Crash Deterministic Fallback: If PyTorch weights are not loaded, the pipeline automatically falls back to an Exponentially Weighted Moving Average (EWMA) + Mahalanobis Distance Trend Extrapolator.

---

## 7. Complete Repository Folder Structure

```text
astra-maintain/
├── README.md                           # Master documentation (this file)
├── docker-compose.yml                  # One-command launch for Backend + Frontend
├── .gitignore
│
├── backend/                            # TRACK 1: FastAPI Server & Physics Digital Twin (Harsh)
│   ├── requirements.txt
│   ├── main.py                         # FastAPI entry point & CORS / WebSocket setup
│   ├── config.py                       # Asset presets (GTSE, Do-228, UAV) & Theatre constants
│   ├── api/
│   │   ├── routes_sim.py               # /sim/init, /sim/step, /sim/inject-fault, /sim/reset
│   │   ├── routes_fleet.py             # Phase 3 squadron readiness & triage endpoints
│   │   └── schemas.py                  # Pydantic request/response models
│   ├── sim_core/
│   │   ├── digital_twin.py             # Coupled physics engine & state degradation math
│   │   ├── profiles.py                 # Fighter GTSE, Do-228, UAV & Radar APU parameters
│   │   ├── environments.py             # Leh (-25C), Thar (+45C dust), Kochi (salt fog) modifiers
│   │   └── fault_injector.py           # Bearing spall, HPC fouling, turbine erosion, elec sag
│   └── decision_engine/
│       ├── recommender.py              # Translates sensor slopes + AI output -> Maintenance Card
│       └── priority_scorer.py          # Computes Maintenance Priority Score (MPS 0-100)
│
├── ai_engine/                          # TRACK 2: PyTorch RUL Models & Data Pipeline (Keshav & Srihari)
│   ├── requirements.txt
│   ├── preprocessor.py                 # 40-tick sliding window, Z-score norm, slope & rolling stats
│   ├── models/
│   │   ├── tcn_quantile.py             # Primary Temporal Convolutional Network with Quantile Head
│   │   ├── bigru_attention.py          # Bi-GRU with Temporal & Sensor Self-Attention
│   │   └── ewma_fallback.py            # Deterministic Mahalanobis + EWMA slope extrapolator
│   ├── training/
│   │   ├── train_cmapss.py             # Training script on NASA C-MAPSS (FD001-FD004)
│   │   ├── train_synthetic.py          # Fine-tuning on coupled Indian Defence Digital Twin logs
│   │   ├── losses.py                   # Pinball Quantile Loss + Asymmetric PHM Loss
│   │   └── evaluate_metrics.py         # RMSE, MAE, NASA Asymmetric Score & coverage plots
│   └── saved_weights/
│       ├── tcn_gtse_v1.pt              # Pre-trained weights for instant live demo inference
│       └── scaler_params.json          # Normalization statistics per asset profile
│
├── frontend/                           # TRACK 3: React + Tailwind + 3D Command Console (Rajvardhan)
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── src/
│       ├── App.jsx
│       ├── api/
│       │   └── simClient.js            # REST & WebSocket client for live telemetry streaming
│       ├── components/
│       │   ├── HeaderBar.jsx           # Asset Profile & Indian Theatre Environment selector
│       │   ├── TelemetryGrid.jsx       # 7-channel synchronized real-time sensor charts
│       │   ├── FaultInjectionPanel.jsx # One-click chaos buttons (Inject Bearing Spall, etc.)
│       │   ├── RulConvergenceChart.jsx # True Simulated RUL vs AI Predicted RUL Fan Chart
│       │   ├── Engine3DTwin.jsx        # Three.js / React Three Fiber 3D X-Ray subsystem view
│       │   └── MaintenanceAlertCard.jsx# Operator triage card with root-cause & action steps
│       └── pages/
│           ├── SingleAssetConsole.jsx  # Phase 1 & 2 main engineering view
│           └── SquadronTriageBoard.jsx # Phase 3 multi-aircraft readiness board
│
├── data/                               # Benchmark & Synthetic Calibration Datasets
│   ├── README_DATA.md                  # Instructions to download NASA C-MAPSS / N-CMAPSS
│   └── sample_logs/                    # Pre-generated run-to-failure CSVs for offline testing
│
├── scripts/                            # Utility & Demo Scripts
│   ├── generate_synthetic_fleet.py     # Generates 100 run-to-failure trajectories for training
│   └── run_headless_benchmark.py       # Runs AI validation and prints RMSE / NASA Score table
│
└── docs/                               # Architecture Diagrams & Phase Reports
    └── Phase1_Engineering_Report.md
```

---

## 8. API Contract & Sample JSON Payloads

Our FastAPI backend exposes clean REST and WebSocket endpoints so the Frontend and AI teams can work independently:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| POST | /api/v1/sim/init | Initializes the Digital Twin with selected asset_profile (GTSE_FIGHTER, DO228_TURBOPROP, MALE_UAV) and environment (LEH_HIGH_ALT, THAR_DESERT, KOCHI_MARITIME). |
| POST | /api/v1/sim/step | Advances simulation by N ticks, runs the 40-tick window through the AI model, and returns telemetry, RUL predictions, and maintenance actions. |
| POST | /api/v1/sim/inject-fault | Injects a progressive physical fault (BEARING_SPALL, HPC_FOULING, HPT_EROSION) with configurable severity (0.1 to 1.0). |
| POST | /api/v1/sim/reset | Resets asset health back to 100% nominal state and clears rolling buffers. |
| WS | /ws/telemetry | Streams live 10 Hz simulation ticks and AI inference packets to the React dashboard. |

### Sample Response from POST /api/v1/sim/step
```json
{
  "tick": 142,
  "asset_id": "IAF-GTSE-04",
  "asset_profile": "GTSE_FIGHTER",
  "environment": "LEH_HIGH_ALT",
  "active_faults": ["BEARING_SPALL"],
  "telemetry": {
    "temperature_c": 638.4,
    "pressure_kpa": 412.0,
    "vibration_mms": 6.45,
    "rpm": 38420,
    "voltage_v": 26.8,
    "current_a": 118.5,
    "load_pct": 85.0
  },
  "ground_truth": {
    "true_health_index": 69.5,
    "true_rul_hours": 5.4
  },
  "ai_prediction": {
    "model_used": "TCN_Quantile_v1",
    "predicted_health_index": 72.0,
    "failure_risk_pct": 78.0,
    "rul_bounds_hours": {
      "lower_q10": 4.1,
      "median_q50": 5.6,
      "upper_q90": 7.0,
      "display_text": "4 - 7 hours"
    },
    "sensor_attribution_pct": {
      "vibration_mms": 41.0,
      "temperature_c": 33.0,
      "current_a": 16.0,
      "pressure_kpa": 10.0
    }
  },
  "maintenance_decision": {
    "status": "PMC_RESTRICTED",
    "priority_score": 82.4,
    "observed_trend": "Rising shaft vibration + rising EGT + elevated starter current draw",
    "recommended_action": "Inspect high-speed turbine bearing and lubrication line before next flight window."
  }
}
```

---

## 9. The 3-Phase Master Roadmap

* Phase 1 (Current Focus): Coupled Physics Digital Twin, Indian Defence Profiles & Baseline RUL Loop
  * Build the single-asset coupled physics simulator (GTSE_FIGHTER, DO228_TURBOPROP, MALE_UAV) with Leh, Thar, and Kochi environmental modifiers.
  * Implement live fault injection, 40-tick feature engineering, baseline TCN/GRU RUL prediction, and Ground-Truth vs. AI validation chart.

* Phase 2: Probabilistic Uncertainty Bounds, Explainable AI & 3D Subsystem X-Ray Console
  * Upgrade AI outputs to full Quantile Regression intervals (4 to 7 hours) and sensor attention heatmaps.
  * Integrate interactive Three.js / React Three Fiber 3D Engine Twin that highlights degrading internal sub-meshes in amber/red.

* Phase 3: Squadron Fleet Scaling, Pre-Flight Mission Simulator & Prescriptive Triage Board
  * Scale to an 8-12 asset squadron dashboard ranking aircraft by FMC (Fully Mission Capable), PMC (Partially Mission Capable), and NMC (Grounded).
  * Add a Pre-Flight Mission Stress-Tester that fast-forwards the Digital Twin at 500x speed to verify if an aircraft can survive an upcoming sortie.

---

## 10. Research Sources, Datasets & iDEX Anchors

### Official Indian Defence Problem Statement Anchors (iDEX DRISHTI)
1. iDEX DRISHTI Challenge (challenges-cpt/2567 on idex.gov.in): Development of High-Fidelity Digital Twin Software Model for Gas Turbine Starter Engine.
2. iDEX DRISHTI Challenge (challenges-cpt/2531 on idex.gov.in): Predictive Maintenance Tool for Do-228 Aircraft Inspections.

### Public Aerospace & Mechanical Datasets Used for Calibration
1. NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation): Run-to-failure turbofan dataset (21 sensors across FD001 to FD004) from the NASA Ames Prognostics Center of Excellence (PCoE) Data Repository.
2. N-CMAPSS (New C-MAPSS by NASA & ETH Zurich): 1 Hz run-to-failure turbofan telemetry across real climb, cruise, and descent flight envelopes with 5 component failure modes.
3. NASA IMS & FEMTO-ST (PRONOSTIA) Bearing Datasets: High-frequency rotating shaft vibration run-to-failure benchmarks from the IEEE PHM Challenge.
4. NASA PCoE Li-Ion Battery Aging Dataset: Used to calibrate electrical voltage sag and internal resistance rise over charge/discharge cycles.

### Foundational Research Papers
* Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation." IEEE International Conference on Prognostics and Health Management (PHM).
* Glaessgen, E., & Stargel, D. (2012). "The Digital Twin Paradigm for Future NASA and U.S. Air Force Vehicles." 53rd AIAA Structures, Structural Dynamics and Materials Conference.
* Zheng, S., Ristovski, K., Farahat, A., & Gupta, C. (2017). "Long Short-Term Memory Network for Remaining Useful Life Estimation." IEEE ICPHM.
* Bai, S., Kolter, J. Z., & Koltun, V. (2018). "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling." arXiv:1803.01271 (TCN Architecture).

---

## 11. Installation & Local Setup Guide

### Prerequisites
* Python 3.10+ or 3.11+
* Node.js 18+ and npm

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/astra-maintain.git
cd astra-maintain
```

### 2. Start the Backend & Physics Simulator (Track 1 & 2)
```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
The FastAPI Swagger UI will be live at http://localhost:8000/docs.

### 3. Train or Evaluate the AI Model (Optional)
```bash
cd ../ai_engine
pip install -r requirements.txt
python training/train_synthetic.py --profile GTSE_FIGHTER --epochs 50
python ../scripts/run_headless_benchmark.py
```

### 4. Start the React Command Console (Track 3)
```bash
cd ../frontend
npm install
npm run dev
```
Open http://localhost:5173 in your browser to launch the ASTRA MAINTAIN Operator Console.

---

## 12. Team Ownership & Contributors (ASTRA Software Team)

| Track | Module Ownership | Team Member(s) | Core Deliverables |
| :--- | :--- | :--- | :--- |
| Track 1 | Backend & Coupled Physics Simulator (Python, FastAPI, NumPy) | Harsh | Coupled degradation math equations across the 7 sensors, 3 Indian defence profiles, 3 theatre weather modifiers, and FastAPI simulation endpoints. |
| Track 2 | AI & Data Science Pipeline (PyTorch, Scikit-Learn, NASA C-MAPSS) | Keshav & Srihari | 40-tick sliding window preprocessor, TCN & Bi-GRU+Attention models with Quantile Loss (4-7 hr RUL bounds), EWMA fallback, and validation benchmarks. |
| Track 3 | Frontend Operator Command Console (React, TailwindCSS, Three.js) | Rajvardhan | Live 7-sensor telemetry dashboard, Ground-Truth vs. AI RUL convergence fan chart, Interactive Fault Injection & Indian Theatre control bar, and Alert cards. |

---
*Built with precision by the ASTRA Software Team.*
