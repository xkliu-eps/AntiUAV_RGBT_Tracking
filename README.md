# AntiUAV RGBT Tracking

This project extends the **PyTracking** single-modal tracking framework with **pyTrackBridge**, a non-intrusive monkey-patch layer that enables multi-modal (IR + RGB) object tracking. Our work **OCTA-SOT** (Accepted by ECCV 2026) — Online Cross-Modal Trajectory Adjustment for RGBT Anti-UAV Single Object Tracking under Spatio-Temporal Misalignment — is fully integrated.

> ⚠️ **Note:** The OCTA-SOT tracker code is currently being organized and will be updated soon. Stay tuned!

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│  pyTrackBridge   ← Multi-modal patch layer            │
│  ├── rig.py          Runtime replacement engine       │
│  ├── core/           Templates & replacement rules    │
│  ├── tracker/        Multi-modal trackers (OCTA-SOT)  │
│  ├── dataset/        Multi-modal dataset adapters     │
│  └── script/         Experiment configs               │
├──────────────────────────────────────────────────────┤
│  pytracking        ← Single-modal tracking & eval     │
│  ltr               ← Training framework               │
└──────────────────────────────────────────────────────┘
```

- 🔷 **PyTracking** — Implementations & pretrained models for ATOM, DiMP, PrDiMP, KYS, LWL, KeepTrack, ToMP, RTS, TaMOs, and more
- 🔶 **pyTrackBridge** — Non-intrusive runtime monkey-patch layer for dual-modal (visible + infrared + event) inputs
- 🏅 **OCTA-SOT** (ECCV 2026) — Multi-modal cooperative tracker with cross-modal alignment & Kalman filter fusion

---

## 🏅 OCTA-SOT

**OCTA-SOT** (Online Cross-Modal Trajectory Adjustment for RGBT Anti-UAV Single Object Tracking under Spatio-Temporal Misalignment) tackles robust RGBT anti-UAV tracking under realistic spatio-temporal misalignment between sensors — a previously unaddressed problem that existing multi-modal trackers overlook by assuming perfectly calibrated inputs.

OCTA-SOT is a **training-free, plug-and-play** online adjustment module that can be applied to any off-the-shelf single-modal tracker without retraining. It maintains per-modality trajectory history and reliability, calibrates cross-modal mappings via a Kalman-driven mechanism, and adaptively adjusts degraded tracker states to produce robust final outputs.

<!-- TODO: insert overview figure -->

On the Anti-UAV300 dataset, OCTA-SOT boosts the DiMP tracker by **+9.3% AUC, +12.9% Precision, and +11.9% Normalized Precision**, surpassing state-of-the-art methods.

> Accepted by ECCV 2026. *(citation details pending update)*

---

## ⚙️ Setup

### 🔧 Step 1: Install PyTracking + LTR

Follow the original PyTracking setup first:

```bash
git clone https://github.com/visionml/pytracking.git
cd pytracking
git submodule update --init
bash install.sh <conda_install_path> pytracking
```

Verify that the following local config files are generated after installation:

- `pytracking/evaluation/local.py` — dataset paths, network weights path, results path
- `ltr/admin/local.py` — training workspace, pretrained networks path

If not auto-generated:

```python
from pytracking.evaluation.environment import create_default_local_file
create_default_local_file()

from ltr.admin.environment import create_default_local_file
create_default_local_file()
```

### ⬇️ Step 2: Download Pretrained Models

Download tracker weights from [MODEL_ZOO.md](MODEL_ZOO.md) and place them under `pytracking/networks/`.

### 🔌 Step 3: pyTrackBridge

No additional dependencies are required. Multi-modal datasets (MMMUAV, Anti-UAV300, etc.) need to be downloaded separately and their paths configured in the experiment scripts.

---

## 🚀 Quick Start

### 🎯 Multi-modal tracking (pyTrackBridge)

```bash
cd pyTrackBridge

# Basic dual-modal tracking (dual DiMP)
python rig.py script/combin_test.py

# Run OCTA-SOT
python rig.py script/OCTA_STO.0.1.py
```

### 🏠 Single-modal tracking (original PyTracking)

```bash
cd pytracking

# Webcam demo
python run_webcam.py dimp dimp50

# Single sequence evaluation
python run_tracker.py dimp dimp50 --dataset_name otb --sequence Soccer

# Batch experiment
python run_experiment.py myexperiments uav_test --dataset_name uav
```

---

## 📝 Experiment Configs

A pyTrackBridge config is a concise Python file:

```python
# include "core.multiModeTrack"

def get_tracker_list():
    return [{
        'name': 'my_experiment',
        'tracker_setting': {
            'type': 'combin',            # Dual independent trackers
            'infrared_params': 'dimp.dimp50',
            'visible_params': 'dimp.dimp50',
        }
    }]

def get_sequence_list():
    import pyTrackBridge.dataset.my_dataset as ds
    return ds.get_sequence_list()
```

**Tracker modes:**
- 🔗 `combin` — Dual independent trackers (one tracker per modality)
- 🧠 `function` — Custom fusion tracker (e.g., OCTA-SOT), created via a callable
- 📦 `params` — Load a tracker from `pyTrackBridge/tracker/` with its parameter file

---

## 📂 Directory Structure

```
pytracking-master/
├── pytracking/            # Single-modal tracking inference & evaluation
│   ├── tracker/           #   Tracker implementations (ATOM, DiMP, ToMP, ...)
│   ├── evaluation/        #   Dataset interfaces & evaluation tools
│   ├── parameter/         #   Tracker parameter files
│   └── ...
├── ltr/                   # Training framework
│   ├── train_settings/    #   Per-tracker training configurations
│   ├── models/            #   Network model definitions
│   └── ...
├── pyTrackBridge/         # Multi-modal patch layer
│   ├── rig.py             #   Runtime replacement engine
│   ├── core/              #   Templates & replacement rules
│   ├── tracker/           #   Multi-modal trackers (includes OCTA-SOT)
│   ├── dataset/           #   Multi-modal dataset adapters
│   ├── paramer/           #   Custom parameter files
│   └── script/            #   Experiment config files
├── install.sh
├── INSTALL.md
├── MODEL_ZOO.md           # Pretrained models & benchmark results
└── README.md
```

---

## 📖 Citation

For the original PyTracking framework and trackers (ATOM, DiMP, PrDiMP, KYS, LWL, KeepTrack, ToMP, RTS, TaMOs), please refer to the corresponding papers listed in [MODEL_ZOO.md](MODEL_ZOO.md).

For OCTA-SOT and the pyTrackBridge multi-modal extension, please cite:

> *OCTA-SOT: Online Cross-Modal Trajectory Adjustment for RGBT Anti-UAV Single Object Tracking under Spatio-Temporal Misalignment*. Accepted by ECCV 2026. *(citation details pending update)*

---

## 🙏 Acknowledgments

This project is built on [visionml/pytracking](https://github.com/visionml/pytracking). We thank the original authors for their outstanding work.
