[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.8+](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)

This is a  PyTorch framework for rotating machinery fault diagnosis. It supports clean, noisy, small‑sample, and joint (noise + few‑shot) settings, and runs on public datasets like CWRU and Ottawa.

---

## ✨ Key Features

- Four experiment modes: `clean_test`, `noise_test`, `small_test`, `joint_test`
- Flexible hyperparameters (sequence length, batch size, learning rate, epochs)
- Automatic repeated trials with mean ± std reporting
- GPU acceleration (single or multi‑GPU)
- Easy to extend with new models and datasets

---

## 📁 Repository Layout

```
.
├── CompareModel/          # Other baseline models
├── exp/                   # Core experiment classes
│   └── exp_classification.py
├── general_process.py     # Data utilities
└── run.py                 # Main entry point
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install torch numpy
```

### 2. Prepare your data
Place your dataset under `../Your dataset/Experiment/repeat_00/` with `train/`, `val/`, `test/` subfolders. For repeated runs, create `repeat_00` … `repeat_09`.

### 3. Run an experiment

**Clean test (full data)**
```bash
python run.py --data_set Your dataset --running_type clean_test --num_class 9
```

**Noise test (SNR = -6 dB)**
```bash
python run.py --data_set Your dataset --running_type noise_test --noise_db -6 --num_class 10
```

**Small‑sample test (2 samples per class)**
```bash
python run.py --data_set Your dataset --running_type small_test --small_level L2 --num_class 9
```

**Joint test (noise -2 dB + 1 sample per class)**
```bash
python run.py --data_set Your dataset --running_type joint_test --joint_noise_db -2 --joint_sample_level L1 --num_class 10
```

Add `--itr 10` to repeat experiments and obtain average performance.

---

## 📊 Output

Training progress and test accuracy are printed to the console.  
When repeated trials are enabled, the final summary shows mean loss and accuracy with standard deviation.

---
