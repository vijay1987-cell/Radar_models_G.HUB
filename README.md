# Radar Models — Vessel Size Classification

**Author:** Vijay Kumar V — Independent Researcher, AA Enterprises  
**Contact:** vinayitsv14@gmail.com  
**Data:** Proprietary X-band coastal radar (AA Enterprises) — not included

---

## Task

3-class vessel size classification from ground-based X-band radar detections:

| Class | Length | Typical types |
|---|---|---|
| small | ≤ 40 m | Tugs, fishing boats, sailing vessels |
| medium | 40 – 130 m | Coastal cargo, tankers, dredgers |
| large | > 130 m | Ocean cargo, VLCC tankers |

Labels derived from AIS-verified dimensions in `VESSEL DATA.xlsx`.  
Dataset: 123,051 radar detections · 712 tracks · 387 unique MMSIs.

---

## Inference strategy — 5-minute strided sampling

Rather than classifying every radar scan (~1/min), the system picks the last
detection in each 5-minute window per track (ObjID must be consistently active
for ≥ 5 min). This gives a **5× reduction in inference calls** with < 2 F1-macro
point loss at track level.

| Strategy | Acc | F1-macro | Inference calls/track |
|---|---|---|---|
| Scan-level (1/min) | 83.6% | 0.753 | ~173 avg |
| **5-min strided** | **82.2%** | **0.738** | **~32 avg** |
| 30-min windowed | 81.8% | 0.718 | ~7 avg |

---

## Models

### Classical / Hybrid (trained on 5-min strided samples — 18,312 train samples)

| Model | File | Acc | F1-macro |
|---|---|---|---|
| XGB Full-24 | `vessel_size_clf/xgb_full24_5min.pkl` | 81.3% | 0.734 |
| EDA-QJL k=512 | `vessel_size_clf/edaqjl_5min.pkl` | 82.2% | 0.738 |

### Quantum ML (trained on track-level aggregated 5-min features — 498 train tracks, 8 qubits)

| Model | File | Acc | F1-macro |
|---|---|---|---|
| HQNN 8q 3L | `vessel_size_clf/HQNN_5min_weights.pt` | 64.5% | 0.585 |
| VQC 8q 3L | `vessel_size_clf/VQC_5min_weights.pt` | 17.8% | 0.182 |

### Reference — scan-level baselines (trained on all 89,335 detections)

| Model | File | Acc | F1-macro |
|---|---|---|---|
| XGB Full-24 | `vessel_size_clf/xgb_full24.pkl` | 83.6% | 0.753 |
| EDA-QJL k=512 | `vessel_size_clf/edaqjl_k512.pkl` | 83.6% | 0.748 |
| GBT Full-24 | `vessel_size_clf/gbt_full24.pkl` | 80.8% | 0.740 |
| DualStream | `vessel_size_clf/dualstream_size_weights.pt` | 81.3% | 0.739 |
| PINN | `vessel_size_clf/pinn_size_weights.pt` | 74.8% | 0.685 |

---

## Feature sets

**Full-24 (Papers 6–11):** 10 EM + 14 kinematic features derived from raw radar.  
**EDA-novel (Paper 10):** 18 additional features identified by multi-metric EDA.  
**QML-8:** 8 size-discriminative features for 8-qubit circuits.

```
QML-8: log_peak_rcs, log_total_rcs, SampleCount, footprint_m2,
        aspect_ratio, size_beam_component, size_bow_stern_component, ellipse_area
```

AIS_LENGTH / AIS_WIDTH are **excluded from all models** (label leakage).  
Ship_Length_m is **excluded** (it is the ground truth for class labels).

---

## Reproducibility

- ObjID-stratified 70/30 split, seed=42
- Minority oversampling (sklearn resample) to balance training classes
- XGBoost: n_estimators=500, max_depth=4, lr=0.05, subsample=0.8
- HQNN: 8 qubits, 3 StronglyEntanglingLayers, Adam lr=5e-3, 60 epochs
- VQC: 8 qubits, 3 BasicEntanglerLayers, Adam lr=5e-3, 60 epochs
- Track-level prediction: mean softmax across all samples per ObjID

---

*This repository contains model weights and configs only. Training data is
proprietary to AA Enterprises and cannot be included.*
