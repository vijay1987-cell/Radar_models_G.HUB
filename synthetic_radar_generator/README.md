# Synthetic Radar Dataset Generator

A Streamlit web app that generates labeled synthetic X-band vessel radar detections from real per-class feature distributions using bootstrap sampling with Gaussian noise.

## What It Does

- Loads a labeled radar CSV (your real data, uploaded at runtime)
- Bootstrap-samples raw signal columns per vessel size class
- Adds configurable Gaussian noise (default 5% of per-feature σ) for diversity
- Re-derives all EM features (`log_peak_rcs`, `log_total_rcs`, `rcs_conc`, etc.) from the noisy raw values to ensure mathematical consistency
- Assigns synthetic track IDs (`SYN_0000001`, …) and timestamps
- Exports the dataset as a downloadable CSV

## Setup

```bash
git clone <this-repo>
cd synthetic_radar_generator

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

The app opens at `http://localhost:8501` by default.

## Usage

1. **Upload your labeled radar CSV** — use the sidebar file uploader
2. Configure class distribution (e.g. 33% / 33% / 34% for balanced)
3. Set total unique tracks, scan window, and track duration
4. Click **Generate Dataset**
5. Download the CSV or save to disk

## Required Input CSV Columns

| Column | Description |
|---|---|
| `size_class` | Vessel size label: `small`, `medium`, or `large` |
| `ObjID` | Track identifier |
| `PeakAmplitude`, `TotalAmplitude` | Raw radar amplitudes |
| `range`, `azimuth` | Detection range (m) and azimuth (deg) |
| `down_range_extent`, `az_extent_m` | Extent dimensions (m) |
| `cr_dr_ratio` | Cross-range to down-range ratio |
| `RSog`, `RCog`, `RLatitude`, `RLongitude` | AIS kinematics |
| `SampleCount` | Scan sample count |
| `size_bow_stern_component`, `size_beam_component`, `ellipse_area` | Physical size estimates |
| Rolling KIN/EDA features | `measured_sog_avg_*`, `measured_cog_std_*`, `rgw`, `azw`, etc. |

> **Note:** The real dataset is proprietary and is not included in this repository.

## Output Columns

The generated CSV includes:
- All raw sampled columns
- Derived EM features: `log_peak_rcs`, `log_total_rcs`, `rcs_conc`, `aspect_ratio`, `footprint_m2`, `cr_dr_ratio_c`, `sog`
- `ObjID` (synthetic track ID), `Rtime_epoch` (Unix timestamp), `size_class`

## Vessel Size Classes

| Class | Length | Examples |
|---|---|---|
| small | ≤ 40 m | Tugs, fishing, sailing |
| medium | 40–130 m | Coastal cargo, smaller tankers |
| large | > 130 m | Ocean cargo, VLCC |

## Validation

Training on 1 M balanced synthetic rows and evaluating on 214 real held-out test tracks:

| Model | Acc | F1-macro | F1-small | F1-medium | F1-large |
|---|---|---|---|---|---|
| EDA-QJL k=512 | **91.6%** | **0.892** | 0.863 | 0.870 | 0.943 |
| XGB Full-24 | 89.7% | 0.872 | 0.831 | 0.840 | 0.944 |

This synthetic-to-real generalization confirms the generator produces statistically faithful distributions.

## Author

Vijay Kumar V — Independent Researcher
