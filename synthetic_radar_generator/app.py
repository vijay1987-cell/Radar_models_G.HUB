#!/usr/bin/env python3
"""
Streamlit UI — Synthetic Radar Dataset Generator
Generates balanced synthetic vessel radar detections from real data distributions.

Run:
  pip install -r requirements.txt
  streamlit run app.py

You must supply your own labeled radar CSV via the sidebar file uploader.
Required columns: size_class, ObjID, PeakAmplitude, TotalAmplitude, range,
azimuth, down_range_extent, az_extent_m, cr_dr_ratio, RSog, RCog,
RLatitude, RLongitude, SampleCount, size_bow_stern_component,
size_beam_component, ellipse_area, and rolling KIN/EDA features.
"""
import sys, io, time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from features import derive_em_features, FULL24, EDA_NOVEL, FULL42, CLASS2IDX, KIN_FEATURES

OUT_DIR     = Path('.')
BASE_EPOCH  = 1_700_000_000

# Raw columns from the CSV that derive_em_features() reads as inputs
RAW_SIGNAL_COLS = [
    'PeakAmplitude', 'TotalAmplitude', 'range', 'azimuth',
    'down_range_extent', 'az_extent_m', 'cr_dr_ratio',
    'RSog', 'RCog', 'RLatitude', 'RLongitude',
]
# FULL42 features that are directly in the CSV (not computed by derive_em_features)
RAW_EM_COLS  = ['SampleCount', 'size_bow_stern_component', 'size_beam_component', 'ellipse_area']
# KIN_FEATURES minus 'sog' (sog = RSog.clip(25) — re-derived, not sampled directly)
RAW_KIN_COLS = [f for f in KIN_FEATURES if f != 'sog']
# All columns to bootstrap-sample (raw signal + raw model features + EDA rolling features)
BASE_SAMPLE_COLS = RAW_SIGNAL_COLS + RAW_EM_COLS + RAW_KIN_COLS + EDA_NOVEL
# Positive-only columns: clip to >0 after noise is added
POSITIVE_COLS = ['PeakAmplitude', 'TotalAmplitude', 'range', 'SampleCount',
                 'down_range_extent', 'az_extent_m']
CLASS_COLORS = {'small': '#3b82f6', 'medium': '#f59e0b', 'large': '#10b981'}
CLASS_INFO   = {
    'small':  '≤ 40 m — tugs, fishing, sailing',
    'medium': '40–130 m — coastal cargo, tankers',
    'large':  '> 130 m — ocean cargo, VLCC',
}

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Synthetic Radar Generator",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📡 Synthetic Radar Dataset Generator")
st.markdown(
    "Generate labeled synthetic X-band radar vessel detections "
    "from real per-class feature distributions via bootstrap sampling."
)

# ── Load and cache real data ───────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading real radar data…")
def load_real_data(file_bytes: bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    # Do NOT call derive_em_features here — we sample raw columns and re-derive later
    sample_cols = [c for c in BASE_SAMPLE_COLS if c in df.columns]
    track_lens  = df.groupby('ObjID').size().values
    class_data  = {}
    for cls in ['small', 'medium', 'large']:
        sub = df[df['size_class'] == cls][sample_cols].copy()
        for col in sample_cols:
            if sub[col].isna().any():
                sub[col] = sub[col].fillna(sub[col].median())
        class_data[cls] = {'vals': sub.values.astype('float64'), 'cols': sample_cols}
    # Also store per-class derived feature arrays for stats display
    df_derived = derive_em_features(df)
    derived_cols = [f for f in FULL42 if f in df_derived.columns]
    derived_data = {}
    for cls in ['small', 'medium', 'large']:
        sub = df_derived[df_derived['size_class'] == cls][derived_cols].copy()
        for col in derived_cols:
            if sub[col].isna().any():
                sub[col] = sub[col].fillna(sub[col].median())
        derived_data[cls] = {'vals': sub.values.astype('float64'), 'cols': derived_cols}
    return class_data, derived_data, track_lens, len(df), df['ObjID'].nunique()

# ── Sidebar: data source ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Data Source")
    uploaded_file = st.file_uploader(
        "Upload labeled radar CSV",
        type=['csv'],
        help="CSV must contain size_class column with values: small, medium, large",
    )
    if uploaded_file is None:
        st.warning("Upload your labeled radar CSV to continue.")
        st.info(
            "Required columns: `size_class`, `ObjID`, `PeakAmplitude`, "
            "`TotalAmplitude`, `range`, `azimuth`, `down_range_extent`, "
            "`az_extent_m`, `cr_dr_ratio`, `RSog`, `RCog`, `RLatitude`, "
            "`RLongitude`, `SampleCount`, `size_bow_stern_component`, "
            "`size_beam_component`, `ellipse_area`, and rolling KIN/EDA features."
        )
        st.stop()

    file_bytes = uploaded_file.read()

class_data, derived_data, real_track_lens, total_real_rows, total_real_tracks = load_real_data(file_bytes)
sample_cols = class_data['small']['cols']
derived_cols = derived_data['small']['cols']

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    # Class percentage split
    st.subheader("Size Class Distribution")
    total_tracks = st.number_input(
        "Total unique tracks", min_value=30, max_value=30_000,
        value=1500, step=50,
        help="Total distinct vessel tracks across all classes.",
    )

    st.markdown("**Class split — must sum to 100 %**")
    col_a, col_b, col_c = st.columns(3)
    pct_small  = col_a.number_input("🔵 Small %",  min_value=0, max_value=100, value=33, step=1)
    pct_medium = col_b.number_input("🟡 Medium %", min_value=0, max_value=100, value=33, step=1)
    pct_large  = col_c.number_input("🟢 Large %",  min_value=0, max_value=100, value=34, step=1)

    total_pct = pct_small + pct_medium + pct_large
    if total_pct != 100:
        st.error(f"Sum = {total_pct}% — adjust so total equals 100%")
        st.stop()
    else:
        st.success(f"Sum = {total_pct}% ✓")

    tracks_per_class = {
        'small':  max(0, round(total_tracks * pct_small  / 100)),
        'medium': max(0, round(total_tracks * pct_medium / 100)),
        'large':  max(0, round(total_tracks * pct_large  / 100)),
    }
    pct_map  = {'small': pct_small, 'medium': pct_medium, 'large': pct_large}
    selected = [cls for cls in ['small', 'medium', 'large'] if pct_map[cls] > 0]

    for cls in selected:
        real_n = class_data[cls]['vals'].shape[0]
        st.caption(
            f"**{cls.capitalize()}** ({CLASS_INFO[cls]})  "
            f"→ {tracks_per_class[cls]:,} tracks  |  real source: {real_n:,} rows"
        )

    st.divider()

    # Track Settings header kept for scan window below
    st.subheader("Track Settings")

    # Scan window / time resolution
    st.subheader("Scan Window")
    window_map = {
        "Per minute  (1 min — scan-level, ~1 scan/min)":    (60,   "scan-level"),
        "Per 5 min   (5-min strided, last det per window)":  (300,  "5-min"),
        "Per 30 min  (30-min windowed aggregate)":           (1800, "30-min"),
        "Per hour    (1-hr aggregate)":                      (3600, "1-hr"),
    }
    win_choice   = st.selectbox("Scan interval", list(window_map.keys()), index=0)
    scan_interval, win_label = window_map[win_choice]

    # Track duration
    track_dur_min = st.slider(
        "Track duration (minutes)", min_value=5, max_value=480,
        value=60, step=5,
        help="How long each simulated vessel track lasts.",
    )
    scans_per_track  = max(1, track_dur_min * 60 // scan_interval)
    rows_for_class   = {cls: tracks_per_class[cls] * scans_per_track for cls in selected}
    total_est        = sum(rows_for_class.values())

    st.divider()

    # Advanced
    with st.expander("Advanced"):
        noise_pct   = st.slider("Feature noise (% of std)", 0, 20, 5) / 100.0
        rand_seed   = st.number_input("Random seed", 0, 99999, 42)
        save_to_disk = st.checkbox("Save CSV to current directory", value=False)
        if save_to_disk:
            out_fname = st.text_input("Output filename", value="radar_features_synthetic_custom.csv")

    st.divider()
    st.markdown("**Estimated output**")
    st.metric("Total rows", f"{total_est:,}")
    col_s, col_m, col_l = st.columns(3)
    col_s.metric("Small tracks",  f"{tracks_per_class['small']:,}  ({pct_small}%)")
    col_m.metric("Medium tracks", f"{tracks_per_class['medium']:,}  ({pct_medium}%)")
    col_l.metric("Large tracks",  f"{tracks_per_class['large']:,}  ({pct_large}%)")
    st.metric("Scans / track", f"{scans_per_track}  ×  {scan_interval}s ({win_label})")

    generate_btn = st.button("🚀 Generate Dataset", type="primary", width='stretch')

# ── Main area: real data summary ───────────────────────────────────────────────
tab_gen, tab_real = st.tabs(["Generate", "Real Data Reference"])

with tab_real:
    st.subheader("Real dataset statistics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total rows", f"{total_real_rows:,}")
    c2.metric("Unique tracks", f"{total_real_tracks:,}")
    c3.metric("Avg scans/track", f"{total_real_rows // total_real_tracks:,}")

    real_cls_counts = {cls: class_data[cls]['vals'].shape[0] for cls in ['small','medium','large']}
    fig_real = go.Figure(go.Bar(
        x=list(real_cls_counts.keys()),
        y=list(real_cls_counts.values()),
        marker_color=[CLASS_COLORS[c] for c in real_cls_counts],
        text=[f"{v:,}" for v in real_cls_counts.values()],
        textposition='outside',
    ))
    fig_real.update_layout(title="Real class distribution", yaxis_title="Rows",
                           height=300, showlegend=False)
    st.plotly_chart(fig_real, width='stretch')

    st.subheader("Key feature means per class (real data)")
    # Show real-data stats for both raw and derived key features
    check_raw     = ['PeakAmplitude', 'TotalAmplitude', 'range', 'azimuth', 'SampleCount', 'rgw', 'azw']
    check_derived = ['log_peak_rcs', 'log_total_rcs', 'rcs_conc', 'footprint_m2', 'sog', 'aspect_ratio']
    rows = []
    for f in check_raw:
        if f not in sample_cols: continue
        fi = sample_cols.index(f)
        row = {'Feature': f, 'Type': 'raw'}
        for cls in ['small', 'medium', 'large']:
            v = class_data[cls]['vals'][:, fi]
            row[f'{cls} μ'] = f"{v.mean():.3f}";  row[f'{cls} σ'] = f"{v.std():.3f}"
        rows.append(row)
    for f in check_derived:
        if f not in derived_cols: continue
        fi = derived_cols.index(f)
        row = {'Feature': f, 'Type': 'derived'}
        for cls in ['small', 'medium', 'large']:
            v = derived_data[cls]['vals'][:, fi]
            row[f'{cls} μ'] = f"{v.mean():.3f}";  row[f'{cls} σ'] = f"{v.std():.3f}"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).set_index('Feature'), width='stretch')

# ── Generation tab ─────────────────────────────────────────────────────────────
with tab_gen:
    if not generate_btn:
        st.info("Configure settings in the sidebar and click **Generate Dataset**.")
        st.stop()

    # ── Run generation ─────────────────────────────────────────────────────────
    progress = st.progress(0, text="Starting…")
    t_start  = time.time()
    rng      = np.random.default_rng(rand_seed)
    parts    = []

    for step, cls in enumerate(selected):
        n_rows_cls = rows_for_class[cls]
        progress.progress(step / len(selected),
                          text=f"Bootstrapping {cls} class ({n_rows_cls:,} rows)…")
        vals   = class_data[cls]['vals']
        cols   = class_data[cls]['cols']
        n_real = vals.shape[0]

        # Bootstrap-sample raw source columns
        idx     = rng.integers(0, n_real, size=n_rows_cls)
        sampled = vals[idx].copy()

        # Add per-feature Gaussian noise
        if noise_pct > 0:
            stds    = vals.std(axis=0, keepdims=True).clip(min=1e-9)
            sampled += rng.standard_normal(sampled.shape) * stds * noise_pct

        # Enforce physical constraints on raw columns
        part = pd.DataFrame(sampled, columns=cols)
        for pc in POSITIVE_COLS:
            if pc in part.columns:
                part[pc] = part[pc].clip(lower=1e-6)

        # Re-derive EM features from the noisy raw columns (ensures consistency)
        part = derive_em_features(part)
        part['size_class'] = cls
        parts.append(part)

    progress.progress(0.7, text="Assigning tracks and timestamps…")
    synth = pd.concat(parts, ignore_index=True)
    synth = synth.sample(frac=1.0, random_state=rand_seed).reset_index(drop=True)

    # Assign ObjIDs and timestamps
    obj_ids, rtimes = [], []
    obj_counter = 1
    i = 0
    while i < len(synth):
        tlen = scans_per_track
        tlen = min(tlen, len(synth) - i)
        t0_track = BASE_EPOCH + int(rng.integers(0, 86_400 * 30))
        obj_ids.extend([f'SYN_{obj_counter:07d}'] * tlen)
        rtimes.extend(int(t0_track) + np.arange(tlen, dtype=int) * scan_interval)
        i += tlen
        obj_counter += 1

    synth['ObjID']       = obj_ids[:len(synth)]
    synth['Rtime_epoch'] = rtimes[:len(synth)]

    elapsed = time.time() - t_start
    progress.progress(1.0, text=f"Done in {elapsed:.1f}s")
    time.sleep(0.3)
    progress.empty()

    # ── Stats ──────────────────────────────────────────────────────────────────
    st.success(f"Generated **{len(synth):,} rows** across {synth['ObjID'].nunique():,} tracks in {elapsed:.1f}s")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total rows",    f"{len(synth):,}")
    m2.metric("Unique tracks", f"{synth['ObjID'].nunique():,}")
    m3.metric("Scans/track",   f"{scans_per_track}")
    m4.metric("Scan interval", f"{scan_interval}s")

    # Class distribution bar chart
    cls_counts = synth['size_class'].value_counts()
    fig_cls = go.Figure(go.Bar(
        x=cls_counts.index.tolist(),
        y=cls_counts.values.tolist(),
        marker_color=[CLASS_COLORS.get(c, '#6b7280') for c in cls_counts.index],
        text=[f"{v:,}" for v in cls_counts.values],
        textposition='outside',
    ))
    fig_cls.update_layout(title="Synthetic class distribution", yaxis_title="Rows",
                          height=300, showlegend=False)
    st.plotly_chart(fig_cls, width='stretch')

    # Real vs synthetic distribution comparison — show both raw and derived features
    st.subheader("Distribution fidelity — real vs. synthetic")
    cmp_raw     = ['PeakAmplitude', 'TotalAmplitude', 'range', 'SampleCount', 'rgw', 'azw']
    cmp_derived = ['log_peak_rcs', 'log_total_rcs', 'rcs_conc', 'footprint_m2', 'sog', 'aspect_ratio']
    cmp_feats   = [f for f in cmp_raw + cmp_derived if f in synth.columns][:6]

    fig_cmp = make_subplots(rows=2, cols=3, subplot_titles=cmp_feats, shared_yaxes=False)
    for fi, feat in enumerate(cmp_feats):
        row, col = divmod(fi, 3)
        # Determine real source: sample_cols (raw) or derived_cols (derived)
        if feat in sample_cols:
            get_real = lambda c: class_data[c]['vals'][:, sample_cols.index(feat)]
        elif feat in derived_cols:
            get_real = lambda c: derived_data[c]['vals'][:, derived_cols.index(feat)]
        else:
            continue
        for cls in selected:
            real_vals  = get_real(cls)
            synth_vals = synth[synth['size_class'] == cls][feat].values
            for label, v, dash in [('real', real_vals, 'dot'), ('synth', synth_vals, 'solid')]:
                counts, edges = np.histogram(v[np.isfinite(v)], bins=40)
                centers = (edges[:-1] + edges[1:]) / 2
                fig_cmp.add_trace(
                    go.Scatter(x=centers, y=counts / counts.sum(), mode='lines',
                               name=f'{cls} {label}',
                               line=dict(color=CLASS_COLORS.get(cls, '#6b7280'),
                                         dash=dash, width=1.5),
                               showlegend=(fi == 0)),
                    row=row + 1, col=col + 1,
                )
    fig_cmp.update_layout(height=450,
                          title_text="Feature distributions: real (dotted) vs synthetic (solid)")
    st.plotly_chart(fig_cmp, width='stretch')

    # Feature stats table covering raw + derived
    st.subheader("Feature statistics comparison")
    stat_feats = [f for f in (cmp_raw + cmp_derived) if f in synth.columns]
    stat_rows = []
    for feat in stat_feats:
        ftype = 'raw' if feat in RAW_SIGNAL_COLS + RAW_EM_COLS else 'derived'
        if feat in sample_cols:
            get_real = lambda c, f=feat: class_data[c]['vals'][:, sample_cols.index(f)]
        elif feat in derived_cols:
            get_real = lambda c, f=feat: derived_data[c]['vals'][:, derived_cols.index(f)]
        else:
            continue
        for cls in selected:
            real_v = get_real(cls)
            syn_v  = synth[synth['size_class'] == cls][feat].dropna().values
            stat_rows.append({
                'Feature': feat, 'Type': ftype, 'Class': cls,
                'Real μ': f'{real_v.mean():.3f}', 'Synth μ': f'{syn_v.mean():.3f}',
                'Real σ': f'{real_v.std():.3f}',  'Synth σ': f'{syn_v.std():.3f}',
                'Δμ':     f'{abs(real_v.mean() - syn_v.mean()):.4f}',
            })
    st.dataframe(pd.DataFrame(stat_rows), width='stretch', hide_index=True)

    # ── Download ───────────────────────────────────────────────────────────────
    st.subheader("Download")
    csv_bytes = synth.to_csv(index=False).encode('utf-8')
    fname = f"synthetic_radar_{len(selected)}cls_{len(synth)//1000}k_{win_label}.csv"
    st.download_button(
        label=f"⬇️ Download CSV ({len(synth):,} rows, {len(csv_bytes)/1e6:.1f} MB)",
        data=csv_bytes,
        file_name=fname,
        mime='text/csv',
        width='stretch',
    )

    # Save to disk
    if save_to_disk:
        out_path = OUT_DIR / out_fname
        synth.to_csv(out_path, index=False)
        st.success(f"Saved to disk: `{out_path}`  ({out_path.stat().st_size/1e6:.1f} MB)")

    n_out_cols = len([c for c in synth.columns if c not in ('ObjID', 'Rtime_epoch', 'size_class')])
    st.caption(
        f"Bootstrap noise: {noise_pct*100:.0f}% of per-feature std  |  "
        f"Seed: {rand_seed}  |  Output columns: {n_out_cols} "
        f"({len(sample_cols)} raw sampled + derived EM features)"
    )
