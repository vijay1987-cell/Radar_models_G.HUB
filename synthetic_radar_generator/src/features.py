"""
Feature engineering for vessel size classification.
Adapts the Full-24 and Full-42 feature sets from the HQNN paper series.
Target: size_class (small ≤40m, medium 40-130m, large >130m)

Note: az_extent_m is cross-range extent (same as cross_range_extent in papers).
AIS_LENGTH / AIS_WIDTH are excluded — label leakage.
Ship_Length_m excluded — it IS the ground truth for our labels.
"""
import numpy as np
import pandas as pd


def derive_em_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive EM/RCS features from raw radar columns."""
    out = df.copy()
    rng = df['range'].clip(lower=1.0)
    out['log_peak_rcs']  = np.log(df['PeakAmplitude'].clip(lower=1e-6)) + 4 * np.log(rng)
    out['log_total_rcs'] = np.log(df['TotalAmplitude'].clip(lower=1e-6)) + 4 * np.log(rng)
    out['rcs_conc']      = df['PeakAmplitude'] / df['TotalAmplitude'].replace(0, np.nan)
    out['aspect_ratio']  = df['down_range_extent'] / df['az_extent_m'].replace(0, np.nan)
    out['footprint_m2']  = np.log((df['down_range_extent'] * df['az_extent_m']).clip(lower=1e-6))
    out['cr_dr_ratio_c'] = df['cr_dr_ratio'].clip(upper=15.0)
    out['sog']           = df['RSog'].clip(upper=25.0)
    return out


# Full-24 feature set (Papers 6-11) — 10 EM + 14 Kin
EM_FEATURES = [
    'log_peak_rcs', 'log_total_rcs', 'rcs_conc', 'aspect_ratio', 'footprint_m2',
    'SampleCount', 'size_bow_stern_component', 'size_beam_component',
    'ellipse_area', 'cr_dr_ratio_c',
]
KIN_FEATURES = [
    'sog',
    'measured_sog_avg_900',  'measured_sog_avg_1800',
    'measured_sog_avg_3600', 'measured_sog_avg_10800',
    'measured_sog_std_900',  'measured_sog_std_1800',
    'measured_sog_std_3600', 'measured_sog_std_10800',
    'measured_cog_std_900',  'measured_cog_std_1800',
    'measured_cog_std_3600', 'measured_cog_std_10800',
    'displacement',
]
FULL24 = EM_FEATURES + KIN_FEATURES

# EDA-novel features (Paper 10) — 18 additional features
EDA_EM_FEATURES = [
    'measured_TotalAmplitude_avg_900',  'measured_TotalAmplitude_avg_1800',
    'measured_TotalAmplitude_avg_3600', 'measured_TotalAmplitude_avg_10800',
    'measured_rangeStd_900',  'measured_rangeStd_1800',
    'measured_rangeStd_3600', 'measured_rangeStd_10800',
    'measured_azimuthStd_900',  'measured_azimuthStd_1800',
    'measured_azimuthStd_3600', 'measured_azimuthStd_10800',
    'rgw', 'azw',
]
EDA_KIN_FEATURES = [
    'measured_cog_stdlog_900',  'measured_cog_stdlog_1800',
    'measured_cog_stdlog_3600', 'measured_cog_stdlog_10800',
]
EDA_NOVEL = EDA_EM_FEATURES + EDA_KIN_FEATURES
FULL42 = FULL24 + EDA_NOVEL

# Extent features for Extent-QJL (Paper 9) — 13 features
EXTENT_FEATURES = [
    'az_extent_m', 'down_range_extent', 'euclid_size',
    'size_beam_component', 'size_bow_stern_component', 'ellipse_area', 'cr_dr_ratio_c',
    'aspect_ratio',
    'displacement',
    'measured_size_avg_900', 'measured_size_avg_1800',
    'measured_size_avg_3600', 'measured_size_avg_10800',
]

# Dual-stream split (Paper 11)
EM_STREAM   = EM_FEATURES + EDA_EM_FEATURES   # 24 features
KIN_STREAM  = KIN_FEATURES + EDA_KIN_FEATURES  # 18 features

SIZE_CLASSES = ['small', 'medium', 'large']
CLASS2IDX    = {c: i for i, c in enumerate(SIZE_CLASSES)}
