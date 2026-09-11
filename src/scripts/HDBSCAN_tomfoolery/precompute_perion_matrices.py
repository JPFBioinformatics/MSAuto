"""
Precomputes per-ion X matrices using different norm_types so that we can access easily later in
parallelization
"""

import json
import numpy as np
from pathlib import Path
from src.scripts.helpers import normalize_matrix, rolling_median_2d

cwt_data = [True, False]
thresholds = [True, False]

include_cwt_data = False

threshold = False

norm_types = ['modz', 'compressed']

root_dir = Path(__file__).resolve().parents[2]
out_dir = root_dir / 'peak_metrics_data'

if include_cwt_data:
    out_dir = out_dir / '5_embedding'
else:
    out_dir = out_dir / '3_embedding'

if threshold:
    out_dir = out_dir / 'threshold_true'
    json_file = root_dir / 'im_data_threshold.json'
else:
    out_dir = out_dir / 'threshold_false'
    json_file = root_dir / 'im_data_no_threshold.json'

with open(json_file, 'r') as f:
    data = json.load(f)

first_derivs = np.array(data['first_derivs'], dtype=float)
fd_trend = rolling_median_2d(first_derivs, window=51)

second_derivs = np.array(data['second_derivs'], dtype=float)

smoothted_signal = np.array(data['smoothed_signal'], dtype=float)
sm_trend = rolling_median_2d(smoothted_signal, window=51)

for norm_type in norm_types:
    if include_cwt_data:
        cwt_scores = np.array(data['cwt_max_scores'], dtype=float)
        cwt_scales = np.array(data['cwt_max_scales'], dtype=float)
        X = np.stack([
            normalize_matrix(first_derivs - fd_trend, norm_method=norm_type),
            normalize_matrix(second_derivs, norm_method=norm_type),
            normalize_matrix(smoothted_signal - sm_trend, norm_method=norm_type),
            normalize_matrix(cwt_scales, norm_method=norm_type),
            normalize_matrix(cwt_scores, norm_method=norm_type)
        ])
    else:
        X = np.stack([
            normalize_matrix(first_derivs - fd_trend, norm_method=norm_type),
            normalize_matrix(second_derivs, norm_method=norm_type),
            normalize_matrix(smoothted_signal - sm_trend, norm_method=norm_type)
        ])
    np.save(out_dir / 'per_row' / 'precomputed' / f'X_{norm_type}.npy', X)

np.save(out_dir / 'per_row' / 'precomputed' / 'unique_mzs.npy', np.array(data['mzs']))
