"""

uses noise model(s) to generate synthetic dataset for testing/training

"""
import json
from pathlib import Path
import pandas as pd
import numpy as np

def load_noise_model(model_dir, sn_min_shape=20):
    """per-sample ok-row noise pools and baselines, plus the peak tables the generator samples from"""
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / 'model_meta.json').read_text())
    rows = pd.read_csv(model_dir / 'row_stats.csv')
    fits = pd.read_csv(model_dir / 'peak_fits.csv')
    qc = rows.set_index(['sample', 'ion'])['qc']

    noise, baselines = {}, {}
    for name in meta['samples']:
        z = np.load(model_dir / name / 'bl_noise.npz')
        off = z['residual_offsets']
        for i, ion in enumerate(z['ions']):
            label = qc.get((name, float(ion)), 'no_data')
            if label != 'no_data':
                baselines[(name, float(ion))] = z['baselines'][i]
            if label == 'ok':
                noise[(name, float(ion))] = z['residuals_flat'][off[i]:off[i + 1]]

    fitted = fits[fits['status'].isin(['ok', 'no_errorbars'])]
    shapes = fitted[(fitted['row_qc'] == 'ok') & (fitted['sn'] > sn_min_shape)]
    return dict(meta=meta, rows=rows, noise=noise, baselines=baselines, heights=fitted, shapes=shapes)
