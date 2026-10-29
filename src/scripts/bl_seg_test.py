# src/scripts/bl_seg_test.py   (python -m src.scripts.bl_seg_test)
from pathlib import Path
from src.data_generator.noise_model import NoiseModel
from src.data_generator.sample_cache import load_sample, _cfg_hash
from src.main_pipeline.utils import get_app_dir
import numpy as np, pandas as pd, matplotlib.pyplot as plt


CACHE = get_app_dir() / 'databases' / 'noise_models' / 'cache' / 'samples'
SAMPLE = CACHE / f"Int 1_{_cfg_hash(get_app_dir() / 'default_config.yaml')}"
IONS = [100, 112, 207, 281]
SETTINGS = [dict(bl_lam =1e9, flat_len=f) for f in (100,)]

s = load_sample(SAMPLE)
fig, axes = plt.subplots(len(IONS), 1, figsize=(12, 3 * len(IONS)), squeeze=False)
for ax, ion in zip(axes[:, 0], IONS):
    row = s['meta']['ion_rows'][str(ion)]
    raw = np.asarray(s['intensity'][row], float)
    nm = np.asarray(s['noise_mask'][row], bool)
    ax.plot(np.where(nm, raw, np.nan), lw=0.3, color='0.7', label='noise pts')
    for kw in SETTINGS:
        w = NoiseModel._make_worker(bic_margin=2.0, bl_noise_kwargs={**NoiseModel.BL_NOISE_KWARGS, **kw})
        bl = w.gen_bl_and_noise_models(raw, nm, s['at_start_idxs'], np.asarray(s['at_values'][row]),
                                       row, **w.bl_noise_kwargs)[0]
        res = np.where(nm, raw - bl, np.nan)
        res[:25] = np.nan; res[-25:] = np.nan                 # match edge_trim
        sigma = 1.4826 * np.nanmedian(np.abs(res - np.nanmedian(res)))
        rough = np.median(np.abs(np.diff(bl, 2))) / sigma                     # wiggle, in noise units
        bias = np.nanmax(np.abs(pd.Series(res).rolling(301, center=True, min_periods=50).median())) / sigma
        ax.plot(bl, lw=1, label=f"flat_len={kw['flat_len']}  rough={rough:.2g}  bias={bias:.2f}σ")

    ax.set_title(f'm/z {ion:g}'); ax.legend(fontsize=7)
    lo, hi = np.nanpercentile(np.where(nm, raw, np.nan), [1, 99]); ax.set_ylim(lo, hi)
fig.tight_layout(); plt.show()
