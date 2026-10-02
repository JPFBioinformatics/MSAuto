"""
Builds intensity matrices once and caches only what noise model fitting needs
"""
import json
from pathlib import Path
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src.main_pipeline.mzml_processor import create_intensity_matrix
from src.main_pipeline.config_loader import ConfigLoader

import hashlib
import logging

logger = logging.getLogger(__name__)

PEAK_KEYS = ('ion', 'center', 'left_bound', 'right_bound', 'rt', 'height',
             'baseline', 'fwhh', 'tailing_factor')


def _cfg_hash(cfg_path):
    return hashlib.md5(Path(cfg_path).read_bytes()).hexdigest()[:8]

def prepare_sample(mzml_path, cfg_path, cache_root):
    """
    builds an IM for one mzML, writes only what model fitting needs, returns the cache dir
    skips the build if a cache already exists
    """
    mzml_path = Path(mzml_path)
    cache_dir = Path(cache_root) / f"{mzml_path.stem}_{_cfg_hash(cfg_path)}"
    if (cache_dir / 'done.flag').exists():
        return cache_dir

    cfg = ConfigLoader(cfg_path)
    im = create_intensity_matrix(mzml_path, cfg, apply_threshold=True, detect_peaks=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # row data as plain .npy so stage 2 can memory-map single rows
    np.save(cache_dir / 'intensity.npy', im.intensity_matrix)
    np.save(cache_dir / 'noise_mask.npy', im.baseline_mask.astype(bool))
    np.save(cache_dir / 'at_values.npy', im.abundance_threshold['values'])
    np.save(cache_dir / 'at_start_idxs.npy', np.asarray(im.abundance_threshold['start_idxs']))

    # slim peaks, one table for the whole sample
    peaks = [{k: p[k] for k in PEAK_KEYS} for plist in im.peak_dict.values() for p in plist]
    pd.to_pickle(pd.DataFrame(peaks, columns=PEAK_KEYS), cache_dir / 'peaks.pkl')

    # small metadata
    times = np.array([t for _, t in sorted(im.time_map.items())])
    meta = {
        'sample': im.sample_name or mzml_path.stem,
        'scan_interval': float(np.median(np.diff(times))),
        'n_scans': int(im.intensity_matrix.shape[1]),
        'ion_rows': {str(ion): int(i) for ion, i in im.ion_map.items() if ion != 9999},
        'n_peaks_by_ion': {str(ion): len(im.peak_dict.get(ion, [])) for ion in im.ion_map if ion != 9999},

    }
    with open(cache_dir / 'cache_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    (cache_dir / 'done.flag').touch()
    return cache_dir

def prepare_samples(mzml_paths, cfg_path, cache_root, n_workers=2):
    """builds/caches all samples in parallel, returns cache dirs in input order"""
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        return list(ex.map(prepare_sample, mzml_paths, repeat(cfg_path), repeat(cache_root)))

_SAMPLE_CACHE = {}

def load_sample(cache_dir):
    """memory-mapped arrays + peaks for one sample, cached per process"""
    cache_dir = str(cache_dir)
    if cache_dir not in _SAMPLE_CACHE:
        d = Path(cache_dir)
        with open(d / 'cache_meta.json') as f:
            meta = json.load(f)
        peaks = pd.read_pickle(d / 'peaks.pkl')
        _SAMPLE_CACHE[cache_dir] = {
            'meta': meta,
            'intensity': np.load(d / 'intensity.npy', mmap_mode='r'),
            'noise_mask': np.load(d / 'noise_mask.npy', mmap_mode='r'),
            'at_values': np.load(d / 'at_values.npy', mmap_mode='r'),
            'at_start_idxs': np.load(d / 'at_start_idxs.npy'),
            'peaks_by_ion': {ion: g.to_dict('records') for ion, g in peaks.groupby('ion')},
        }
    return _SAMPLE_CACHE[cache_dir]

