"""
sweeps various noise model parameters and writes reports so we can figure out good
metrics 
"""

import itertools, time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.data_generator.noise_model import NoiseModel
from src.data_generator.sample_cache import prepare_samples
from src.main_pipeline.utils import get_app_dir, get_noise_model_dir
from src.main_pipeline.utils import get_log_dir
from src.data_generator.sample_cache import _cfg_hash

import logging

logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(get_log_dir() / "nm_sweep.log"),
                  logging.StreamHandler()],
    )

QC_GRID = [dict(tail_max=t, missed_max=m) for t in (1.5, 2.0, 3.0) for m in (0.05, 0.10, 0.20)]

def plot_sweep(df, pdf_path):
    with PdfPages(pdf_path) as pdf:
        # cost (noise removed) vs benefit (tail symmetry), one marker per setting
        fig, ax = plt.subplots(figsize=(8, 6))
        for (run, grow), g in df.groupby(['min_run', 'grow']):
            g = g.sort_values('k')
            ax.plot(g['med_frac_missed'], g['med_tail_post'], marker='o', label=f'min_run={run}, grow={grow}')
            for _, r in g.iterrows():
                ax.annotate(f"k={r['k']}", (r['med_frac_missed'], r['med_tail_post']), fontsize=7)
        ax.axhline(1.0, color='k', ls='--', lw=1)
        ax.set_xlabel('Median fraction of noise flagged (cost)')
        ax.set_ylabel('Median tail ratio after filter (1 = symmetric)')
        ax.set_title('Missed-peak filter sweep'); ax.legend(fontsize=7)
        pdf.savefig(fig); plt.close(fig)

        # does the filter shrink real noise?
        fig, ax = plt.subplots(figsize=(8, 4))
        for (run, grow), g in df.groupby(['min_run', 'grow']):
            g = g.sort_values('k')
            ax.plot(g['k'], g['med_sigma_ratio'], marker='o', label=f'min_run={run}, grow={grow}')
        ax.axhline(1.0, color='k', ls='--', lw=1)
        ax.set_xlabel('k'); ax.set_ylabel('Median sigma after / before')
        ax.set_title('Noise sigma change from filtering (well below 1 = cutting real noise)'); ax.legend(fontsize=7)
        pdf.savefig(fig); plt.close(fig)

def summarize(nm):
    rd = nm.row_df
    out = {
        'med_frac_missed': rd['frac_missed'].median(),
        'med_tail_pre': rd['tail_ratio_pre'].median(),
        'med_tail_post': rd['tail_ratio_post'].median(),
        'med_skew_post': rd['skew_post'].median(),
        'med_sigma_ratio': (rd['sigma_post'] / rd['sigma_pre']).median(),
    }

    # % of rows passing QC for each QC-limit combo, under this filter setting
    for q in QC_GRID:
        labels, _ = NoiseModel.qc_labels(rd, **q)
        out[f"ok_t{q['tail_max']}_m{q['missed_max']}"] = (labels == 'ok').mean()
    return out

def cached_samples(cache_root, cfg_path=None, max_files=None):
    """cache dirs that finished building, optionally only ones built with this config"""
    suffix = f"_{_cfg_hash(cfg_path)}" if cfg_path else ""
    dirs = sorted(d for d in Path(cache_root).iterdir()
                  if (d / 'done.flag').exists() and d.name.endswith(suffix))
    if not dirs:
        raise FileNotFoundError(f"No finished caches in {cache_root} (suffix '{suffix}')")
    return dirs[:max_files] if max_files else dirs

def get_samples(cache_root, cfg_path, mzml_dir=None, max_files=None):
    """
    mzml_dir given and present -> cached samples reused, missing ones built
    otherwise -> only already-built caches for this config
    """
    if mzml_dir and Path(mzml_dir).is_dir():
        mzml_paths = sorted(Path(mzml_dir).glob('*.mzML'))[:max_files]
        if mzml_paths:
            cache_dirs = prepare_samples(mzml_paths, cfg_path, cache_root, n_workers=2)
            logger.info(f"Samples from {mzml_dir} (cached reused, missing built): {[d.name for d in cache_dirs]}")
            return cache_dirs
        logger.warning(f"No .mzML files in {mzml_dir}, falling back to existing caches")

    cache_dirs = cached_samples(cache_root, cfg_path, max_files)
    logger.info(f"Using existing caches only: {[d.name for d in cache_dirs]}")
    return cache_dirs


def main():

    cfg_path = get_app_dir() / 'default_config.yaml'
    cache_root = get_app_dir() / 'databases' / 'noise_models' / 'cache' / 'samples'
    cache_dirs = get_samples(cache_root, cfg_path,
                             mzml_dir=r"C:\Users\jackf\MSAuto\testdata\int_subset_mzml",
                             max_files=3)


    out = get_noise_model_dir() / 'sweeps'
    out.mkdir(parents=True, exist_ok=True)

    grid = {'k': [1.5, 2.0, 2.5, 3.0], 'min_run': [2, 3, 4], 'grow': [1, 2]}
    rows = []
    combos = list(itertools.product(*grid.values()))
    for i, vals in enumerate(combos, 1):
        kw = dict(zip(grid, vals))
        logger.info(f"=== Sweep {i}/{len(combos)}: {kw} ===")

        t0 = time.perf_counter()
        name = f"sweep_k{kw['k']}_run{kw['min_run']}_g{kw['grow']}"
        
        nm = NoiseModel(cache_dirs, model_name=name, missed_kwargs=kw, fit_peaks=False, make_report=False)
        nm.row_df.to_csv(out / f'{name}_rows.csv', index=False)
        
        rows.append({**kw, **summarize(nm), 'secs': round(time.perf_counter() - t0, 1)})
        logger.info(rows[-1])
        pd.DataFrame(rows).to_csv(out / 'missed_filter_sweep.csv', index=False)

    plot_sweep(pd.DataFrame(rows), out / 'missed_filter_sweep.pdf')

if __name__ == '__main__':
    setup_logging()
    try:
        main()
    except Exception:
        logger.exception("Sweep failed")      # full traceback also goes to the log file
        raise