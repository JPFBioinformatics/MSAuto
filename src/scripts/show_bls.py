"""
overlays the noise-model baseline from every sample for each ion, scaled 0-1, for visual QC
of baseline shape consistency across samples

    python -m src.scripts.plot_baselines
"""
import logging
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.data_generator.noise_model import NoiseModel
from src.data_generator.sample_cache import prepare_samples, load_sample
from src.main_pipeline.utils import get_app_dir, get_noise_model_dir, get_log_dir

logger = logging.getLogger(__name__)

MZML_DIR = Path(r"C:\Users\jackf\MSAuto\testdata\int_subset_mzml")
IONS_PER_PAGE = 6           # 3 x 2 grid
STEP = 4                    # plot every 4th scan, baselines are smooth so nothing is lost
NORMALIZE = True            # False = raw intensity on a log axis instead


def row_baselines(cache_dirs):
    """
    {ion: {sample: baseline}} using the same baseline code the noise model uses
    """
    w = NoiseModel._make_worker(bic_margin=2.0)
    out = {}
    for d in cache_dirs:
        s = load_sample(d)
        name = s['meta']['sample']
        for ion_str, row in s['meta']['ion_rows'].items():
            raw = np.asarray(s['intensity'][row], dtype=float)
            noise_mask = np.asarray(s['noise_mask'][row], dtype=bool)
            if noise_mask.sum() < 20:                       # nothing to fit a baseline to
                continue
            bl = w.gen_bl_and_noise_models(raw, noise_mask, s['at_start_idxs'],
                                           np.asarray(s['at_values'][row]), row,
                                           **w.bl_noise_kwargs)[0]
            out.setdefault(float(ion_str), {})[name] = bl
        logger.info(f"Baselines done: {name}")
    return out


def sample_key(name):
    """sort 'Int 2' before 'Int 10'"""
    tail = name.split()[-1]
    return (0, int(tail)) if tail.isdigit() else (1, name)


def plot(baselines, pdf_path):
    samples = sorted({s for per_ion in baselines.values() for s in per_ion}, key=sample_key)
    colors = {s: plt.cm.tab10(i % 10) for i, s in enumerate(samples)}
    ions = sorted(baselines)

    with PdfPages(pdf_path) as pdf:
        for p in range(0, len(ions), IONS_PER_PAGE):
            page_ions = ions[p:p + IONS_PER_PAGE]
            fig, axes = plt.subplots(3, 2, figsize=(11, 8.5), squeeze=False)
            for ax, ion in zip(axes.flat, page_ions):
                lo_all, hi_all = np.inf, -np.inf
                for s in sorted(baselines[ion], key=sample_key):
                    bl = baselines[ion][s]
                    lo, hi = bl.min(), bl.max()
                    lo_all, hi_all = min(lo_all, lo), max(hi_all, hi)
                    y = ((bl - lo) / (hi - lo) if hi > lo else np.zeros_like(bl)) if NORMALIZE else bl
                    ax.plot(np.arange(0, len(bl), STEP), y[::STEP], lw=0.8, color=colors[s], rasterized=True)
                if NORMALIZE:
                    ax.set_ylim(-0.05, 1.05)
                else:
                    ax.set_yscale('log')
                ax.set_title(f"m/z {ion:g}  ({len(baselines[ion])}/{len(samples)} samples)  "
                             f"range {lo_all:.3g}-{hi_all:.3g}", fontsize=8)
                ax.tick_params(labelsize=7)
            for ax in axes.flat[len(page_ions):]:
                ax.set_axis_off()
            handles = [plt.Line2D([], [], color=colors[s], label=s) for s in samples]
            fig.legend(handles=handles, loc='lower center', ncol=min(len(samples), 10), fontsize=7)
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            pdf.savefig(fig); plt.close(fig)


def main():
    cfg_path = get_app_dir() / 'default_config.yaml'
    cache_root = get_app_dir() / 'databases' / 'noise_models' / 'cache' / 'samples'
    cache_dirs = prepare_samples(sorted(MZML_DIR.glob('*.mzML')), cfg_path, cache_root, n_workers=2)

    baselines = row_baselines(cache_dirs)
    out = get_noise_model_dir() / ('baseline_overlays_norm.pdf' if NORMALIZE else 'baseline_overlays_raw.pdf')
    plot(baselines, out)
    logger.info(f"Wrote {len(baselines)} ions to {out}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(get_log_dir() / 'plot_baselines.log'), logging.StreamHandler()])
    main()
