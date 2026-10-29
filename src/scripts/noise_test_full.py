"""
fits the full noise/baseline/peak model on every sample, saves it, then writes the report

    python -m src.scripts.nm_test
"""
import time
import logging
from pathlib import Path

import pandas as pd

from src.data_generator.noise_model import NoiseModel as NM
from src.data_generator.sample_cache import prepare_samples
from src.main_pipeline.utils import get_app_dir, get_noise_model_dir, get_log_dir

logger = logging.getLogger(__name__)

# run settings
MZML_DIR = Path(r"C:\Users\jackf\MSAuto\testdata\int_subset_mzml")
MAX_FILES = 1                  # 1 for a timing/crash test run, None for all
MODEL_NAME = f"int_full_{time.strftime('%Y%m%d')}"
BUILD_WORKERS = 2
FIT_WORKERS = None                # None = cpu_count - 1


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(get_log_dir() / f"{MODEL_NAME}.log"), logging.StreamHandler()],
    )


def main():
    cfg_path = get_app_dir() / 'default_config.yaml'
    cache_root = get_app_dir() / 'databases' / 'noise_models' / 'cache' / 'samples'

    mzml_paths = sorted(MZML_DIR.glob('*.mzML'))[:MAX_FILES]
    logger.info(f"Model '{MODEL_NAME}': {len(mzml_paths)} files from {MZML_DIR}")

    # caches, reused when already built for this config
    t0 = time.perf_counter()
    cache_dirs = prepare_samples(mzml_paths, cfg_path, cache_root, n_workers=BUILD_WORKERS)
    t1 = time.perf_counter()
    logger.info(f"STAGE cache: {(t1 - t0) / 60:.1f} min for {len(cache_dirs)} samples")

    # fit, no report yet so nothing is lost if plotting fails
    nm = NM(cache_dirs=cache_dirs, model_name=MODEL_NAME, n_workers=FIT_WORKERS, make_report=False)
    t2 = time.perf_counter()
    logger.info(f"STAGE fit: {(t2 - t1) / 60:.1f} min, {len(nm.fits_df)} peaks, {len(nm.row_df)} rows")
    logger.info(f"Row QC:\n{nm.row_df['qc'].value_counts()}")

    # save model + full object (report can be rebuilt from nm.pkl without refitting)
    nm.save_data()
    pd.to_pickle(nm, get_noise_model_dir() / MODEL_NAME / 'nm.pkl')
    t3 = time.perf_counter()
    logger.info(f"STAGE save: {t3 - t2:.0f} s")

    try:
        nm.write_report()
    except Exception:
        logger.exception("Report failed, model data already saved")
    logger.info(f"STAGE report: {(time.perf_counter() - t3) / 60:.1f} min | "
                f"TOTAL {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == '__main__':
    setup_logging()
    try:
        main()
    except Exception:
        logger.exception("Run failed")
        raise
