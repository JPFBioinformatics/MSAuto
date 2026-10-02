
from pathlib import Path

import time

from src.data_generator.noise_model import NoiseModel as NM
from src.main_pipeline.utils import get_app_dir
from src.data_generator.sample_cache import prepare_samples

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    filename=Path(__file__).parent.parent / "logs" / "noise_model_test.log"
)
logger = logging.getLogger(__name__)

def main():

    # run settings
    mzml_dir = Path(r"C:\Jack\Projects\IlyaAura Mouse Labelling\7_10_26 Intestine Subset\mzML_files")
    max_files = 1
    model_name = 'intestine_single_test'
    build_workers = 2
    fit_workers = None

    cfg_path = get_app_dir() / 'default_config.yaml'
    cache_root = get_app_dir() / 'databases' / 'noise_models' / 'cache' / 'samples'

    mzml_paths = sorted(mzml_dir.glob('*.mzML'))
    if max_files is not None:
        mzml_paths = mzml_paths[:max_files]
    logger.info(f"Running {len(mzml_paths)} files from {mzml_dir}")

    t0 = time.perf_counter()
    cache_dirs = prepare_samples(mzml_paths, cfg_path, cache_root, n_workers=build_workers)
    t1 = time.perf_counter()
    logger.info(f"STAGE build/cache: {(t1 - t0) / 60:.1f} min for {len(cache_dirs)} samples")

    nm = NM(cache_dirs=cache_dirs, model_name=model_name, n_workers=fit_workers)
    t2 = time.perf_counter()
    logger.info(f"STAGE fit + plots: {(t2 - t1) / 60:.1f} min, {len(nm.fits_df)} peaks, {len(nm.row_df)} rows")

    nm.save_data()
    logger.info(f"STAGE save: {(time.perf_counter() - t2):.0f} s | TOTAL {(time.perf_counter() - t0) / 60:.1f} min")

if __name__ == '__main__':
    try:
        main()
    except Exception:
        logger.exception("Run failed")
        raise