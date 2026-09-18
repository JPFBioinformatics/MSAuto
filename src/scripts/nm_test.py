
from pathlib import Path

from src.data_generator.noise_model import NoiseModel as NM
from src.main_pipeline.mzml_processor import create_intensity_matrix
from src.main_pipeline.config_loader import ConfigLoader as CL
from src.main_pipeline.utils import get_app_dir

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    filename=Path(__file__).parent.parent / "logs" / "noise_model_test.log"
)
logger = logging.getLogger(__name__)

def main():
    mzml_path = Path(r"C:\Jack\Projects\IlyaAura Mouse Labelling\7_10_int1\mzML_files\Int 1.mzML")
    cfg_path = get_app_dir() / 'default_config.yaml'
    cfg = CL(cfg_path)

    im = create_intensity_matrix(mzml_path, cfg, apply_threshold=True, detect_peaks=True)

    noise_model = NM(intensity_matrix=im, model_name='first_test')

if __name__ == '__main__':
    main()