
import numpy as np
from pathlib import Path
from datetime import datetime
from src.main_pipeline.mzml_processor import full_bulk_convert
from src.main_pipeline.config_loader import ConfigLoader
import json
from matplotlib import pyplot as plt

from src.main_pipeline.intensity_matrix import IntensityMatrix as IM

from src.data_generator.noise_model import (calculate_arr_baseline)
from src.main_pipeline.utils import sanitize_name

# region logging

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    filename=Path(__file__).parent.parent / "logs" / "gen_ims.log"
)
logger = logging.getLogger(__name__)

# endregion

if __name__ == '__main__':

    file_dir = Path(r"C:\Jack\Projects\IlyaAura Mouse Labelling\7_10_26 Intestine Subset\mzML_files")
    out_file = Path(r"C:\Jack\code\GCMS_Automation\databases\test_matrices\10_test_matrices.npz")

    cfg_path = Path(__file__).parent.parent / 'config.yaml'
    cfg = ConfigLoader(cfg_path)
    peak_mode = cfg.get('peak_mode')

    starttime = datetime.now()

    ims = full_bulk_convert(input_dir=file_dir, file_type='.mzML', cfg=cfg, serial=True, detect_peaks=False)

    matrices_dict = {}
    for i,im in enumerate(ims):
        matrices_dict[sanitize_name(im.sample_name)] = im.intensity_matrix

    np.savez(out_file, **matrices_dict)
