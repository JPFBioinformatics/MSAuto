# region Imports
import subprocess, base64, zlib, sys
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET

# location of pipeline root dir
root_dir = Path(__file__).resolve().parent.parent
# tell python to look here for modules
sys.path.insert(0, str(root_dir))

from src.mzml_processor import MzMLProcessor
from src.intensity_matrix import IntensityMatrix
from src.config_loader import ConfigLoader
from src.utils import log_subprocess

# endregion

sim_dir = Path(r"C:\Jack\Projects\Danica SD 2025\11_25_25 Plasma\Sims")
sim_path = sim_dir / "Danica SIM Sample 1.D"
scan_dir = Path(r"C:\Jack\Projects\Danica SD 2025\11_25_25 Plasma\Scans")
scan_path = scan_dir / "Danica Scan Sample 1 .D"

def main():
    cfg = ConfigLoader("config.yaml")

    mp = MzMLProcessor(cfg)

    mp.full_bulk_convert()


if __name__ ==  "__main__":
    main()