"""

Class to hold the data matrix of identified peaks where each row is a sample and each column
is a different measured intensity value

"""
# region Imports

import sys
import numpy as np
from pathlib import Path

# location of pipeline root dir
root_dir = Path(__file__).resolve().parent.parent
# tell python to look here for modules
sys.path.insert(0, str(root_dir))

from src.config_loader import ConfigLoader

# endregion

class DataMatrix:

    def __init__(self, cfg: ConfigLoader, peak_dict: dict):
        