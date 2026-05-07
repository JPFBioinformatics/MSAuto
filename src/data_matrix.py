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

        if not peak_dict:
            raise ValueError("peak_dict is empty")
        
        self.sample_map = {name:i for i,name in enumerate(peak_dict.keys())}
        self.mol_map = {peak["molecule"]: i for i,peak in enumerate(next(iter(peak_dict.values())))}
        self.n_samples = len(self.sample_map)
        self.n_molecules = len(self.mol_map)

        self.cfg = cfg

        # all matrices are samples x features (rows x columns)
        empty = lambda: np.full(self.n_samples, self.n_molecules)
        self.intensity_matrix = empty()                                     # peak areas
        self.rt_matrix = empty()                                            # rt/precise maximization time
        self.bounds_matrix = empty()                                        # (left_bound, right_bound)
        self.height_matrix = empty()                                        # baseline corrected height
        self.symmetry_matrix = empty()                                      # bound symmetry
        self.conv_matrix = empty()                                          # convolution value


    def _fill_matrices(peak_dict: dict):
        """
        Fills the qc calculation and feature value matrices
        """
        