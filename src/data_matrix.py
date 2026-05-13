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
from src.db import get_samples, get_molecules, connect

# endregion

class DataMatrix:

    def __init__(self, cfg: ConfigLoader, peak_data: dict, db_path: Path):

        if not peak_data:
            raise ValueError("List of IntensityMatrix objects is is empty")
        
        conn = connect(db_path)
        self.samples = get_samples(conn)
        self.molecules = get_molecules(conn)
        self.sample_map = {row['sample_name']:i for i,row in enumerate(self.samples)}
        self.mol_map = {row['molecule_name']: i for i,row in enumerate(self.molecules)}
        self.n_samples = len(self.sample_map)
        self.n_molecules = len(self.mol_map)
        self.cfg = cfg

        # all matrices are samples x features (rows x columns)
        empty = lambda: np.full(self.n_samples, self.n_molecules)
        self.area_matrix = empty()                                          # peak areas
        self.fwhh_matrix = empty()                                          # full width at half height
        self.rt_matrix = empty()                                            # rt/precise maximization time
        self.bounds_matrix = empty()                                        # (left_bound, right_bound)
        self.height_matrix = empty()                                        # baseline corrected height
        self.tailing_matrix = empty()                                      # tailing factor matrix
        self.conv_matrix = empty()                                          # convolution value
        self.sn_matrix = empty()                                            # signal to noise values

        self._fill_matrices(peak_data)

    def _fill_matrices(self, peak_data: dict):
        """
        Fills the qc calculation and feature value matrices
        """

        for name,peak_list in peak_data.items():
            row_i = self.sample_map[name]

            for peak in peak_list:
                if peak['molecule'] is None:
                    raise ValueError(f"Sample {name} has no valid peaks collected")
                
                col_i = self.mol_map[peak['molecule']]

                self.area_matrix[row_i,col_i] = peak['area']
                self.fwhh_matrix[row_i,col_i] = peak['fwhh']
                self.rt_matrix[row_i,col_i] = peak['rt']
                self.bounds_matrix[row_i,col_i] = (peak['left_bound'],peak['right_bound'])
                self.height_matrix[row_i,col_i] = peak['height']
                self.tailing_matrix[row_i,col_i] = peak['tailing_factor']
                self.conv_matrix[row_i,col_i] = peak['conv']
                self.sn_matrix[row_i,col_i] = peak['sn_ratio']
