"""

Class to hold the data matrix of identified peaks where each row is a sample and each column
is a different measured intensity value

"""
# region Imports

import sys
import numpy as np
from pathlib import Path

# logging
import logging
logger = logging.getLogger(__name__)

# location of pipeline root dir
root_dir = Path(__file__).resolve().parent.parent
# tell python to look here for modules
sys.path.insert(0, str(root_dir))

from src.config_loader import ConfigLoader
from src.db import get_samples, get_molecules, connect
from src.utils import get_app_dir

# endregion

class DataMatrix:

    def __init__(self, cfg: ConfigLoader, peak_data: dict):

        if not peak_data:
            raise ValueError("List of IntensityMatrix objects is is empty")

        # config/attrs
        self.cfg = cfg
        run_name = cfg.get("run_name")
        project_name = cfg.get("proj_name")
        appdir = get_app_dir()
        db_path = appdir / "databases" / project_name / run_name
        self.db_path = db_path
        self.run_name = run_name

        conn = connect(db_path)
        self.samples = get_samples(conn)
        self.molecules = get_molecules(conn)
        conn.close()

        self.sample_map = {row['sample_name']:i for i,row in enumerate(self.samples)}
        self.mol_map = {row['molecule_name']: i for i,row in enumerate(self.molecules)}
        self.n_samples = len(self.sample_map)
        self.n_molecules = len(self.mol_map)

        # data matrices (samples x features, rows x columns)
        float_empty = lambda: np.full((self.n_samples, self.n_molecules), np.nan, dtype=np.float64)
        int_empty = lambda: np.full((self.n_samples, self.n_molecules), -1, dtype=np.int32)
        self.data = {
            'area': float_empty(),
            'fwhh': float_empty(),
            'rt': float_empty(),
            'right_bound': int_empty(),
            'left_bound': int_empty(),
            'height': float_empty(),
            'tailing': float_empty(),
            'conv': float_empty(),
            'sn': float_empty(),
            'tp': float_empty()
        }
        
        # outlier matrices (samples x features, rows x columns), 1/True if outlier
        bool_empty = lambda: np.zeros((self.n_samples, self.n_molecules), dtype=np.bool)
        self.outliers = {
            'area': bool_empty(),
            'fwhh': bool_empty(),
            'rt': bool_empty(),
            'height': bool_empty(),
            'tailing': bool_empty(),
            'conv': bool_empty(),
            'sn': bool_empty(),
            'tp': bool_empty()
        }

        # missiness matrix (samples x features, rows x columns), 1/True if missing, looks only at
        # area matrix, not other QC calculations
        self.missing = None

        # fill the matrices from peak data
        self.fill_matrices(peak_data)

    # region                 ---------- Matrix Building ----------

    def fill_matrices(self, peak_data: dict, outlier_threshold: float = 3.5):
        """
        Fills the qc calculation and feature value matrices
        """

        # data/QC metric matrices
        for name,peak_list in peak_data.items():
            row_i = self.sample_map[name]

            for peak in peak_list:
                if peak['molecule'] is None:
                    continue
                
                col_i = self.mol_map[peak['molecule']]

                self.data['area'][row_i,col_i] = peak['area']
                self.data['fwhh'][row_i,col_i] = peak['fwhh']
                self.data['rt'][row_i,col_i] = peak['rt']
                self.data['left_bound'][row_i,col_i] = peak['left_bound']
                self.data['right_bound'][row_i,col_i] = peak['right_bound']
                self.data['height'][row_i,col_i] = peak['height']
                self.data['tailing'][row_i,col_i] = peak['tailing_factor']
                self.data['conv'][row_i,col_i] = peak['conv']
                self.data['sn'][row_i,col_i] = peak['sn_ratio']
                self.data['tp'][row_i,col_i] = self._theoretical_plates(peak['rt'],peak['fwhh'])

        # outlier matrix
        for metric in self.outliers:
            self._detect_outliers(metric, outlier_threshold)

        # missingness matrix
        self.missing = np.isnan(self.data['area'])

    def _theoretical_plates(self, rt: np.float64, fwhh: np.float64):
        """
        Calculates theroretical plates for the entire data matrix and saves to tp_matrix

        Params
        ------
        rt                          retention time of peak
        fwhh                        full width at half height of peak

        Returns
        -------
        tp                          theoretical plates of the peak
        """
        if np.isnan(rt) or np.isnan(fwhh) or fwhh == 0:
            tp = np.nan
        else:
            tp = 5.545 * (rt / fwhh)**2
        return tp

    def _detect_outliers(self, metric: str, threshold: float = 3.5):
        """
        uses median/mad based outlier detection to return a list if (i,j) index values for outliers
        with respect to a specific metric, determined by comparing feature values across all samples
        """
        matrix = self.data[metric]

        for j in range(matrix.shape[1]):
            col = matrix[:,j]
            self.outliers[metric][:,j] = self._is_outlier(col, threshold)

    def _is_outlier(self, array: np.ndarray, threshold: float = 3.5):
        """
        median/MAD based outlier detection, returns boolean mask array where outliers are
        1 and normal entries are 0
        """
        median = np.nanmedian(array)
        mad = np.nanmedian(np.abs(array - median))

        if mad == 0:
            return np.zeros(len(array), dtype=bool)
        
        modz = 0.6745 * (array - median) / mad

        return np.abs(modz) > threshold

    # endregion

    # region                 ---------- QC Calculations ----------

    def _avg_err(self, array: np.ndarray):
        """
        calculates average and std error for a given distribution of values
        """
        if len(array) == 0:
            raise ValueError("Array is empty")
        
        avg = np.nanmean(array)
        stdev = np.nanstd(array)

        return avg, stdev
    
    def _pct_cv(self, array: np.ndarray):
        """
        calculates the percent coefficient of variation for a given array
        """
        if len(array) == 0:
            raise ValueError("Array is empty")
        
        avg = np.nanmean(array)
        stdev = np.nanstd(array)

        if avg == 0:
            return np.nan
        
        return (stdev / avg) * 100

    # endregion

    # region                 ---------- QC Plotting ----------



    # endregion