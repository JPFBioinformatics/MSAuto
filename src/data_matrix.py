"""

Class to hold the data matrix of identified peaks where each row is a sample and each column
is a different measured intensity value, used for analysis/QC

"""

# region Imports

import numpy as np
from src.config_loader import ConfigLoader
from src.db import get_run_samples, get_run_molecules, connect
from src.utils import get_app_dir

# logging
import logging
logger = logging.getLogger(__name__)

# endregion

class DataMatrix:

    def __init__(self, cfg: ConfigLoader, peak_data: dict, collection_metric: str = 'area'):

        if not peak_data:
            logger.warning("List of IntensityMatrix objects is empty")
            raise ValueError("List of IntensityMatrix objects is is empty")

        # config/attrs
        self.cfg = cfg
        run_name = cfg.get("run_name")
        project_name = cfg.get("proj_name")
        appdir = get_app_dir()
        db_path = appdir / "databases" / project_name / run_name
        self.db_path = db_path
        self.run_name = run_name

        # sample/molecule tables
        conn = connect(db_path)
        self.samples = {r['sample_name']: r for r in get_run_samples(conn)}
        self.molecules = {r['molecule_name']: r for r in get_run_molecules(conn)}
        conn.close()

        # maps
        self.sample_map = {name: i for i,name in enumerate(self.samples)}
        self.mol_map = {name: i for i,name in enumerate(self.molecules)}
        self.group_map = {key: value['group_name'] for key, value in self.samples.items()}
        self.group_indices = {}
        for sample_name, group_name in self.group_map.items():
            i = self.sample_map[sample_name]
            if group_name not in self.group_indices:
                self.group_indices[group_name] = []
            self.group_indices[group_name].append(i)

        # matrix shapes
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
            'tp': float_empty(),
            'normalized': float_empty()
        }
        
        # outlier matrices (samples x features, rows x columns), 1/True if outlier
        bool_empty = lambda: np.zeros((self.n_samples, self.n_molecules), dtype=bool)
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
        self.missing = bool_empty()

        # save standards
        self.standards = []
        self.std_map = {}

        # normalized data matrix metric
        self.collection_metric = collection_metric

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

        # normalize matrix
        self._normalize_matrix(self.collection_metric)

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

    # region                 ---------- Normalization ----------

    def _normalize_matrix(self, metric: str):
        """
        Takes a data matrix (area or height) and normalizes to norm factor and internal standards,
        standards are kept in the normalized matrix and will just be a column of all 1's
        """
        # copy data matrix
        matrix = self.data[metric].copy()

        # normalize to norm factor        
        norm_factors = np.zeros(self.n_samples)
        for name,i in self.sample_map.items():
            if self.samples[name]['norm_factor'] is None:
                norm_factors[i] = 1
            else:
                norm_factors[i] = self.samples[name]['norm_factor']
        matrix = matrix / norm_factors[:,np.newaxis]

        # generate map of molecule to standard column
        mol_std_map = {}
        for key,value in self.molecules.items():
            mol_i = self.mol_map[key]
            if value['std'] is None:
                continue
            mol_std_map[mol_i] = self.mol_map[value['std']]

        # normalize to istd
        for mol_i,std_i in mol_std_map.items():
            matrix[:,mol_i] = matrix[:,mol_i] / matrix[:,std_i]

        # ensure normalization worked correctly
        std_indices = set(mol_std_map.values())
        for std_i in std_indices:
            col = matrix[:,std_i]
            if not np.allclose(col, 1.0, atol=1e-6):
                logger.warning(f"ISTD did not normalize to 1.0, check normalization/ISTD")

        # save data
        self.data['normalized'] = matrix
        self.collection_metric = metric

    # endregion
