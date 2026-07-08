"""

Class to hold the data matrix of identified peaks where each row is a sample and each column
is a different measured intensity value, used for analysis/QC

"""

# region Imports

import numpy as np
from src.config_loader import ConfigLoader
from src.db import get_run_samples, get_run_molecules, connect
from src.utils import get_run_dir, get_proj_db
from src.analysis import full_preprocess

from scipy.optimize import curve_fit

# logging
import logging
logger = logging.getLogger(__name__)

# endregion

class DataMatrix:

    def __init__(self, proj_name: str, run_name: str, peak_data: dict):

        if not peak_data:
            logger.warning("List of IntensityMatrix objects is empty")
            raise ValueError("List of IntensityMatrix objects is is empty")

        # config/attrs
        self.run_name = run_name
        self.project_name = proj_name
        rundir = get_run_dir(proj_name,run_name)
        self.cfg = ConfigLoader(rundir / 'config.yaml')
        self.db_path = get_proj_db(proj_name)
        self.run_name = run_name

        # sample/molecule tables
        conn = connect(self.db_path)
        self.samples = {r['sample_name']: r for r in get_run_samples(conn, run_name)}
        self.molecules = {r['molecule_name']: r for r in get_run_molecules(conn, run_name)}
        conn.close()

        # maps
        self.sample_map = {name: i for i,name in enumerate(self.samples)}
        self.sample_list = list(self.sample_map.keys())
        self.mol_map = {name: i for i,name in enumerate(self.molecules)}
        self.mol_list = list(self.mol_map.keys())
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

        # matrix builders
        float_empty = lambda: np.full((self.n_samples, self.n_molecules), np.nan, dtype=np.float64)
        int_empty = lambda: np.full((self.n_samples, self.n_molecules), -1, dtype=np.int32)
        bool_empty = lambda: np.zeros((self.n_samples, self.n_molecules), dtype=bool)

        # data matrices (samples x features, rows x columns)
        self.data = {
            'Area': float_empty(),
            'FWHH': float_empty(),
            'RT': float_empty(),
            'RT_Diff': float_empty(),
            'right_bound': int_empty(),
            'left_bound': int_empty(),
            'Height': float_empty(),
            'Tailing_Factor': float_empty(),
            'Sharpness': float_empty(),
            'SN_Ratio': float_empty(),
            'Theoretical_Plates': float_empty(),
            'peak_idx': int_empty(),
            'bl_slope': float_empty(),
            'flat': bool_empty(),
            'gaussian_similarity': float_empty(),
            'norm_Area': float_empty(),
            'norm_Height': float_empty(),
            'clean_Area': float_empty(),
            'clean_Height': float_empty()
        }
        
        # outlier matrices (samples x features, rows x columns), 1/True if outlier
        self.outliers = {
            'Area': bool_empty(),
            'FWHH': bool_empty(),
            'RT': bool_empty(),
            'Height': bool_empty(),
            'Tailing_Factor': bool_empty(),
            'Sharpness': bool_empty(),
            'SN_Ratio': bool_empty(),
            'Theoretical_Plates': bool_empty(),
            'bl_slope': bool_empty(),
            'gaussian_similarity': bool_empty()
        }

        # missiness matrix (samples x features, rows x columns), 1/True if missing, looks only at
        # area matrix, not other QC calculations
        self.missing = bool_empty()

        # save standards
        self.standards = []
        self.std_map = {}

        # clean molecule list/map
        self.clean_mol_list = []
        self.clean_mol_map = {}

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
                if not peak['rt_valid']:
                    continue
                
                col_i = self.mol_map[peak['molecule']]

                self.data['Area'][row_i,col_i] = peak['area']
                self.data['FWHH'][row_i,col_i] = peak['fwhh']
                self.data['RT'][row_i,col_i] = peak['rt']
                self.data['RT_Diff'][row_i,col_i] = peak['rt_diff']
                self.data['left_bound'][row_i,col_i] = peak['left_bound']
                self.data['right_bound'][row_i,col_i] = peak['right_bound']
                self.data['Height'][row_i,col_i] = peak['height']
                self.data['Tailing_Factor'][row_i,col_i] = peak['tailing_factor']
                self.data['Sharpness'][row_i,col_i] = peak['conv']
                self.data['SN_Ratio'][row_i,col_i] = peak['sn_ratio']
                self.data['Theoretical_Plates'][row_i,col_i] = self._theoretical_plates(peak['rt'],peak['fwhh'])
                self.data['peak_idx'][row_i][col_i] = peak['peak_idx']
                self.data['bl_slope'][row_i][col_i] = peak['bl_slope']
                self.data['flat'][row_i][col_i] = peak['flat_top']
                self.data['gaussian_similarity'][row_i][col_i] = self.gaussian_similarity(peak['peak_array'])

        # outlier matrix
        for metric in self.outliers:
            self.detect_outliers(metric, outlier_threshold)

        # missingness matrix
        self.missing = np.isnan(self.data['Area'])

        # normalize matrix
        self.normalize_matrix('Area')
        self.normalize_matrix('Height')

        # clean matrices
        self.preprocess_data()

    def preprocess_data(self):
        """
        Preprocesses data to clean matrices for advanced analyses (drop sparse feaures, impute nan, log2 transform,
        autoscale)
        MUST be called EVERY time the config is updated
        """
        # clean matrices
        try:
            self.data['clean_Area'], keep = full_preprocess(self.data['norm_Area'], self.missing, self.group_indices, self.cfg)
            self.data['clean_Height'], _ = full_preprocess(self.data['norm_Height'], self.missing, self.group_indices, self.cfg)
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}", exc_info=True)
            
        # rebuild clean mol map and mo list
        self.clean_mol_list = [self.mol_list[i] for i in keep]
        self.clean_mol_map = {mol:i for i,mol in enumerate(self.clean_mol_list)}

    def _theoretical_plates(self, rt: np.float64, fwhh: np.float64):
        """
        Calculates theroretical plates for a peak

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

    def detect_outliers(self, metric: str, threshold: float = 3.5):
        """
        uses median/mad based outlier detection to return a list if (i,j) index values for outliers
        with respect to a specific metric, determined by comparing feature values across all samples
        """
        matrix = self.data[metric]

        for j in range(matrix.shape[1]):
            col = matrix[:,j]
            outlier_mask = np.zeros(matrix.shape[0], dtype=bool)

            if self.group_indices:
                for indices in self.group_indices.values():
                    group_col = col[indices]
                    outlier_mask[indices] = self._is_outlier(group_col, threshold)
            else:
                outlier_mask = self._is_outlier(col, threshold)
            
            self.outliers[metric][:,j] = outlier_mask

    def _is_outlier(self, array: np.ndarray, threshold: float = 3.5):
        """
        median/MAD based outlier detection, returns boolean mask array where outliers are
        1 and normal entries are 0
        """
        if np.all(np.isnan(array)):
            return np.zeros(len(array), dtype=bool)
        
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
        calculates the average and percent coefficient of variation for a given array
        """
        if len(array) == 0:
            raise ValueError("Array is empty")
        
        avg = np.nanmean(array)
        stdev = np.nanstd(array)

        if avg == 0:
            return np.nan
        
        return (stdev / avg) * 100

    def _gaussian(self, x, amp, mu, sigma):
        return amp*np.exp(-((x-mu)**2) / (2 * sigma**2))
    
    def gaussian_similarity(self, array):
        x = np.arange(len(array))
        y = array.astype(float)

        try:
            p0 = [y.max(), np.argmax(y), len(y) / 4]
            popt, _ = curve_fit(self._gaussian, x, y, p0=p0)
            y_fit = self._gaussian(x,*popt)
            ss_res = np.sum((y - y_fit)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            if ss_tot == 0:
                return np.nan
            return 1 - (ss_res / ss_tot)
        except:
            return np.nan

    # endregion

    # region                 ---------- Normalization ----------

    def normalize_matrix(self, metric: str):
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

        # generate std column dict
        stds = set()
        for value in self.molecules.values():
            stds.add(value['std'])
        self.standards = stds

        std_vals = {}
        for std in stds:
            std_i = self.mol_map[std]
            std_vals[std] = matrix[:,std_i].copy()

        for row in self.molecules.values():
            mol = row['molecule_name']
            std = row['std']
            mol_i = self.mol_map[mol]
            matrix[:,mol_i] = matrix[:,mol_i] / std_vals[std]

        # save data
        self.data[f"norm_{metric}"] = matrix

    # endregion
