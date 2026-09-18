"""

Models noise to use for bootstrapping realistic chromatograms for validation/benchmarking
of analytical techniques and model training.

---------- Algorithm Overview ----------

SIGNAL PROCESSING

1) Use already implemented CWT + endpoint detection algorithm to differentiate between signal and noise sections
2) take each ion chromatogram's noise segments and fill in signal blank sections by connecting the nearest signal
point on each side with a line
4) for each segment of 13 scans:
    - linear fit for baseline storage
    - find the difference between fit and true signal at each point, add to a distribution for our noise model
        perhaps do abs(diff)?  don't know yet

BASELINE MODEL
1) stich together baseline segments, fill in gaps again by connecting nearset point on either side with a line
2) smooth baseline with 2 savgol passes: first is 5-poit lienar second is 7-point linear
3) save the full array to a npy file, index matched to the noise model it corresponds to

NOISE MODEL
1) take sampled differernces from baseline and store as residual list, store a censored bool mask as well
that contains a 1 in indices where the raw signal is <= abundance threshold for that segment of that matrix
(meaning it is truncated)
2) if censored.any() then submit the residuals and censored to a Regression Order Statistics (ROS) method to
impute the missing left-censored data and smooth the distribution
    Parameters:
    transform_in/transform_out      defaults to log/exp but since our data will be negative sometimes (diff
                                    from mean is theoretically centered at 0) we will make them both linear
                                    with lambda x: x
    as_array = True                 just makes it return a plain numpy array same length and order as the
                                    input residuals with uncesored entries left alone and censroed
                                    replaced with corrected values
    min_uncensored = 2              as the minimum number of uncensored values before
                                    ROS can be used to impute results, falls back to simple substitution if
                                    ROS cannot be used
    max_fraction_censored = 0.8     if <= 80% of data is censored we can use the ROS to impute, if this is 
                                    exceeded then simple substitution is used
3) 

PEAK MODEL

"""

# region Imports
from pathlib import Path
from wqio.ros import ROS
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import savgol_filter
import critband
from matplotlib.backends.backend_pdf import PdfPages

from src.scripts.helpers import (rolling_median_2d, normalize_matrix)
from src.main_pipeline.mzml_processor import (full_bulk_convert)
from src.main_pipeline.config_loader import ConfigLoader
from src.main_pipeline.intensity_matrix import IntensityMatrix as IM
from src.main_pipeline.utils import (get_app_dir)

from src.scripts.helpers import (plot_histogram, plot_table)

from pybaselines import Baseline

# endregion

# region logging

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    filename=Path(__file__).parent.parent / "logs" / "noise_model.log"
)
logger = logging.getLogger(__name__)

# endregion

# region critband zero-crossing hueristic

def _silverman_rot(array, type:str = "MAD"):
    """
    Uses silverman's rule of thumb to estimate sigma for exponentially modified valley
    method
    """

    n = len(array)
    std = np.nanstd(array)

    if type == 'IQR':
        q75, q25 = np.nanpercentile(array,[75,25])
        iqr = (q75 - q25) / 1.349
        return 0.9 * min(std, iqr) * n**(-1/5)

    else:
        mad = np.nanmedian(np.abs(array - np.nanmedian(array))) * 1.4826
        return 0.9  * min(std, mad) * n**(-1/5)
    
def _optimal_bins(array, type:str = "MAD"):
    """
    uses freedman-diaconis rule to optimize bin width for a given distribution
    """
    n = len(array)

    if type == 'IQR':
        q75, q25 = np.nanpercentile(array,[75,25])
        iqr = (q75 - q25) / 1.349
        bin_width = 2 * iqr * n**(-1/3)
    else:
        mad = np.nanmedian(np.abs(array - np.nanmedian(array))) * 1.4826
        bin_width = 2 * mad * n**(-1/3)

    edges = np.arange(array.min(), array.max() + bin_width, bin_width) 
    binned_array, bin_edges = np.histogram(array, bins=edges)
    bin_values = (bin_edges[:-1] + bin_edges[1:]) / 2

    return binned_array, bin_values, bin_edges

def otsus_method(array, bin_vals):
    """
    Uses the otsus method to determine the best thresholdfor a given distribution (1D/histogram 
    not a 2D matrix/image), only splits 2 groups
    returns the bin value to use as the threshold
    """

    t_vals = np.arange(len(array))
    p_vals = array / np.sum(array)
    u = np.sum(bin_vals * p_vals)

    bcvs = np.zeros_like(t_vals, dtype=float)
    for i,t in enumerate(t_vals):

        if i == 0 or i == len(array)-1:
            continue

        split = t+1

        p1 = p_vals[:split]
        b1 = bin_vals[:split]
        w1 = np.sum(p1)
        u1 = np.sum(b1 * p1 / w1)

        p2 = p_vals[split:]
        b2 = bin_vals[split:]
        w2 = np.sum(p2)
        u2 = np.sum(b2 * p2 / w2)

        bcvs[i] = w1*(u1-u)**2 + w2*(u2-u)**2

    max_idx = np.nanargmax(bcvs)
    return bin_vals[max_idx]

def valley_emphasis_exp(array, log_norm=True):
    """
    Uses the exponentially-modified valley emphasis method to determine the best thresholdfor a 
    given distribution (1D/histogram not a 2D matrix/image), only splits 2 groups
    returns the bin value to use as the threshold
    """
    # log normalize array
    if log_norm:
        array = np.log1p(array)

    # calculate sigma using silverman's rule of thumb
    sigma = _silverman_rot(array, type='MAD')
    array_counts, bin_vals, _ = _optimal_bins(array, type='MAD')

    p_vals = array_counts / np.sum(array_counts)
    u = np.sum(bin_vals * p_vals)

    bcvs = np.zeros_like(array_counts, dtype=float)
    for i,val in enumerate(bin_vals):

        if i == 0 or i == len(array_counts)-1:
            continue

        split = i+1

        weight = 1 - np.sum(p_vals * np.exp(-((bin_vals - bin_vals[i])**2) / (2*sigma**2)))

        p1 = p_vals[:split]
        b1 = bin_vals[:split]
        w1 = np.sum(p1)
        u1 = np.sum(b1 * p1 / w1)

        p2 = p_vals[split:]
        b2 = bin_vals[split:]
        w2 = np.sum(p2)
        u2 = np.sum(b2 * p2 / w2)

        bcvs[i] = weight*(w1*(u1-u)**2 + w2*(u2-u)**2)

    max_idx = np.nanargmax(bcvs)

    threshold = bin_vals[max_idx]
    if log_norm:
        threshold = np.expm1(threshold)
    return threshold

def critband_threshold(array, data_label, log_norm=True):
    """
    Takes a raw data array and uses critband to generate a KDE optimized for bimodality and then
    finds the threshold between the modes of the distribution and returns confidence statistics
    """
    logger.info(f"{data_label} Critband started")

    # make sure input is an array
    array = np.asarray(array)

    # log correct data
    if log_norm:
        array = np.log1p(array)

    # find critical bandwith for k=2 modal distribution
    k=2
    silverman_bandwith = critband.silverman_bandwidth(array)
    min_h = silverman_bandwith / 20
    max_h = silverman_bandwith * 10
    h_crit, success = critband.critical_bandwidth(array, k=k, h_min=min_h, h_max=max_h)
    logger.info("Critband complete")

    # check if it was successful, if not fallback to MAD-based thresholding
    if success is False:
        logger.info(f'Critband could not generate {k}-modal distribution for {data_label}')
        med = np.median(array)
        mad = np.median(np.abs(array-med)) * 1.4826
        threshold = min(med + 2 * mad, array.max())
        if h_crit == min_h:
            calc_type = 'unimodal'
        elif h_crit == max_h:
            calc_type = 'hypermodal'
        else:
            calc_type = 'critband_failure'
        logger.info('Fallback_critband calculation complete')

    # if successful then use find_trough
    else:
        threshold = critband.find_trough(array,h_crit)
        if threshold is None:
            logger.info(f"{k}-modal distributiuon calculated, find_trough failed for {data_label}")
            med = np.median(array)
            mad = np.median(np.abs(array-med)) * 1.4826
            threshold = min(med + 2 * mad, array.max())
            calc_type = 'trough_failure'
            logger.info('Fallback_trough calculation complete')
        else:
            calc_type = 'critband'
            logger.info('Critband calculation fully successful')

    # adjust threshold if needed
    if threshold is not None and log_norm:
        threshold = np.expm1(threshold)
    logger.info("Threshold Calculated")

    # calculte hueristics
    bimod_test = critband.bimodality_strength(array)
    logger.info('Bimodality test completed')

    return threshold, bimod_test, calc_type

def process_row(array, label, segment_size:int=13, step:int = 5):
    """
    fully processes a row and denotes which sections are signal and which are noise
    """
    # determine number of segments
    n_segments = len(array) // step

    # process segments
    zero_crossings, energies = [], []
    for i in range(n_segments):

        # get this segment
        start = i * step
        end = start + segment_size
        segment = array[start:end]

        # determine segment energy
        energies.append(np.sum(np.abs(segment)))

        # linear fit the segment and see how many times this fit is 'crossed'
        x = np.arange(len(segment))
        slope, intercept = np.polyfit(x,segment,1)
        fit = slope*x + intercept
        count = 0
        for j, _ in enumerate(segment):

            if j == len(segment)-1:
                continue

            if segment[j]-fit[j] > 0 and segment[j+1]-fit[j+1] < 0:
                count += 1
            elif segment[j]-fit[j] < 0 and segment[j+1]-fit[j+1] > 0:
                count += 1

        zero_crossings.append(count)

    zc_thresh, zc_bimod, zc_type = critband_threshold(zero_crossings, f'{label} zero-crossings', log_norm=False)
    energy_thresh, e_bimod, e_type = critband_threshold(energies, f'{label} energies', log_norm=True)

    data = {
        'zero_crossings':{
            'threshold': zc_thresh,
            'bimod_test': zc_bimod.strength_score,
            'type': zc_type
        },
        'energies':{
            'threshold': energy_thresh,
            'bimod_test': e_bimod.strength_score,
            'type': e_type
        }
    }
    return data

# endregion

# region airPLS w/ L-curve lambda for baseline correction

"""
L-curve used to select the labmda for airPLS baseline determination, plots how well the baseline
fits the data vs how smooth the resulting curve is by sweeping lambda from tiny to large
plotting results and traces a L-shaped curve looking for the corner

airPLS then uses an adaptive iteratively reweighted penalized least squares approach to 
determine a baseline for the plot

"""

def calculate_arr_baseline(arr, min=0, max=12, n_tests=40, min_noise=1e-6):
    """
    uses the L-curve method to determine an optimal labmda for a fit
    """
    med = np.median(arr)
    mad = np.median(np.abs(arr-med)) * 1.4826
    relmad = mad / np.abs(med + 1e-12)
    if relmad < min_noise:
        baseline = np.full_like(arr, med)
        return baseline, None, None, None, None
    
    lambdas = np.logspace(min, max, n_tests)
    residuals, roughnesses, baselines = [], [], []

    fitter = Baseline(x_data=np.arange(0,len(arr),dtype=int))
    for lam in lambdas:
        baseline,_ = fitter.airpls(arr, lam=lam)
        baselines.append(baseline)
        residuals.append(np.linalg.norm(arr-baseline))
        roughnesses.append(np.linalg.norm(np.diff(baseline,n=2)) + 1e-12)

    log_lam = np.log(lambdas)
    log_res = np.log(residuals)
    log_rough = np.log(roughnesses)

    # find L-curve corner with hansen's L-curv curvature
    eta_p = np.gradient(log_res, log_lam)
    rho_p = np.gradient(log_rough, log_lam)
    eta_pp = np.gradient(eta_p, log_lam)
    rho_pp = np.gradient(rho_p, log_lam)

    curvature = (eta_p * rho_pp - eta_pp * rho_p) / (eta_p**2 + rho_p**2)**1.5
    best_idx = np.nanargmax(np.abs(curvature))
    l_lam = lambdas[best_idx]

    # use V-curve as a check
    dist = np.sqrt(np.diff(log_res)**2 + np.diff(log_rough)**2)
    geo_mean_lam = np.sqrt(lambdas[:-1] * lambdas[1:])
    v_lam = geo_mean_lam[np.nanargmin(dist)]

    return baseline, l_lam, v_lam, relmad, curvature



# endregion

# region model production

class NoiseModel:
    def __init__(self,
                 intensity_matrix:IM,
                 model_name:str,
                 ):

        # paths
        self.out_dir = get_app_dir() / 'databases' / 'noise_models' / model_name
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # obj info
        self.model_name = model_name
        self.intensity_matrix = intensity_matrix
        self.n_rows, self.n_cols = intensity_matrix.intensity_matrix.shape
        self.n_rows -= 1

        # procesing data
        self.outlier_counts = np.full(self.n_rows, np.nan, dtype=float)
        self.outlier_locations = np.zeros(self.n_cols, dtype=int)
        self.kde_bandwiths = np.full(self.n_rows, np.nan, dtype=float)

        # calculate baseline and noise models, save them and then plot values
        for ion, i in self.intensity_matrix.ion_map.items():
            if ion == 9999:
                continue
            raw_signal = self.intensity_matrix.intensity_matrix[i]
            noise_mask = self.intensity_matrix.baseline_mask[i].astype(bool)
            at_start_idxs = self.intensity_matrix.abundance_threshold['start_idxs']
            at_vals = self.intensity_matrix.abundance_threshold['values'][i]
            _,_ = self.calculate_bl_and_noise(raw_signal, noise_mask, at_start_idxs, at_vals, ion,
                                              seg_size=13, end_window=5, outlier_threshold=3.5)

        pdf_file = self.out_dir / 'oc_h_data.pdf'
        with PdfPages(pdf_file) as pdf:
            self.plot_oc_and_h(pdf)

    def plot_oc_and_h(self, pdf):
        """
        plots a histogram and table of outliers in KDE and Outleir count values
        """
        oc_data = self.outlier_counts
        oc_clean = oc_data[~np.isnan(oc_data)]
        oc_median = np.nanmedian(oc_data)
        oc_mad = np.nanmedian(np.abs(oc_data - oc_median))
        oc_title = f"Outlier Counts\nMedian:{oc_median:.2f} | MAD:{oc_mad:.2f} | Min:{np.nanmin(oc_clean):.2f} | Max:{np.nanmax(oc_clean):.2f}"
        plot_histogram(pdf, oc_title, "Outlier Count", oc_clean, bin_size=1)

        n_bls = np.zeros(self.n_rows)
        ol_pcts = np.zeros(self.n_rows)
        for i in range(self.n_rows):
            n_bls[i] = np.sum(self.intensity_matrix.baseline_mask[i])
            ol_pcts[i] = 100 * self.outlier_counts[i] / n_bls[i] if n_bls[i] > 0 else -1
        ol_med = np.nanmedian(ol_pcts)
        ol_mad = np.median(np.abs(ol_pcts - ol_med))
        n_outliers = np.sum(self.outlier_locations)
        ol_title = f"Outlier Locations\nTotal Outliers:{n_outliers} Outlier Pct Median:{ol_med:.2f} Outlier Pct MAD:{ol_mad:.2f}"
        plot_histogram(pdf, ol_title, "Scan Idx", self.outlier_locations, bin_size=1)


        inv_map = {v:k for k,v in self.intensity_matrix.ion_map.items()}
        top10_ol_pct_idxs = np.argsort(ol_pcts)[-10:][::-1]
        top10_ol_pcts = ol_pcts[top10_ol_pct_idxs]
        top10_pct_ions = [inv_map[idx] for idx in top10_ol_pct_idxs]
        pct_labels = ['Ion', 'Outlier Pct', 'Num Noise Points']
        ol_pct_title = f"Top 10 Ions by Outlier Percent | Total Row Scans:{self.n_cols}"
        plot_table(pdf,
                   title=ol_pct_title,
                   data=[
                       [top10_pct_ions[j], f"{top10_ol_pcts[j]:.2f}", f"{n_bls[top10_ol_pct_idxs[j]]}"]
                       for j in range(len(top10_ol_pct_idxs))
                   ],
                   collabels=pct_labels)

        h_data = self.kde_bandwiths
        h_clean = h_data[~np.isnan(h_data)]
        h_nans = np.sum(np.isnan(h_data))
        h_nums = np.sum(~np.isnan(h_data))
        h_median = np.nanmedian(h_data)
        h_mad = np.nanmedian(np.abs(h_data - h_median))
        h_title = f"KDE h values NaN's:{h_nans} | Successes:{h_nums}\nMedian:{h_median:.2f} | MAD:{h_mad:.2f} | Min:{np.nanmin(h_clean):.2f} | Max:{np.nanmax(h_clean):.2f}"
        plot_histogram(pdf, h_title, "KDE Bandwithd (h)", h_clean, log=True)

        top10_h_idxs = np.argsort(h_data)[-10:][::-1]
        top10_h = h_data[top10_h_idxs]
        top10_ions = [inv_map[idx] for idx in top10_h_idxs]
        labels = ['Ion', 'Bandwidth (h)']
        plot_table(pdf,
                   title='Top 10 Ions by Bandwith Value',
                   data=[
                       [f"{ion}", f"{val:.2f}"]
                       for ion,val in zip(top10_ions, top10_h)
                   ],
                   collabels=labels)

    def calculate_bl_and_noise(self, raw_signal: np.ndarray, noise_mask: np.ndarray, at_start_idxs: list, 
                                at_vals: np.ndarray, ion: int, seg_size: int=13, end_window: int=5, 
                                outlier_threshold: float=3.5):
        """
        Produces a noise and baseline model for a given row of an intensity matrix

        Params
        ------
        raw_signal                      raw signal array
        noise_mask                      bool mask 1=noise 0=signal for raw_signal
        at_start_idxs                   start idxs for abundance thresholds
        at_vals                         values array for each sgment for abundance thresholds
        seg_size                        size of segments for baseline generation
        end_window                      how far to search for median when imputing baseline of signal sections
        outlier_threshold               threshold for median/mad based outlier-detection

        Returns
        -------
        smoothed_baseline               smoothed baseline array for this row
        corrected_residuals             corrected residuals array for this row's noise (signal - baseline)

        """
        # get the number of segments for this portion of signal
        n_segments = len(raw_signal) // seg_size
        if len(raw_signal) % seg_size != 0:
            n_segments += 1

        # remove peaks from signal and replace with linear imputed baseline values
        signal = NoiseModel._impute_missing_signal(raw_signal, noise_mask, window=end_window)

        # process each segment for noise and baseline
        start_idxs = []
        baseline = np.zeros_like(raw_signal)
        residuals = np.zeros_like(raw_signal)
        valid = np.zeros_like(raw_signal, dtype=bool)
        outliers = np.zeros_like(raw_signal, dtype=bool)
        n_outliers = 0
        for i in range(n_segments):

            # get start and end points
            start = 0 + seg_size * i
            start_idxs.append(start)
            end = start + seg_size

            # get segment info
            segment_noise_mask = noise_mask[start:end]
            segment = signal[start:end]

            # process for baseline, residuals, validity and otuliers
            seg_bl, seg_res, seg_valid, seg_out_mask = NoiseModel._process_segment(segment, segment_noise_mask, 
                                                                              threshold=outlier_threshold)
            n_outliers += np.sum(seg_out_mask)

            # update stored values
            baseline[start:end] = seg_bl
            residuals[start:end] = seg_res
            valid[start:end] = seg_valid
            outliers[start:end] = seg_out_mask

        # smooth baseline
        smoothed_baseline = NoiseModel._smooth_baseline(baseline, start_idxs, filter_1=4, filter_2=5)

        # detect row censoring
        censored = NoiseModel._detect_censored(raw_signal,
                                               abundance_thresholds=at_vals,
                                               start_idxs=at_start_idxs)

        # mask outlier scans
        valid_censored = censored[valid]
        valid_residuals = residuals[valid]
        
        if censored.any():
            df = pd.DataFrame({'residual': valid_residuals,
                              'censored': valid_censored})
            corrected_residuals = ROS(
                'residual', 'censored', df=df,
                transform_in=lambda x: x, transform_out=lambda x: x,
                min_uncensored=2, max_fraction_censored=0.8,
                as_array=True
            )
        else:
            corrected_residuals = valid_censored

        # save outler info and kde bandwith info
        row_idx = self.intensity_matrix.ion_map[ion]
        self.outlier_counts[row_idx] = n_outliers
        self.outlier_locations += outliers
        h = NoiseModel._silverman_rot(corrected_residuals)
        self.kde_bandwiths[row_idx] = h

        # return the baseline and corrected residuals
        return smoothed_baseline, corrected_residuals

    @ staticmethod
    def _detect_censored(raw_signal, abundance_thresholds, start_idxs):
        """
        Looks through a raw signal for values that = abundance threshold for that segment and marks
        them as censored (bool True or 1) for ROC imputation
        
        Params
        ------
        raw_signal                      raw signal array
        abundance_thresholds            array of 10 possible abundance thresholds
        start_idxs                      start idxs of each 10 segments corresponding to the abundance thresholds

        Returns
        -------
        censored                        bool array len(raw_signal) where True = censored False = uncensored
        """
        censored = np.zeros(len(raw_signal), dtype=bool)
        for i, signal in enumerate(raw_signal):
            seg_idx = np.searchsorted(start_idxs, i, side='right') - 1
            floor = abundance_thresholds[seg_idx]
            censored[i] = np.isclose(signal, floor)
        return censored

    @staticmethod
    def _impute_missing_signal(raw_signal:np.ndarray, noise_mask:np.ndarray, window: int=5):
        """
        Finds the medain in window points on eithe rside of a signal section of raw_signal and 
        fits a linear fit betweenthese two points, replacing the signal with this imputed value

        Params
        ------
        raw_signal                      raw signal array to process
        noise_mask                      0=signal 1=noise mask of raw_signal
        window                          how far to look from noise section when finding median

        Returns
        -------
        clean_signal                    signal with actual peaks removed and replaced with imputed baselines
        """

        clean_signal = np.copy(raw_signal)
        n = len(raw_signal)
        is_signal = ~noise_mask

        # find start/end ponits of signal sections
        diff = np.diff(is_signal.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1

        if is_signal[0]:
            starts = np.concatenate([[0],starts])
        if is_signal[-1]:
            ends = np.concatenate([ends, [n]])

        # replace signal sections with lienarly imputed value
        for start,end in zip(starts,ends):
            left_slice = slice(max(0, start-window), start)
            right_slice = slice(end, min(n, end+window))

            left_vals = raw_signal[left_slice][noise_mask[left_slice]]
            right_vals = raw_signal[right_slice][noise_mask[right_slice]]

            if len(left_vals) == 0 and len(right_vals) == 0:
                continue
            elif len(left_vals) == 0:
                clean_signal[start:end] = np.median(right_vals)
            elif len(right_vals) == 0:
                clean_signal[start:end] = np.median(left_vals)
            else:
                left_anchor = np.median(left_vals)
                right_anchor = np.median(right_vals)
                clean_signal[start:end] = np.linspace(left_anchor, right_anchor, end-start)

        return clean_signal

    @staticmethod
    def _process_segment(segment_signal, segment_noise_mask, threshold=3.5):
        """
        takes a segment of signal and generates a local linear fit to it and calculates differences between
        true signal and the fit (residual)
        
        Params
        ------
        segment_signal                  section of signal for this segment

        Returns
        -------
        baseline                        array linear fit for this segment
        residuals                       array of residual difference between signal and fit
        valid_mask                      bool mask of poitns which are noise and not outliers for filtering 
                                        (true means include in noise model)
        n_outliers                      number of outliers detected in the fit
        """

        baseline, outlier_mask = NoiseModel._local_linear_fit(
            segment_signal, outlier_detection=True, threshold=threshold
            )
        residuals = segment_signal - baseline
        valid_mask = segment_noise_mask & ~outlier_mask

        return baseline, residuals, valid_mask, outlier_mask

    @ staticmethod
    def _smooth_baseline(baseline, start_idxs, filter_1=4, filter_2=5):
        """
        Smoothes a given baseline fist using a filter_1-scan local linear fit around each start_idx and then a global 
        filter_2-scan savgol filter and returns the smoothted baseline array

        Params
        ------
        baseline                        array of bseline values
        start_idxs                      start idex of segements in baseline
        filter_1/filter_2               size of filter for smoothing pass 1 and 2
                                        filter_1 should be even filter_2 should be odd

        Returns
        -------
        smoothed_baseline               smoothetd baseline array
        """
        # copy of basline for smoothing
        smoothed_baseline = np.copy(baseline)

        # make sure filter_1 is even
        if filter_1 % 2 != 0:
            filter_1 += 1

        # make sure filter_2 is odd
        if filter_2 % 2 == 0:
            filter_2 += 1

        for start_idx in start_idxs:

            # flags for starting and ending segments
            start = False
            end = False

            # get window on each side for the corner-smoothing
            window = filter_1 // 2
            segment = baseline[max(0,start_idx-window):start_idx+window]

            # pad segment if it is at the start or end
            if start_idx - window < 0:
                start_diff = abs(start_idx - window)
                pad = np.full(start_diff, baseline[0])
                segment = np.concatenate([pad, segment])
                start = True
            elif start_idx + window > len(baseline):
                end_diff = abs(start_idx + window - len(baseline))
                pad = np.full(end_diff, baseline[-1])
                segment = np.concatenate([segment,pad])
                end = True

            # generate local linear fit
            fit, _ = NoiseModel._local_linear_fit(segment, outlier_detection=False)

            # replace baseline fit with the adjusted fit
            if start:
                smoothed_baseline[:start_idx+window] = fit[start_diff:]
            elif end:
                smoothed_baseline[start_idx-window:start_idx+window-end_diff] = fit[:filter_1-end_diff]
            else:
                smoothed_baseline[start_idx-window:start_idx+window] = fit

        # final filter_2 savgol order=1 filter pass over entire baseline 
        smoothed_baseline = savgol_filter(smoothed_baseline, window_length=filter_2, polyorder=1)
        return smoothed_baseline

    @staticmethod
    def _local_linear_fit(y, outlier_detection=True, threshold=3.5):
        """
        Generates a local linear fit of a given y array using closed form least-squares linear regression on
        evenly spaced points, also rejects outlier points to cover in case of a missed parital peak at the start
        or end of the run
        """

        n = len(y)
        x = np.arange(n)

        if outlier_detection:
            outlier_mask = NoiseModel._detect_outliers(y, threshold=threshold)
            fit_y = y[~outlier_mask]
            fit_x = x[~outlier_mask]
            if len(fit_y) < 2:
                fit_x, fit_y = x,y
                outlier_mask = np.zeros(n, dtype=bool)
        else:
            fit_x, fit_y = x,y
            outlier_mask = np.zeros(n, dtype=bool)

        x_mean = fit_x.mean()
        y_mean = fit_y.mean()

        slope = np.sum((fit_x - x_mean) * (fit_y-y_mean)) / np.sum((fit_x-x_mean)**2)
        intercept = y_mean - slope * x_mean

        return slope * x + intercept, outlier_mask

    @staticmethod
    def _detect_outliers(y, threshold=3.5):
        """
        uses modz score to detect outliers in an array, returns a mask of outlier location (outlier = True)
        to exclude these points from linear fits etc...
        """
        median = np.nanmedian(y)
        mad = np.median(np.abs(y-median))
        if mad == 0:
            return np.zeros(len(y), dtype=bool)
        mod_z = 0.6745 * (y-median) / mad
        return np.abs(mod_z) > threshold

    @staticmethod
    def _silverman_rot(array, type:str = "MAD"):
        """
        Uses silverman's rule of thumb to estimate sigma for exponentially modified valley
        method
        """

        n = len(array)
        std = np.nanstd(array)

        if type == 'IQR':
            q75, q25 = np.nanpercentile(array,[75,25])
            iqr = (q75 - q25) / 1.349
            return 0.9 * min(std, iqr) * n**(-1/5)

        else:
            mad = np.nanmedian(np.abs(array - np.nanmedian(array))) * 1.4826
            return 0.9  * min(std, mad) * n**(-1/5)
        

# endregion