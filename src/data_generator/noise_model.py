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
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from wqio.ros import ROS
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import critband
from matplotlib.backends.backend_pdf import PdfPages
from lmfit.models import GaussianModel
from collections import Counter
from scipy.special import erfc, erfcx
from lmfit import Model

from src.main_pipeline.utils import (get_noise_model_dir)

from src.scripts.helpers import (plot_histogram, plot_table)

from pybaselines import Baseline

# endregion

# region logging

import logging
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
                 cache_dirs,
                 model_name:str,
                 bic_margin: float = 2.0,
                 n_workers: int = None
                 ):

        # obj info
        self.model_name = model_name
        self.bic_margin = bic_margin
        cache_dirs = [Path(d) for d in cache_dirs]

        # per-sample info from cahce metadata
        self.samples = {}
        tasks = []
        for d in cache_dirs:
            with open (d / 'cache_meta.json') as f:
                meta = json.load(f)
            name = meta['sample']
            if name in self.samples:
                name = f"{name}_{d.name}"
            self.samples[name] = {
                'cache_dir': str(d),
                'scan_interval': meta['scan_interval'],
                'n_scans': meta['n_scans']
            }
            for ion, row in meta['ion_rows'].items():
                tasks.append({
                    'cache_dir': str(d),
                    'sample': name,
                    'ion': float(ion),
                    'row': row,
                    'n_peaks': meta['n_peaks_by_ion'][ion]
                })
        tasks.sort(key=lambda t: t['n_peaks'], reverse=True)  # process longest rows first

        # one BLAS thread per worker (st before workers start)
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'

        # fit all rows of all samples in a single pool
        n_workers = n_workers or max(1, (os.cpu_count() or 2) -1)
        logger.info(f"Began Model Fitting: {len(self.samples)} samples, {len(tasks)} rows, {n_workers} workers")
        results = []
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                                 initargs=(self.bic_margin,)) as ex:
            futures = [ex.submit(_run_ion, t) for t in tasks]
            for k,fut in enumerate(as_completed(futures), 1):
                results.append(fut.result())
                if k%50 == 0 or k == len(tasks):
                    logger.info(f"{k}/{len(tasks)} rows done")
        logger.info("Finished Model Fitting")

        # merge results
        results.sort(key=lambda r: (r['sample'], r['row']))
        self.exec_counter = Counter()
        fits = []
        self.bl_models, self.noise_models = {}, {}
        self.at_vals = {}
        max_scans = max(s['n_scans'] for s in self.samples.values())
        self.outlier_locations = np.zeros(max_scans,dtype=int)
        for r in results:
            key = (r['sample'], r['ion'])
            self.bl_models[key] = r['baseline']
            self.noise_models[key] = r['residuals']
            self.at_vals[key] = r['at_vals']
            self.outlier_locations[:len(r['outliers'])] += r['outliers']
            self.exec_counter.update(r['errors'])
            fits.extend(r['fits'])
        logger.info(f"Fit Errors:\n{self.exec_counter.most_common(10)}")

        # tables
        row_cols = ('sample', 'ion', 'row', 'n_outliers', 'kde_h', 'bl_count', 'signal_count', 'n_peaks')
        self.row_df = pd.DataFrame([{k: r[k] for k in row_cols} for r in results])
        self.row_df['model_name'] = self.model_name
        self.fits_df = pd.DataFrame(fits)
        self.fits_df['model_name'] = self.model_name

        # pooled arrays so plot_oc_and_h works unchanged
        self.outlier_counts = self.row_df['n_outliers'].to_numpy(dtype=float)
        self.kde_bandwiths = self.row_df['kde_h'].to_numpy(dtype=float)
        self.n_rows = len(self.row_df)

        # collect fit resutls (shorter name for easy typing)
        fits_df = self.fits_df

        # summary
        is_emg = fits_df['model'] == 'emg'
        is_gauss = fits_df['model'] == 'gauss'
        is_front = fits_df['fronting'] & is_emg
        logger.info(f"Fit status counts:\n{fits_df['status'].value_counts(dropna=False)}")
        logger.info(f"Model Counts:\n{pd.crosstab(fits_df['model'], fits_df['fronting'])}")
        logger.info(f"Fit Errors: {self.exec_counter.most_common(10)}")

        # prepare fit result plotting info
        df = fits_df
        is_emg = is_emg.to_numpy()
        is_gauss = is_gauss.to_numpy()
        is_front = is_front.to_numpy()
        is_tail = is_emg & ~is_front
        fit_ok = df['status'].isin(['ok', 'no_errorbars']).to_numpy()

        n_total = len(df)
        n_informed = int((df['guess_type'] == 0).sum())
        n_ok = int((df['status'] == 'ok').sum())
        n_noerr = int((df['status'] == 'no_errorbars').sum())
        n_failed = int((df['status'] == 'both_failed').sum())
        n_short = int((df['status'] == 'short').sum())
        tg_fails = int(df['tg_fail'].sum())
        sg_fails = int(df['sg_fail'].sum())

        # tail length relative to wdith, scale free shape metric (tau / sigma = 1 / (gamma * sigma))
        tau_over_sigma = 1.0 / (df['gamma'] * df['sigma'])

        # pct formatter
        def pct(k):
            return f"{100 * k / max(n_total,1):.2f}%"

        # out directory
        out_dir = get_noise_model_dir() / self.model_name
        pdf_file = out_dir / 'nm_metrics.pdf'

        with PdfPages(pdf_file) as pdf:

            # plot outlier counts and KDE results
            self.plot_oc_and_h(self.row_df['bl_count'].to_numpy(), pdf)

            # plot row information
            plot_histogram(pdf, "Baseline Points per-row", "N Points", self.row_df['bl_count'])
            plot_histogram(pdf, "Signal Points per-row", "N Points", self.row_df['signal_count'])
            plot_histogram(pdf, "Peak Count per-row", "N Peaks", self.row_df['n_peaks'])


            # fit outcome summary table
            outcome = pd.crosstab(df['status'].fillna('none'),
                                  df['model'].fillna('none'), margins=True)
            plot_table(pdf, f"Peak Fit Outcomes (n={n_total}, informed={n_informed}, "
                            f"TG_fails={tg_fails}, SG_fails={sg_fails})",
                       [[idx] + list(row) for idx, row in outcome.iterrows()],
                       ['status'] + list(outcome.columns))

            # model choice
            self._hist(pdf, f"Delta BIC (gauss - emg), >{bic_margin} = EMG chosen\n"
                            f"EMG={is_emg.sum()} ({pct(is_emg.sum())}) "
                            f"Gauss={is_gauss.sum()} ({pct(is_gauss.sum())}) "
                            f"Fronting={is_front.sum()}",
                       "Delta BIC", df['delta_bic'], symlog=True, title_fontsize=10)

            # amplitudes
            self._hist(pdf, f"Amplitudes (area), all fitted\nok={n_ok} no_errbars={n_noerr}",
                       "Amplitude", df.loc[fit_ok, 'amplitude'], log=True)

            # sigmas by model
            for mask, name in [(is_gauss, "Gaussian"), (is_emg, "EMG")]:
                s = df.loc[mask, 'sigma']
                if len(s):
                    self._hist(pdf, f"Sigma ({name}), n={len(s)}\n"
                                    f"Med={np.nanmedian(s):.2f} Max={np.nanmax(s):.2f} scans",
                               "Sigma (scans)", s, log=True)

            # gammas by direction
            for mask, name in [(is_tail, "tailing"), (is_front, "fronting")]:
                g = df.loc[mask, 'gamma']
                if len(g):
                    at_bound = int((g >= 9.99).sum() + (g <= 0.0101).sum())
                    self._hist(pdf, f"Gamma (EMG {name}), n={len(g)}\n"
                                    f"Med={np.nanmedian(g):.2f} at_bounds={at_bound}",
                               "Gamma (1/scans)", g, log=True)

            # tail shape, scale-free
            self._hist(pdf, "Tail length / width (tau/sigma), EMG only\n<0.3 ~ gaussian-like, >1 = strong tail",
                       "tau / sigma", tau_over_sigma[is_emg], log=True)

            # fit quality
            self._hist(pdf, f"Reduced Chi-Squared (chosen model)\nok={n_ok} no_errbars={n_noerr} "
                            f"both_failed={n_failed} short={n_short}",
                       "Redchi", df.loc[fit_ok, 'redchi'], log=True, title_fontsize=10)

            # where do failures come from
            self._hist(pdf, f"Peak height: fitted (n={fit_ok.sum()})", "Height",
                       df.loc[fit_ok, 'height'], log=True)
            self._hist(pdf, f"Peak height: failed/short (n={(~fit_ok).sum()})", "Height",
                       df.loc[~fit_ok, 'height'], log=True)
            self._hist(pdf, f"Window size: failed/short (n={(~fit_ok).sum()})", "N points",
                       df.loc[~fit_ok, 'n_points'], bin_size=1)
    @staticmethod
    def _hist(pdf, title, xlabel, values, **kwargs):
        """
        plot_hisogram wrapper that skips bad values
        """
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if kwargs.get('log'):
            vals = vals[vals>0]
        if len(vals) == 0:
            logger.info(f"Skipped plot '{title.splitlines()[0]}' (no data)")
            return
        plot_histogram(pdf, title, xlabel, vals, **kwargs)

    def plot_oc_and_h(self, baseline_mask, pdf):
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
            n_bls[i] = np.sum(baseline_mask[i])
            ol_pcts[i] = 100 * self.outlier_counts[i] / n_bls[i] if n_bls[i] > 0 else -1
        ol_med = np.nanmedian(ol_pcts)
        ol_mad = np.median(np.abs(ol_pcts - ol_med))
        n_outliers = np.sum(self.outlier_locations)
        ol_title = f"Outlier Locations\nTotal Outliers:{n_outliers} Outlier Pct Median:{ol_med:.2f} Outlier Pct MAD:{ol_mad:.2f}"
        plot_histogram(pdf, ol_title, "Scan Idx", self.outlier_locations, bin_size=1)

        h_data = self.kde_bandwiths
        h_clean = h_data[~np.isnan(h_data)]
        h_nans = np.sum(np.isnan(h_data))
        h_nums = np.sum(~np.isnan(h_data))
        h_median = np.nanmedian(h_data)
        h_mad = np.nanmedian(np.abs(h_data - h_median))
        h_title = f"KDE h values NaN's:{h_nans} | Successes:{h_nums}\nMedian:{h_median:.2f} | MAD:{h_mad:.2f} | Min:{np.nanmin(h_clean):.2f} | Max:{np.nanmax(h_clean):.2f}"
        plot_histogram(pdf, h_title, "KDE Bandwithd (h)", h_clean, log=True)

    def gen_bl_and_noise_models(self, raw_signal: np.ndarray, noise_mask: np.ndarray, at_start_idxs: list, 
                                at_vals: np.ndarray, ion_idx: int, seg_size: int=13, end_window: int=5, 
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
        
        if censored.any() and len(valid_residuals) >= 3:
            df = pd.DataFrame({'residual': valid_residuals,
                              'censored': valid_censored})
            corrected_residuals = ROS(
                'residual', 'censored', df=df,
                transform_in=lambda x: x, transform_out=lambda x: x,
                min_uncensored=2, max_fraction_censored=0.8,
                as_array=True
            )
        else:
            corrected_residuals = valid_residuals

        # save outler info and kde bandwith info
        h = NoiseModel._silverman_rot(corrected_residuals) if len(corrected_residuals) >= 2 else np.nan

        # return the baseline and corrected residuals
        return smoothed_baseline, corrected_residuals, n_outliers, outliers, h

    def save_data(self):
        """
        saves models to <app_dir>/noise_models/<model_name>/
                        peak_fits.csv, row_stats.csv, model_meta.json, <sample>/bl_noise.npz
        """
        out_dir = get_noise_model_dir() / self.model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # combined tables
        self.fits_df.to_csv(out_dir / 'peak_fits.csv', index=False)
        self.row_df.to_csv(out_dir / 'row_stats.csv', index=False)

        # per-sample baselines and residuals (scan counts can differ between samples)
        for name in self.samples:
            keys = [k for k in self.bl_models if k[0] == name]
            ions = np.array([k[1] for k in keys])
            residuals = [self.noise_models[k] for k in keys]
            offsets = np.concatenate([[0], np.cumsum([len(r) for r in residuals])]).astype(np.int64)
            at_start_idxs = np.load(Path(self.samples[name]['cache_dir']) / 'at_start_idxs.npy')

            sample_dir = out_dir / name
            sample_dir.mkdir(exist_ok=True)
            np.savez_compressed(
                sample_dir / 'bl_noise.npz',
                ions=ions,
                baselines=np.vstack([self.bl_models[k] for k in keys]),
                at_values=np.vstack([self.at_vals[k] for k in keys]),
                at_start_idxs=at_start_idxs,
                residuals_flat=np.concatenate(residuals) if residuals else np.array([]),
                residual_offsets=offsets,
            )

        meta = {
            'model_name': self.model_name,
            'bic_margin': self.bic_margin,
            'samples': self.samples,                      # name -> scan_interval, n_scans
            'n_rows': int(len(self.row_df)),
            'n_peaks': int(len(self.fits_df)),
        }
        with open(out_dir / 'model_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved noise model '{self.model_name}' ({len(self.samples)} samples) to {out_dir}")

    def _estimate_emg_guess(self, peak, front_thresh=1.1):
        """
        Estimates satrting paramters for exponentially modified gaussian fit of a peak
        USP Tailing factor, 1 = symmetric, > 1 = tailing, < 1 = fronting
        """

        # peak params
        height = peak['height']
        fwhh = peak['fwhh']
        tailing_factor = peak['tailing_factor']

        # amplitude guess
        A_guess = height

        # center guess
        center_guess = peak['center'] - peak['left_bound']
        
        # counters and base gamma_guess value
        sg_fail = False
        tg_fail = False

        # gaussian estiamte of sigma
        scan_interval = self.scan_interval
        fwhh_to_sigma = 2 * np.sqrt(2*np.log(2))
        sigma_guess = (fwhh / scan_interval) / fwhh_to_sigma
        if sigma_guess == 0:
            sg_fail = True

        # estimate gamma from tailing factor 
        ratio = 2 * tailing_factor - 1
        if ratio <= 0:
            tg_fail = True
            return A_guess, center_guess, sigma_guess, np.nan, tg_fail, sg_fail, False

        fronting = ratio < 1 / front_thresh
        excess_tailing = (1 / ratio - 1) if fronting else (ratio - 1)

        if excess_tailing <= 0 or sigma_guess == 0:
            tg_fail = excess_tailing == 0
            gamma_guess = np.nan
        else:
            gamma_guess = 1 / (sigma_guess * excess_tailing)

        # adjust center guess if fronting
        if fronting:
            n = peak['right_bound'] - peak['left_bound']
            center_guess = (n - 1) - center_guess

        return A_guess, center_guess, sigma_guess, gamma_guess, tg_fail, sg_fail, fronting

    def _safe_fit(self, model, y, params, x):
        """
        Fits a model returns (result, None) on sccess or (None, reason) on a failure
        """

        try:
            result = model.fit(y, params, x=x)
        except (ValueError, RuntimeError, FloatingPointError, TypeError) as e:
            return None, f"{type(e).__name__}: {str(e)[:80]}"
        if not result.success:
            return None, f"no_converge {str(result.message)[:60]}"
        return result, None

    @staticmethod
    def emg_stable(x, amplitude=1.0, center=0.0, sigma=1.0, gamma=1.0):
        """
        Fits an EMG like lmfit expgaussain but prevents exp overflow to improve performance
        """
        z = (center + gamma * sigma**2 - x) / (np.sqrt(2.0) * sigma)
        out = np.empty_like(z, dtype=float)
        pos = z >= 0

        out[pos] = np.exp(-(x[pos] - center)**2 / (2 * sigma**2)) * erfcx(z[pos])

        arg1 = gamma * (center - x[~pos] + gamma * sigma**2 / 2)
        out[~pos] = np.exp(arg1) * erfc(z[~pos])
        
        return amplitude * (gamma / 2) * out

    def gen_peak_model(self, peak: dict, raw_signal: np.ndarray, bic_margin=2.0):
        """
        Models a given peak as an exponentially modified gaussian or gaussian depending on which
        is a more accurate fit
        """

        # format outupt
        out = {
            # peak context
            'height': peak['height'],
            'ion': peak['ion'],
            'center_scan': peak['center'],
            'rt': peak['rt'],
            'n_points': int(peak['right_bound'] - peak['left_bound']) + 1,
            # fit results
            'model': None,
            'amplitude': np.nan,
            'sigma': np.nan,
            'gamma': np.nan,
            'redchi': np.nan,
            'delta_bic': np.nan,
            'fronting': False,
            # diagnostics
            'guess_type': 1,
            'tg_fail': False,
            'sg_fail': False,
            'status': None
        }

        # get peak segment
        left = int(peak['left_bound'])
        right = int(peak['right_bound'])
        signal = raw_signal[left:right+1]
        if len(signal) < 5:
            out['status'] = 'short'
            return out
        x = np.arange(len(signal), dtype=float)
        y = signal - peak['baseline']
        n = len(y)

        # metric guess usability checks
        fwhh_usable = np.isfinite(peak.get('fwhh', np.nan)) and peak['fwhh'] > 1e-9
        tf_usable = np.isfinite(peak.get('tailing_factor', np.nan))

        # make initial parameter guesses
        if fwhh_usable and tf_usable:
            A_guess, center_guess, sigma_guess, gamma_guess, tg_fail, sg_fail, fronting = self._estimate_emg_guess(peak)
            guess_type = 0
        else:
            center_guess = float(np.argmax(y))
            sigma_guess = n/6
            gamma_guess = 1.0
            A_guess = y.max()
            guess_type = 1
            tg_fail = False
            sg_fail = False
            fronting = False
        out.update(guess_type=guess_type, tg_fail=tg_fail, sg_fail=sg_fail, fronting=fronting)

        # clip guesses within bounds
        gamma_guess = np.clip(gamma_guess, 0.05, 8) if np.isfinite(gamma_guess) else 8
        sigma_guess = np.clip(sigma_guess, 0.4, 0.45*n)
        center_guess = np.clip(center_guess, 1, n-2)

        # convert a_guess from height to area
        A_guess = A_guess if np.isfinite(A_guess) else y.max()
        A_guess = max(A_guess, 1) * sigma_guess * np.sqrt(2*np.pi)

        # gaussian fit
        g_center = (n-1) - center_guess if fronting else center_guess
        g_params = self.gauss_model.make_params(amplitude=A_guess, center=g_center, sigma=sigma_guess)
        g_params['center'].set(min=0, max=n-1)
        g_params['sigma'].set(min=0.3, max=n/2)
        g_params['amplitude'].set(min=0)
        g_res, g_err = self._safe_fit(self.gauss_model, y, g_params, x)

        # EMG fit
        e_params = self.emg_model.make_params(amplitude=A_guess, center=center_guess, sigma=sigma_guess,
                                              gamma=gamma_guess)
        e_params['center'].set(min=0, max=n-1)
        e_params['sigma'].set(min=0.3, max=n/2)
        e_params['gamma'].set(min=0.01, max=10)
        e_params['amplitude'].set(min=0)
        y_fit = y[::-1] if fronting else y
        e_res, e_err = self._safe_fit(self.emg_model, y_fit, e_params, x)

        # tally features by model
        if g_err is not None:
            self.exec_counter[f"gauss | {g_err}"] += 1
        if e_err is not None:
            self.exec_counter[f"emg | {e_err}"] += 1

        # choose best model
        if g_res is None and e_res is None:
            out['status'] = 'both_failed'
            return out
        if g_res is not None and e_res is not None:
            out['delta_bic'] = g_res.bic - e_res.bic        # > 0 means EMG fits better
        if e_res is not None and (g_res is None or out['delta_bic'] > bic_margin):
            chosen = e_res
            out['model'] = 'emg'
            out['gamma'] = e_res.params['gamma'].value
        else:
            chosen = g_res
            out['model'] = 'gauss'
            out['fronting'] = False

        out['amplitude'] = chosen.params['amplitude'].value
        out['sigma'] = chosen.params['sigma'].value
        out['redchi'] = chosen.redchi
        out['status'] = 'ok' if chosen.errorbars else 'no_errorbars'

        return out
        
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
        seg_idx = np.searchsorted(start_idxs, np.arange(len(raw_signal)), side='right') - 1
        return np.isclose(raw_signal, np.asarray(abundance_thresholds)[seg_idx])

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

    @classmethod
    def _make_worker(cls, bic_margin):
        """
        lightweight instance for worker process, conifig + lmfit models only
        """
        w = cls.__new__(cls)
        w.scan_interval = None
        w.bic_margin = bic_margin
        w.emg_model = Model(cls.emg_stable)
        w.gauss_model = GaussianModel()
        w.exec_counter = Counter()
        return w

    def process_ion(self, task):
        """
        fits baseline, noise, and peak models for one ion row, returns results for merging
        """
        self.scan_interval = task['scan_interval']
        self.exec_counter = Counter()
        bl, residuals, n_outliers, outliers, h = self.gen_bl_and_noise_models(
            task['raw_signal'], task['noise_mask'], task['at_start_idxs'], task['at_vals'],
            task['row'], seg_size=13, end_window=5, outlier_threshold=3.5)
        fits = [self.gen_peak_model(p, task['raw_signal'], bic_margin=self.bic_margin)
                for p in task['peaks']]
        for f in fits:
            f['sample'] = task['sample']
        return {
            'sample': task['sample'], 'ion': task['ion'], 'row': task['row'],
            'baseline': bl, 'residuals': np.asarray(residuals, dtype=float),
            'n_outliers': n_outliers, 'outliers': outliers, 'kde_h': h,
            'fits': fits, 'errors': self.exec_counter,
            'bl_count': int(task['noise_mask'].sum()),
            'signal_count': int((~task['noise_mask']).sum()),
            'n_peaks': len(task['peaks']),
            'at_vals': np.asarray(task['at_vals'], dtype=float)
        }

    @staticmethod
    def make_peak(x, height, center, model, sigma, gamma=np.nan, fronting=False):
        """peak with a given apex height, placed so its EMG/gauss 'center' parameter sits at `center`"""
        if model == 'gauss':
            return height * np.exp(-(x - center)**2 / (2 * sigma**2))

        # EMG: unit-area curve on a fine grid -> its maximum -> rescale to the requested height
        grid = np.linspace(-6 * sigma, 6 * sigma + 10 / gamma, 4001)
        unit_max = NoiseModel.emg_stable(grid, 1.0, 0.0, sigma, gamma).max()
        xx = (center - x) if fronting else (x - center)            # mirror for fronting peaks
        return (height / unit_max) * NoiseModel.emg_stable(xx, 1.0, 0.0, sigma, gamma)

# endregion

# region parallel workers

from src.data_generator.sample_cache import load_sample

_WORKER = None

def _init_worker(bic_margin):
    """runs once per worker process"""
    global _WORKER
    _WORKER = NoiseModel._make_worker(bic_margin)

def _run_ion(task):
    """loads one row from the sample cache and fits it"""
    s = load_sample(task['cache_dir'])
    row = task['row']
    full_task = {
        **task,
        'scan_interval': s['meta']['scan_interval'],
        'raw_signal': np.array(s['intensity'][row]),
        'noise_mask': np.array(s['noise_mask'][row]),
        'at_start_idxs': s['at_start_idxs'],
        'at_vals': np.array(s['at_values'][row]),
        'peaks': s['peaks_by_ion'].get(task['ion'], []),
    }
    return _WORKER.process_ion(full_task)

# endregion