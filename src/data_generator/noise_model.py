"""

Models noise to use for bootstrapping realistic chromatograms for validation/benchmarking
of analytical techniques and model training.

---------- Algorithm Overview ----------

Each individual ion chromatogram will create its own model in order to keep noise consistent
and realistic for downstream sampling

For each chromatogram use an adapted zero-crossings-rate huristic:
    1) Split signal into 13-scan sections, ignore sections where the mean is crossed < 6 
    (scan width / 2 truncated) times then measure absolute deviation from each point to a 
    linear fit, plot as a distribution to sample
        13-scan, 6 crossings are going to be optimized
        measuring deviation from mean vs height above min is up for debate as well, depends
        on the baseline model
    2) Store distribution for later sampling

"""

# region Imports
from pathlib import Path
import numpy as np
from scipy.stats import gaussian_kde
import critband
from src.scripts.helpers import (rolling_median_2d, normalize_matrix)

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

