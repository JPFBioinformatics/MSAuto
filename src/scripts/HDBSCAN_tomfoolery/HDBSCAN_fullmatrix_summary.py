"""
Goes over a list of seeds for HDBSCAN and finds the best parameters for clustering full
dataset, plotting distributions of output data so that we can see which parameter
combinations tend to be best.  Sampling is done on all samples put together, rather than
per-ion trace which is investiagted seperately.

A couple already known conventions for our dataset:
    cluster seperation epsilon  a value above 0 is best, and any small integer gave the same
                                values so we will go with 1
    n_clusters                  we know that we don't want a lot of clusters, so any clustering
                                that results in more than 5 clusters will be included in the 
                                distributions but will be dropped for final selection
"""

# region Imports

import numpy as np
from pathlib import Path
import json
import itertools, hdbscan, contextlib
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

from src.scripts.helpers import (quantile_normalization, normalize_matrix, rolling_median_2d, 
                                 hdbscan_fullmatrix, hdbscan_gs)

# endregion

# region logging

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    filename=Path(__file__).parent / "logs" / "HDBSCAN_fullmatrix_summary.log"
)
logger = logging.getLogger(__name__)

verbose_log_path = Path(__file__).parent / "logs" / "HDBSCAN_fullmatrix_summary_parallelization.log"

# endregion

n_samples = 250_000
seeds = np.arange(1,101, dtype=int)
distance_metrics = ['euclidean', 'manhattan']
matrix_norm = ['modz', 'compressed']

param_grid = {
    'min_cluster_size': [500, 1000, 2000, 3000, 4000],
    'min_samples': [20, 50, 100, 150, 200],
    'eps': [0, 0.5, 1]
}

per_row = False

include_cwt_data = False

knn_info = False

threshold = False

root_dir = Path(__file__).resolve().parents[2]
out_path = root_dir / 'peak_metrics_data'

if __name__ == '__main__':

    logger.info(f"\n---------- Began HDBSCAN_fullmatrix_summary ----------\nn_samples={n_samples}\n"
                f"distance_metrics={distance_metrics}\nmatrix_norm={matrix_norm}\nper_row={per_row}\n"
                f"include_cwt_data={include_cwt_data}\nknn_info={knn_info}\nthreshold={threshold}\n"
                f"out_path={out_path}\nparam_grid={param_grid}\nseeds={seeds}\n")

    if threshold:
             
        # get input path
        json_path = Path(f'im_data_threshold.json')

        # build output path
        if include_cwt_data:
            out_path = out_path / '5_embedding'
        else:
            out_path = out_path / '3_embedding'

        out_path = out_path / 'threshold_true'

        if per_row:
            out_path = out_path / 'per_row'
        else:
            out_path = out_path / 'full_matrix'
        
    else:

        # get input path
        json_path = Path(f'im_data_no_threshold.json')

        # build output path
        if include_cwt_data:
            out_path = out_path / '5_embedding'
        else:
            out_path = out_path / '3_embedding'

        out_path = out_path / 'threshold_false'

        if per_row:
            out_path = out_path / 'per_row'
        else:
            out_path = out_path / 'full_matrix'

    # region data loading

    if not json_path.exists():
        raise ValueError('No json data found')

    with open(json_path, 'r') as f:
        data = json.load(f)

    unique_mzs = data['mzs']
    logger.info("Data Loaded")

    # endregion
    
    for norm_type in matrix_norm:

        # normalize 3-embedding matrices
        first_derivs = np.array(data['first_derivs'], dtype=float)
        fd_trend = rolling_median_2d(first_derivs, window=51)
        first_derivs_norm = normalize_matrix(first_derivs-fd_trend, norm_method=norm_type)

        second_derivs = np.array(data['second_derivs'], dtype=float)
        second_derivs_norm = normalize_matrix(second_derivs, norm_method=norm_type)

        smoothed_signal = np.array(data['smoothed_signal'], dtype=float)
        sm_trend = rolling_median_2d(smoothed_signal, window=51)
        smoothed_signal_norm = normalize_matrix(smoothed_signal-sm_trend, norm_method=norm_type)
        logger.info(f"FD, SD, SS matrices normalized for norm_type={norm_type}")

        # Normalize CWT matrices if needed and stack to produce X
        if include_cwt_data:

            cwt_max_scores = np.array(data['cwt_max_scores'], dtype=float)
            cwt_max_scores_norm = quantile_normalization(cwt_max_scores)

            cwt_max_scales = np.array(data['cwt_max_scales'], dtype=float)
            cwt_max_scales_norm = normalize_matrix(cwt_max_scales, norm_method=norm_type)
            X = np.column_stack([
                first_derivs_norm.ravel(),
                second_derivs_norm.ravel(),
                smoothed_signal_norm.ravel(),
                cwt_max_scales_norm.ravel(),
                cwt_max_scores_norm.ravel()
            ])
            names = ['First Derivatives', 'Second Derivatives', 'Smoothed Signal', 'CWT Scales', 'CWT Scores']
            logger.info(f"CWT matrices normalized")
        else:
            X = np.column_stack([
                first_derivs_norm.ravel(),
                second_derivs_norm.ravel(),
                smoothed_signal_norm.ravel(),
            ])
            names = ['First Derivatives', 'Second Derivatives', 'Smoothed Signal']
        logger.info(f'X matrix generated norm_type={norm_type} for features:\n{names}')

        # run clustering per distance metric parrallelized by seed
        for distance_metric in distance_metrics:


            # run gs hdbscan parallelized
            with open(verbose_log_path, 'a') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                per_seed_row_results = Parallel(n_jobs=4, verbose=10)(
                    delayed(hdbscan_gs)(X, param_grid, distance_metric, n_samples=n_samples, ion=None, seed=seed)
                    for seed in seeds
                )
            gs_results = [row for seed_result in per_seed_row_results for row in seed_result]
            for seed in seeds:
                gs_results.extend(hdbscan_gs(X, param_grid, distance_metric, n_samples=n_samples, ion=None, seed=seed))
            logger.info(f"HDBSCAN complete for all seeds norm_type={norm_type};distance_metric={distance_metric}")

            # convert to matrix
            results_matrix = np.array(gs_results)
            logger.info(f"Results matrix generated")

            # produce pdf output
            file_name = f'HDBSCAN_summary_{distance_metric}_{norm_type}_fullmatrix.pdf'
            file = out_path / file_name
            with PdfPages(file) as pdf:
                hdbscan_fullmatrix(pdf, param_grid, results_matrix, top_k=5, n_clusters_max=5)
            logger.info(f"PDF generated at:\n{file}")
