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
from joblib import Parallel, delayed

from src.scripts.helpers import (hdbscan_gs_saveshard)

# endregion

# region logging

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    filename=Path(__file__).parent / "logs" / "HDBSCAN_perion_summary.log"
)
logger = logging.getLogger(__name__)

# endregion

n_samples = 250_000
distance_metrics = ['euclidean', 'manhattan']
matrix_norm = ['modz', 'compressed']

param_grid = {
    'min_cluster_size': [500, 1000, 2000, 3000, 4000],
    'min_samples': [20, 50, 100, 150, 200],
    'eps': [0, 0.5, 1]
}

knn_info = False

cwts = [False, True]
thresholds = [False, True]

root_dir = Path(__file__).resolve().parents[2]

if __name__ == '__main__':

    for include_cwt_data in cwts:
        for threshold in thresholds:

            # setup out paths and logger
            out_path = root_dir / 'peak_metrics_data'
            logger.info(f"\n---------- Began HDBSCAN_perion_summary ----------\nn_samples={n_samples}\n"
                        f"distance_metrics={distance_metrics}\nmatrix_norm={matrix_norm}\nper_row=True\n"
                        f"include_cwt_data={include_cwt_data}\nknn_info={knn_info}\nthreshold={threshold}\n"
                        f"out_path={out_path}\nparam_grid={param_grid}\n")

            # region Output Path

            if include_cwt_data:
                out_path = out_path / '5_embedding'
                names = ['First Derivatives', 'Second Derivatives', 'Smoothed Signal', 'CWT Scales', 'CWT Scores']
            else:
                out_path = out_path / '3_embedding'
                names = ['First Derivatives', 'Second Derivatives', 'Smoothed Signal']

            if threshold:
                out_path = out_path / 'threshold_true'
            else:
                out_path = out_path / 'threshold_false'

            out_path = out_path / 'per_row'

            # endregion

            # build MZ map
            unique_mzs = np.load(out_path / 'precomputed' / 'unique_mzs.npy')
            mz_map = {mz:i for i,mz in enumerate(unique_mzs)}
            
            for norm_type in matrix_norm:

                # load feature stack
                feature_stack = np.load(out_path / 'precomputed' / f'X_{norm_type}.npy')
                logger.info('Data Loaded')

                # run clustering per distance metric parrallelized by seed
                for distance_metric in distance_metrics:

                    # run gs hdbscan parallelized
                    shard_dir = out_path / f'{norm_type}_{distance_metric}' / 'shards'
                    shard_dir.mkdir(parents=True, exist_ok=True)

                    # see if files already exist and skip if they do
                    existing_files = list(shard_dir.glob("*.pk1"))
                    if len(existing_files) >= len(unique_mzs):
                        logger.info(f"Skipping {norm_type}_{distance_metric}: all {len(unique_mzs)} shards already present")
                        continue

                    # parallelized processing
                    per_ion_row_results = Parallel(n_jobs=8, verbose=10)(
                        delayed(hdbscan_gs_saveshard)(
                            X=np.column_stack([fm[mz_map[ion]] for fm in feature_stack]),
                            param_grid=param_grid, distance_metric=distance_metric, out_dir=shard_dir,
                            n_samples=n_samples, ion=ion, seed=None)
                            for ion in unique_mzs
                    )
                    logger.info(f"HDBSCAN complete for all seeds norm_type={norm_type};distance_metric={distance_metric}")
    