"""
Script to be ran in parallel for 
"""

import os, pickle
import numpy as np
from pathlib import Path
from src.scripts.helpers import hdbscan_gs

seeds = np.arange(1,101,dtype=int)
distance_metrics = ['euclidean','manhattan']
norm_types = ['modz','compressed']

param_grid = {
    'min_cluster_size': [500, 1000, 2000, 3000, 4000],
    'min_samples': [20, 50, 100, 150, 200],
    'eps': [0, 0.25, 0.5, 0.75, 1]
}

n_samples = 250_000

root_dir = Path(__file__).resolve().parents[2]
out_dir = root_dir / 'peak_metrics_data' / '3_embedding' / 'threshold_false' / 'full_matrix' / 'shards'
out_dir.mkdir(parents=True, exist_ok=True)

task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
combos = [(seed,dm,nt) for nt in norm_types for dm in distance_metrics for seed in seeds]
seed, distance_metric, norm_type = combos[task_id]

if norm_type == 'modz':
    x_file = root_dir / 'peak_metrics_data' / '3_embedding' / 'threshold_false' / 'X_modz.npy'
elif norm_type == 'compressed':
    x_file = root_dir / 'peak_metrics_data' / '3_embedding' / 'threshold_false' / 'X_compressed.npy'

X = np.load(x_file)

gs_results = hdbscan_gs(X, param_grid=param_grid, distance_metric=distance_metric, n_samples=n_samples, seed=seed)

shard_file = out_dir / f'{distance_metric}_{norm_type}_{seed}.pk1'
with open(shard_file, 'wb') as f:
    pickle.dump(gs_results, f)