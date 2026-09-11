import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from src.main_pipeline.config_loader import ConfigLoader
from matplotlib import pyplot as plt

from src.data_generator.noise_model import (calculate_arr_baseline, process_row)

# region logging

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    filename=Path(__file__).parent.parent / "logs" / "noise_model.log"
)
logger = logging.getLogger(__name__)

# endregion

logger.info("Critband testing began")

file_path = Path(r"C:\Jack\code\GCMS_Automation\databases\test_matrices\10_test_matrices.npz")

cfg_path = Path(__file__).parent.parent / 'config.yaml'
cfg = ConfigLoader(cfg_path)
peak_mode = cfg.get('peak_mode')

starttime = datetime.now()

ims = np.load(file_path)

zc_thresholds, e_thresholds = [], []
zc_success, zc_tf, zc_cf = 0, 0, 0
e_success, e_tf, e_cf = 0, 0, 0
zc_bimod_scores, e_bimod_scores = [], []

fail_tracker = {
    'critband':0,
    'unimodal':1,
    'hypermodal':2,
    'trough_failure':3,
    'critband_failure':4
}
labels = list(fail_tracker.keys())
zc_fails = np.zeros(len(fail_tracker))
e_fails = np.zeros(len(fail_tracker))

for i,(name,matrix) in enumerate(ims.items()):
    # just deal with the first three matrices for a moment
    if i > 3:
        continue

    # remove TIC
    matrix = matrix[:-1]

    # get zc-hueristics
    for i,row in enumerate(matrix):

        data = process_row(row, f'{name}_row{i}', segment_size=13)

        zc_thresholds.append(data['zero_crossings']['threshold'])
        e_thresholds.append(data['energies']['threshold'])

        zc_calc = data['zero_crossings']['type']
        zc_fails[fail_tracker[zc_calc]] += 1

        e_calc = data['energies']['type']
        e_fails[fail_tracker[e_calc]] += 1

        zc_bimod_scores.append(data['zero_crossings']['bimod_test'])
        e_bimod_scores.append(data['energies']['bimod_test'])

fig,ax = plt.subplots(2,3, figsize=(10,12), squeeze=False)

ax[0,0].hist(zc_thresholds)
ax[0,0].set_title(f'Zero-Crossing Thresholds')
ax[0,1].hist(zc_bimod_scores)
ax[0,1].set_title(f'Zero-Crossing Bimodality Scores')
ax[0,2].bar(labels, zc_fails)
ax[0,2].set_title('Zero-Crossing Failure Modes')

ax[1,0].hist(e_thresholds)
ax[1,0].set_title(f'Energy Thresholds')
ax[1,1].hist(e_bimod_scores)
ax[1,1].set_title(f'Energy Bimodality Scores')
ax[1,2].bar(labels, e_fails)
ax[1,2].set_title('Energy Failure modes')

plt.tight_layout()
plt.show()

"""
# testing airPLS (decided not to use b/c of disjoint nature of baselines in GC-MS)
none_count = 0
good_count = 0
diffs, l_vals, v_vals, relmads, curvatures = [], [], [], [], []
with warnings.simplefilter(record=True) as caught:
    warnings.simplefilter("always")
    for name, matrix in ims.items():
        #remove TIC
        matrix = matrix[:-1]

        # airPLS fit rows
        total_rows = matrix.shape[0]
        processed_rows = 0
        for row in matrix:
            processed_rows += 1
            bl, l_val, v_val, relmad, curvature = calculate_arr_baseline(row)
            if l_val is not None:
                l_vals.append(l_val)
                v_vals.append(v_val)
                relmads.append(relmad)
                curvatures.append(curvature)
                good_count += 1
            else:
                none_count += 1
param_warnings = [w for w in caught if 'ParameterWarning' in str(w.category)]
div_warnings = [w for w in caught if 'invalid value encountered in divide' in str(w.message)]

print(f"ParameterWarning count: {len(param_warnings)}")
print(f"DivideWarning count: {len(div_warnings)}")
print(f"Rows fit with meidan: {none_count}\nRows fit with airPLS: {good_count}\n PCT airPLS: {100 * good_count / (good_count + none_count):.2f}")

diffs = [l-v for l,v in zip(l_vals, v_vals)]
plt_relmad = [r for r in relmads if abs(r) < 10]
nan_masks = []
for curv in curvatures:
    nan_masks.append(np.isnan(curv))
nan_masks = np.array(nan_masks)
nan_counts = nan_masks.sum(axis=1)

fig, ax = plt.subplots(2,1)
sc1 = ax[0].scatter(relmads, l_vals, c=nan_counts, cmap='viridis', alpha=0.5)
ax[0].set_xscale('linear')
ax[0].set_yscale('log')
ax[0].set_xlabel('relmad')
ax[0].set_ylabel('L-curve lambda')
fig.colorbar(sc1, ax=ax, label='NaN fraction')

sc2 = ax[1].scatter(relmads, v_vals, c=nan_counts, cmap='viridis', alpha=0.5)
ax[1].set_xscale('linear')
ax[1].set_yscale('log')
ax[1].set_xlabel('relmad')
ax[1].set_ylabel('V-curve lambda')
fig.colorbar(sc2, ax=ax, label='NaN fraction')
plt.show()"""

"""fig,ax = plt.subplots(3,2, figsize=(15,15), squeeze=False)
nan_counts = nan_masks.sum(axis=0)

ax[0,0].hist(diffs, bins=100)
ax[0,0].set_title('Score Differences (L - V)')
ax[0,1].hist(l_vals, bins=100)
ax[0,1].set_title(f'L-Scores Max:{np.max(l_vals)} Min:{np.min(l_vals)}')
ax[1,0].hist(v_vals, bins=100)
ax[1,0].set_title(f'V-Scores Max:{np.max(v_vals)} Min:{np.min(v_vals)}')
ax[1,1].hist(plt_relmad, bins=100)
ax[1,1].set_title('Relative MAD (MAD/Median) for relmad < 0.1')
ax[2,0].bar(range(len(nan_counts)), nan_counts)
ax[2,0].set_title('Nan counts per index position in curvature')

plt.suptitle(f"Best fit lambdas for {good_count} rows")
plt.tight_layout()
plt.show()"""