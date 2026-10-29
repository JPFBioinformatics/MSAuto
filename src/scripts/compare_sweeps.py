import json, pandas as pd
from pathlib import Path
from src.main_pipeline.utils import get_app_dir
from src.data_generator.sample_cache import _cfg_hash

root = get_app_dir() / 'databases' / 'noise_models' / 'cache' / 'samples'
new_hash = _cfg_hash(get_app_dir() / 'default_config.yaml')
for sample in ('Int 1', 'Int 2'):
    old = pd.read_pickle(root / f'{sample}_602e2549' / 'peaks.pkl')
    new = pd.read_pickle(root / f'{sample}_{new_hash}' / 'peaks.pkl')
    cmp = pd.DataFrame({'old': old.groupby('ion').size(), 'new': new.groupby('ion').size()}).fillna(0)
    cmp['change'] = cmp['new'] - cmp['old']
    print(sample, 'total', len(old), '->', len(new))
    print(cmp.loc[[298.0, 505.0, 212.0]].to_string())          # sparse, sparse, dense
    print(cmp.sort_values('change').head(10).to_string())      # biggest drops
