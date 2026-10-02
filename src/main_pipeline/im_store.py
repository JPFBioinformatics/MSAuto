"""
Disk-backed storage for IntensityMatrix objects, keeps at most 'max_in_memory' loaded, rest are
saved as full IM state in <run_dir>/samples so unchanged samples never get rebuilt
"""
import json
from pathlib import Path
from collections import OrderedDict
from contextlib import contextmanager

from src.main_pipeline.intensity_matrix import IntensityMatrix
from src.main_pipeline.utils import cfg_hash

import logging
logger = logging.getLogger(__name__)

def write_sample(im, samples_dir, cfg, name=None, molecules_hash=None):
    """
    saves an IM's full state plus a small sidecar used for validity checks
    module-level so worker processes can call it directly
    """
    samples_dir = Path(samples_dir)
    name = name or im.sample_name
    im.save_state(samples_dir / f"{name}.pkl")
    meta = {'sample_name': name,
            'state_version': IntensityMatrix.STATE_VERSION,
            'cfg_hash': cfg_hash(cfg),
            'molecules_hash': molecules_hash}
    with open(samples_dir / f"{name}.meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    return name


class IMStore:
    def __init__(self, samples_dir, cfg, max_in_memory=1):
        self.samples_dir = Path(samples_dir)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.cfg_tag = cfg_hash(cfg)
        self.max_in_memory = max_in_memory
        self._loaded = OrderedDict()

    # region paths / validity
    def path_for(self, name):
        return self.samples_dir / f"{name}.pkl"

    def is_valid(self, name):
        """saved state exists and was built with this config + state version"""
        meta_path = self.samples_dir / f"{name}.meta.json"
        if not (meta_path.exists() and self.path_for(name).exists()):
            return False
        with open(meta_path) as f:
            meta = json.load(f)
        return (meta.get('state_version') == IntensityMatrix.STATE_VERSION
                and meta.get('cfg_hash') == self.cfg_tag)

    # endregion

    # region write / read
    def put(self, name, im):
        write_sample(im, self.samples_dir, self.cfg, name=name)
        self._loaded.pop(name, None)

    def __getitem__(self, name):
        if name in self._loaded:
            self._loaded.move_to_end(name)
            return self._loaded[name]
        if not self.path_for(name).exists():
            raise KeyError(name)
        im = IntensityMatrix.load_state(self.path_for(name), cfg=self.cfg)
        self._loaded[name] = im
        while len(self._loaded) > self.max_in_memory:
            self._loaded.popitem(last=False)
        return im

    def labels_hash(self, name):
        """molecule-table fingerprint this sample's peaks were labelled with (None if never)"""
        meta_path = self.samples_dir / f"{name}.meta.json"
        if not meta_path.exists():
            return None
        with open(meta_path) as f:
            return json.load(f).get('molecules_hash')

    @contextmanager
    def edit(self, name, molecules_hash=None):
        """load, modify, write back (optionally recording the molecule labels it now carries)"""
        im = self[name]
        yield im
        write_sample(im, self.samples_dir, self.cfg, name=name, molecules_hash=molecules_hash)

    # endregion

    # region dict-like
    def __contains__(self, name):
        return self.path_for(name).exists()

    def keys(self):
        return sorted(p.stem for p in self.samples_dir.glob('*.pkl'))

    def __len__(self):
        return len(self.keys())

    def get(self, name, default=None):
        return self[name] if name in self else default

    def items(self):
        for name in self.keys():
            yield name, self[name]

    def values(self):
        for _, im in self.items():
            yield im

    # endregion
