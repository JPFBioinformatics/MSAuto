"""

Data container for loading a given run from database to feed to GUI

"""
import sqlite3
import sys
from pathlib import Path

# location of pipeline root dir
root_dir = Path(__file__).parent.parent.resolve()
# tell python to look here for modules
sys.path.insert(0, str(root_dir))

from src.db import get_run_samples, get_run_molecules, load_run_peak_data
from src.intensity_matrix import IntensityMatrix as IM
from src.config_loader import ConfigLoader
from src.data_matrix import DataMatrix as DM

class RunData:
    def __init__(self, conn: sqlite3.Connection, run_name: str, proj_name: str, cfg: ConfigLoader):

        self.cfg = cfg
        self.proj_name = proj_name
        self.run_name = run_name
        
        self.samples = {r['sample_name']: dict(r) for r in get_run_samples(conn, run_name)}             # dict sample_name: row_dict
        self.molecules = [dict(r) for r in get_run_molecules(conn, run_name)]                           # list of dicts
        
        self.intensity_matrices = {}
        for entry in self.samples:
            sample_name = entry['sample_name']
            self.intensity_matrices[sample_name] = IM.load_h5_object(sample_name, proj_name, run_name)

        peaks = load_run_peak_data(conn, run_name)

        self.data_matrix = DM(cfg, peaks)


