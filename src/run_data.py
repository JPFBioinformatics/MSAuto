"""

Data container for loading a given run from database to feed to GUI, used for visualization

"""

# region Imports

import sqlite3
from src.db import get_run_samples, get_run_molecules, get_run_peaks
from src.intensity_matrix import IntensityMatrix as IM
from src.config_loader import ConfigLoader
from src.data_matrix import DataMatrix as DM
from src.db import connect
from src.utils import get_proj_db

# logging
import logging
logger = logging.getLogger(__name__)

# endregion

class RunData:
    def __init__(self, run_name: str, proj_name: str):

        self.proj_name = proj_name
        self.run_name = run_name
        
        db_path = get_proj_db(proj_name)
        try:
            conn = connect(db_path)
            self.samples = {r['sample_name']: dict(r) for r in get_run_samples(conn, run_name)}             # dict sample_name: row_dict
            self.molecules = {r["molecule_name"]: dict(r) for r in get_run_molecules(conn, run_name)}       # dict of mol_name: row_dict
            peaks = get_run_peaks(conn, run_name)
        finally:
            conn.close()

        self.intensity_matrices = {}
        for sample_name in self.samples:
            self.intensity_matrices[sample_name] = IM.load_h5_object(sample_name, proj_name, run_name)

        self.data_matrix = DM(proj_name, run_name, peaks)


