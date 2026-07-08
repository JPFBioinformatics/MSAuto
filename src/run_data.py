"""

Data container for loading a given run from database to feed to GUI, used for visualization

"""

# region Imports

import numpy as np
from src.db import get_run_samples, get_run_molecules
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
        finally:
            conn.close()

        mols = []
        mzs = []
        rts = []
        for entry in self.molecules.values():
            mols.append(entry['molecule_name'])
            mzs.append(np.int64(entry['ion']))
            rts.append(entry['rt'])

        self.intensity_matrices = {}
        peaks = {}
        for sample_name in self.samples:
            matrix = IM.load_h5_object(sample_name, proj_name, run_name)
            peak_list = matrix.collect_data(mols, mzs, rts)
            peaks[sample_name] = peak_list
            self.intensity_matrices[sample_name] = matrix

        self.data_matrix = DM(proj_name, run_name, peaks)

        # reassign detected molecules
        sample_names = {v:k for k,v in self.data_matrix.sample_map.items()}
        mol_names = {v:k for k,v in self.data_matrix.mol_map.items()}
        for i in range(self.data_matrix.data['peak_idx'].shape[0]):         # rows/samples
            for j in range(self.data_matrix.data['peak_idx'].shape[1]):     # cols/molecules
                
                peak_idx = self.data_matrix.data['peak_idx'][i][j]
                if peak_idx == -1:
                    continue

                name = sample_names[i]
                molecule = mol_names[j]

                im = self.intensity_matrices[name]
                ion = np.int64(self.molecules[molecule]['ion'])

                peak_list = im.peak_dict[ion]
                peak = peak_list[peak_idx]

                peak['molecule'] = molecule
