"""

Data container for loading a given run from database to feed to GUI, used for visualization

"""

# region Imports

import numpy as np
from pathlib import Path
from src.main_pipeline.im_store import IMStore
from src.main_pipeline.utils import molecules_hash
from src.gui.data_matrix import DataMatrix as DM
from src.main_pipeline.utils import get_run_dir
from src.main_pipeline.db import (connect, ensure_db, save_run_to_db, get_run,
                                  get_run_samples, get_run_molecules)

# logging
import logging
logger = logging.getLogger(__name__)

# endregion

class RunData:
    def __init__(self, run_name: str, proj_name: str, cfg):
        """
        Opens a saved run: samples/molecules from the project db, IMs from the run's saved sample files
        """
        self.proj_name = proj_name
        self.run_name = run_name
        self.run_type = None
        self.cfg = cfg
        self.failed_samples = []

        conn = None
        try:
            conn = connect(ensure_db())
            self.samples = {r['sample_name']: {k: v for k, v in dict(r).items()
                                               if k not in ('run_id', 'sample_id')}
                            for r in get_run_samples(conn, proj_name, run_name)}
            self.molecules = {r['molecule_name']: {k: v for k, v in dict(r).items()
                                                   if k not in ('run_id', 'molecule_id')}
                              for r in get_run_molecules(conn, proj_name, run_name)}

            # restore run type so re-saving a loaded run keeps it
            run_row = get_run(conn, proj_name, run_name)
            self.run_type = run_row['run_type'] if run_row else None
        finally:
            if conn:
                conn.close()

        # saved IMs load lazily from <run_dir>/samples (no peak detection on open)
        self.intensity_matrices = IMStore(get_run_dir(proj_name, run_name) / 'samples', cfg)
        for sample_name in self.samples:
            if sample_name not in self.intensity_matrices:
                logger.warning(f"No saved IntensityMatrix for {sample_name} in run {run_name}")
                self.failed_samples.append([sample_name, None])

        self._build_from_store()

    # region Data loading/saving

    @classmethod
    def from_processing(cls, proj_name, run_name, samples, molecules, intensity_matrices, run_type, cfg):
        """
        Creates rundata object before anything has been saved to sql database
        intensity_matrices is the IMStore the processing step wrote to
        """
        obj = cls.__new__(cls)
        obj.proj_name = proj_name
        obj.run_name = run_name
        obj.samples = samples
        obj.molecules = molecules
        obj.intensity_matrices = intensity_matrices
        obj.run_type = run_type
        obj.cfg = cfg

        # record samples that failed to build
        input_dir = Path(cfg.get('input_dir'))
        obj.failed_samples = []
        for sample_name in obj.samples:
            if sample_name not in obj.intensity_matrices:
                logger.warning(f"No IntensityMatrix found for {sample_name}, sample was not processed successfully")
                obj.failed_samples.append([sample_name, input_dir / f"{sample_name}.mzML"])

        obj._build_from_store()
        return obj

    def _build_from_store(self):
        """
        Shared by __init__ and from_processing:
            1) per sample: collect molecule peaks + generate spectra (written back to disk)
            2) build the DataMatrix from all samples' peaks
            3) per sample: label matched peaks with their molecule (written back to disk)
        Each IM is loaded once per step, so only one is in memory at a time
        """
        mols, mzs, rts = [], [], []
        for entry in self.molecules.values():
            mols.append(entry['molecule_name'])
            mzs.append(np.int64(entry['ion']))
            rts.append(entry['rt'])

        # 1) collect peaks + spectra, one sample at a time (read only, spectra live on the peak copies)
        peaks, max_mz = {}, 0
        for name in self.samples:
            if name not in self.intensity_matrices:
                continue
            matrix = self.intensity_matrices[name]
            peak_list = matrix.collect_data(mols, mzs, rts)
            for peak in peak_list:
                matrix.generate_spectra(peak, label=peak['molecule'], n_closest=10, save_spectra=True)
            peaks[name] = peak_list
            max_mz = max(max_mz, max(mz for mz in matrix.unique_mzs if mz != 9999))
        self.max_mz = max_mz + 1

        # 2) data matrix across all samples
        self.data_matrix = DM(self.proj_name, self.run_name, peaks, self.samples,
                              self.molecules, self.max_mz, self.cfg)

        # 3) group molecule assignments by sample so each IM is loaded once
        sample_names = {v: k for k, v in self.data_matrix.sample_map.items()}
        mol_names = {v: k for k, v in self.data_matrix.mol_map.items()}
        peak_idx = self.data_matrix.data['peak_idx']
        by_sample = {}
        for i in range(peak_idx.shape[0]):          # rows/samples
            for j in range(peak_idx.shape[1]):      # cols/molecules
                if peak_idx[i][j] != -1:
                    by_sample.setdefault(sample_names[i], []).append((peak_idx[i][j], mol_names[j]))

        mol_tag = molecules_hash(self.molecules)
        relabelled = 0
        for name in self.samples:
            if name not in self.intensity_matrices:
                continue
            if self.intensity_matrices.labels_hash(name) == mol_tag:
                continue
            with self.intensity_matrices.edit(name, molecules_hash=mol_tag) as im:
                for peak_list in im.peak_dict.values():
                    for peak in peak_list:
                        peak['molecule'] = None
                for idx, molecule in by_sample.get(name, []):
                    ion = np.int64(self.molecules[molecule]['ion'])
                    im.peak_dict[ion][idx]['molecule'] = molecule
            relabelled += 1
        logger.info(f"Relabelled {relabelled}/{len(self.samples)} samples for molecule table {mol_tag}")


    def reassign_molecules(self, molecules: dict):
        """
        swap in a new molecule table and redo matching using the saved samples (no peak detection)
        returns True if the table actually changed
        """
        if molecules_hash(molecules) == molecules_hash(self.molecules):
            return False
        self.molecules = molecules
        self._build_from_store()
        return True

    def _iter_ims(self):
        """(sample_name, IM) pairs, loaded one at a time from the store"""
        for name in self.samples:
            if name in self.intensity_matrices:
                yield name, self.intensity_matrices[name]

    def save_run(self, cfg):
        """saves config to the run folder and the run's metadata + results to the global db"""
        run_dir = get_run_dir(self.proj_name, self.run_name)
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg.save()

        conn = connect(ensure_db())
        try:
            counts = save_run_to_db(conn, self.proj_name, self.run_name, self.run_type, cfg,
                                    self.samples, self.molecules, self._iter_ims(), overwrite=True)
        finally:
            conn.close()
        logger.info(f"Saved run {self.run_name}: {counts}")
        return counts

# endregion