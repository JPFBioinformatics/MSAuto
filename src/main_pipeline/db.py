"""

Some useful functions for sql database managment

Databasese are desinged so each project gets its own set of runs (batches) and project
dbs are physically sepearated into different folders in the larger database directory.

Generally speaking you would first pull all samples from a given run, then use the sampleIDs
there to get all the intensity_matrices/features, and you can pull all peaks associated with
these intensity matrices using imID to get all the data you need.  Also molecules table is
associated with runs via run_name, so you can pull that too and you can get all data associated
with a given run/batch.

"""

# region Imports
import numpy as np
from pathlib import Path
from datetime import datetime
import sqlite3, json
from src.main_pipeline.utils import get_global_db, get_schema_path, cfg_hash


# logging
import logging
logger = logging.getLogger(__name__)

# endregion

SCHEMA_VERSION = 1          # bump by hand whenever the db structure changes

# upgrade steps: version -> SQL that takes a db from (version - 1) to version
# the base schema (GCMSdata.sql) creates everything up to the current version for new dbs
MIGRATIONS = {
    # 2: "ALTER TABLE peaks ADD COLUMN new_metric REAL;",
}

# region fetch

def get_project_names(conn):
    return [r['project_name'] for r in
            conn.execute("SELECT project_name FROM projects ORDER BY project_name").fetchall()]

def get_run_names(conn, project_name):
    rows = conn.execute(
        """SELECT r.run_name FROM runs r JOIN projects p ON p.project_id = r.project_id
           WHERE p.project_name = ? ORDER BY r.created_at""", (project_name,)).fetchall()
    return [r['run_name'] for r in rows]

def get_run(conn, project_name, run_name):
    return conn.execute(
        """SELECT r.* FROM runs r JOIN projects p ON p.project_id = r.project_id
           WHERE p.project_name = ? AND r.run_name = ?""", (project_name, run_name)).fetchone()

def get_run_samples(conn, project_name, run_name):
    return conn.execute(
        """SELECT s.* FROM samples s JOIN runs r ON r.run_id = s.run_id
           JOIN projects p ON p.project_id = r.project_id
           WHERE p.project_name = ? AND r.run_name = ?""", (project_name, run_name)).fetchall()

def get_run_molecules(conn, project_name, run_name):
    return conn.execute(
        """SELECT m.* FROM molecules m JOIN runs r ON r.run_id = m.run_id
           JOIN projects p ON p.project_id = r.project_id
           WHERE p.project_name = ? AND r.run_name = ?""", (project_name, run_name)).fetchall()

# endregion

# region setup

def init_db(db_path: Path, schema_path: Path):
    """creates the database from the schema file (safe to rerun, tables use IF NOT EXISTS)"""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path, create=True)
    try:
        with open(schema_path) as f:
            conn.executescript(f.read())
    finally:
        conn.close()

def ensure_db():
    """returns the global db path, creating it or upgrading its schema to SCHEMA_VERSION"""
    db_path = get_global_db()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path, create=True)
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            # brand new (or failed first setup): create everything from the schema file
            with open(get_schema_path()) as f:
                conn.executescript(f.read())
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        elif version < SCHEMA_VERSION:
            # existing db from an older app version: apply each upgrade in order
            for v in range(version + 1, SCHEMA_VERSION + 1):
                if v in MIGRATIONS:
                    conn.executescript(MIGRATIONS[v])
                conn.execute(f"PRAGMA user_version = {v}")
        elif version > SCHEMA_VERSION:
            raise RuntimeError(f"Database version {version} is newer than this app ({SCHEMA_VERSION}), "
                               f"please update the app")
    finally:
        conn.close()
    return db_path



# endregion

# region projects/runs

def get_project_id(conn, project_name):
    row = conn.execute("SELECT project_id FROM projects WHERE project_name = ?",
                       (project_name,)).fetchone()
    return row['project_id'] if row else None

def get_or_create_project(conn, project_name, description=None):
    pid = get_project_id(conn, project_name)
    if pid is not None:
        return pid
    return conn.execute(
        "INSERT INTO projects (project_name, created_at, description) VALUES (?, ?, ?)",
        (project_name, datetime.now().isoformat(), description)).lastrowid

def get_run_id(conn, project_name, run_name):
    row = conn.execute(
        """SELECT r.run_id FROM runs r JOIN projects p ON p.project_id = r.project_id
           WHERE p.project_name = ? AND r.run_name = ?""",
        (project_name, run_name)).fetchone()
    return row['run_id'] if row else None

def run_exists(conn, project_name, run_name):
    return get_run_id(conn, project_name, run_name) is not None

def insert_run(conn, project_id, run_name, run_type, cfg, user='default', method='default',
               norm_type='default'):
    """inserts a run with the config it was processed with, returns run_id"""
    return conn.execute(
        """INSERT INTO runs (project_id, run_name, run_type, created_at, user, method, norm_type,
                             input_dir, cfg_hash, config_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (project_id, run_name, run_type, datetime.now().isoformat(), user, method, norm_type,
         str(cfg.get('input_dir')), cfg_hash(cfg), json.dumps(cfg.config, default=str))).lastrowid

def delete_run(conn, project_name, run_name):
    """removes a run and (via ON DELETE CASCADE) its samples, molecules, peaks, features, stats"""
    run_id = get_run_id(conn, project_name, run_name)
    if run_id is not None:
        conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))

# endregion

# region run contents

def insert_peak_batch(conn, im, sample_id, molecule_ids, matched_only=True):
    """
    inserts an IM's peaks; molecule_ids maps molecule_name -> molecule_id for this run
    matched_only=True stores only peaks assigned to a molecule (full sets live in the .pkl)
    """
    cols = ('sample_id', 'molecule_id', 'ion', 'peak_idx') + PEAK_COLUMNS
    rows = []
    for ion, peak_list in im.peak_dict.items():
        for idx, peak in enumerate(peak_list):
            mol = peak.get('molecule')
            if matched_only and mol is None:
                continue
            rows.append((sample_id, molecule_ids.get(mol), _py(ion), idx)
                        + tuple(_py(peak.get(c)) for c in PEAK_COLUMNS))
    conn.executemany(
        f"INSERT INTO peaks ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", rows)
    return len(rows)


def insert_molecules(conn, run_id, molecules: dict):
    """inserts the run's molecule table, returns {molecule_name: molecule_id}"""
    ids = {}
    for row in molecules.values():
        ids[row['molecule_name']] = conn.execute(
            """INSERT INTO molecules (run_id, molecule_name, ion, rt, std, casNo)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (run_id, row['molecule_name'], _py(row.get('ion')), _py(row.get('rt')),
             row.get('std'), row.get('casNo'))).lastrowid
    return ids

def insert_sample(conn, run_id, row: dict, im=None):
    """inserts sample metadata (+ IM summary if available), returns sample_id"""
    matrix_type = noise_factor = n_ions = n_scans = scan_interval = None
    if im is not None:
        n_ions, n_scans = im.intensity_matrix.shape
        times = np.array([t for _, t in sorted(im.time_map.items())])
        scan_interval = float(np.median(np.diff(times))) if len(times) > 1 else None
        matrix_type, noise_factor = im.matrix_type, im.noise_factor
    return conn.execute(
        """INSERT INTO samples (run_id, sample_name, modelID, group_name, sex, norm_factor,
                                injection_order, matrix_type, noise_factor, n_ions, n_scans, scan_interval)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, row['sample_name'], row.get('modelID'), row.get('group_name'), row.get('sex'),
         _py(row.get('norm_factor')), _py(row.get('injection_order')),
         matrix_type, _py(noise_factor), _py(n_ions), _py(n_scans), scan_interval)).lastrowid

def insert_sample_stats(conn, sample_id, im):
    """per-sample summary of ALL detected peaks (for global comparisons without storing every peak)"""
    peaks = [p for plist in im.peak_dict.values() for p in plist]
    def med(key):
        vals = np.array([_py(p.get(key)) for p in peaks], dtype=float)
        vals = vals[np.isfinite(vals)]
        return float(np.median(vals)) if len(vals) else None
    n = len(peaks)
    frac_overlap = (sum(1 for p in peaks if p.get('overlap_left') or p.get('overlap_right')) / n) if n else None
    conn.execute(
        """INSERT INTO sample_stats (sample_id, n_peaks, median_fwhh, median_sn, median_tailing,
                                     median_height, frac_overlapped)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (sample_id, n, med('fwhh'), med('sn_ratio'), med('tailing_factor'), med('height'), frac_overlap))

# endregion

# region save full run

def save_run_to_db(conn, project_name, run_name, run_type, cfg, samples: dict, molecules: dict,
                   ims, overwrite=True, matched_only=True):
    """
    saves a whole run in one transaction (all-or-nothing)

    Params
    ------
    samples                 {sample_name: row dict} (sample table rows)
    molecules               {molecule_name: row dict}
    ims                     iterable of (sample_name, IntensityMatrix), loaded one at a time
    overwrite               replace an existing run with the same name instead of failing
    matched_only            only store molecule-matched peaks (full peak sets stay in the .pkl files)

    Returns
    -------
    dict of counts for logging / user feedback
    """
    counts = {'samples': 0, 'molecules': 0, 'peaks': 0}
    with conn:
        if run_exists(conn, project_name, run_name):
            if not overwrite:
                raise ValueError(f"Run {run_name} already saved in project {project_name}")
            delete_run(conn, project_name, run_name)

        project_id = get_or_create_project(conn, project_name)
        run_id = insert_run(conn, project_id, run_name, run_type, cfg)

        molecule_ids = insert_molecules(conn, run_id, molecules)
        counts['molecules'] = len(molecule_ids)

        saved = set()
        for sample_name, im in ims:
            sample_id = insert_sample(conn, run_id, samples[sample_name], im)
            insert_sample_stats(conn, sample_id, im)
            counts['peaks'] += insert_peak_batch(conn, im, sample_id, molecule_ids, matched_only)
            saved.add(sample_name)

        # samples with no IM (failed processing) still get their metadata row
        for sample_name, row in samples.items():
            if sample_name not in saved:
                insert_sample(conn, run_id, row, im=None)
        counts['samples'] = len(samples)

    return counts

# endregion

# region helpers

def connect(db_path: Path, create: bool = False):
    if not create and not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn

def _py(v):
    """numpy -> plain python for sqlite (np.int64 is not accepted)"""
    if v is None:
        return None
    if isinstance(v, (np.integer, np.bool_, bool)):
        return int(v)
    if isinstance(v, np.floating):
        return None if np.isnan(v) else float(v)
    if isinstance(v, float) and np.isnan(v):
        return None
    return v

PEAK_COLUMNS = ('center', 'left_bound', 'right_bound', 'rt', 'raw_height', 'height', 'area',
                'sn_ratio', 'fwhh', 'tailing_factor', 'bound_symmetry', 'valley_ratio', 'conv',
                'flat_top', 'symmetry_valid', 'overlap_left', 'overlap_right',
                'cwt_score', 'cwt_scale', 'ridge_span', 'cluster')

# endregion