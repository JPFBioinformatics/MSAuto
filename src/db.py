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

from pathlib import Path
from datetime import datetime
import sqlite3

# logging
import logging
logger = logging.getLogger(__name__)

# region                 ---------- BASIC ----------

def connect(db_path: Path):
    """
    Connects to db
    
    Params
    ------
    db_path                 path to .db file to conect to

    Returns
    -------
    conn                    connection to db
    """

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: Path, schema_path: Path):
    """
    Creates empty database

    Params
    ------
    db_path                 path to put .db file at
    schema_path             path to .sql schema file
    """

    conn = connect(db_path)
    with open(schema_path) as f:
        conn.executescript(f.read())
    conn.close()

def init_run(run_name: str, project_name: str, app_dir: Path):
    """
    Initializes a run directory and prepares database
    """
    run_dir = app_dir / "databases" / project_name / run_name
    run_dir.mkdir(parents=True,exist_ok=True)
    db_path = run_dir / f"{run_name}.db"
    schema_path = app_dir / "GCMSdata.sql"
    init_db(db_path,schema_path)
    conn = connect(db_path)
    return conn, run_dir

# endregion

# region                 ---------- INSERT ----------

def insert_sample(conn: sqlite3.Connection,
                  sample_name: str,
                  run_name: str,
                  mouseID: str,
                  group_name: str,
                  sex: str,
                  norm_factor: float,
                  norm_factor_type: str,
                  injection_order: int):
    """
    Inserts into samples table
    """
    cur = conn.execute(
        """ INSERT INTO samples (sample_name, run_name, mouseID, group_name, sex, norm_factor, norm_factor_type, injection_order)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (sample_name, run_name, mouseID, group_name, sex, norm_factor, norm_factor_type, injection_order)
    )
    conn.commit()
    return cur.lastrowid

def insert_molecule(conn: sqlite3.Connection,
                    molecule_name: str,
                    run_name: str,
                    ion: int,
                    rt: float,
                    std: str,
                    casNO: str):
    """
    Inserts into the molecules table
    """
    cur = conn.execute(
        """ INSERT INTO molecules (molecule_name, run_name, ion, rt, std, casNO)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (molecule_name, run_name, ion, rt, std, casNO)
    )
    conn.commit()
    return cur.lastrowid

def insert_feature(conn: sqlite3.Connection,
                   sampleID: int,
                   feat_rt: float,
                   collection_ion: int,
                   identification: str):
    """
    Inserts into the features table
    """
    cur = conn.execute(
        """ INSERT INTO features (sampleID, feat_rt, collection_ion, identification)
            VALUES (?, ?, ?, ?)""",
            (sampleID, feat_rt, collection_ion, identification)
    )
    conn.commit()
    return cur.lastrowid

def insert_im(conn: sqlite3.Connection,
              sample_name: str,
              run_name: str,
              sample_type: str,
              noise_factor: float,
              abundance_threshold: float,
              n_ions: int,
              n_timepoints: int):
    """
    Inserts into the intensity_matrices table
    """
    created_at = datetime.now().isoformat()
    cur = conn.execute(
        """ INSERT INTO intensity_matrices (sample_name, run_name, created_at, sample_type, noise_factor, abundance_threshold, n_ions, n_timepoints)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (sample_name, run_name, created_at, sample_type, noise_factor, abundance_threshold, n_ions, n_timepoints)
    )
    conn.commit()
    return cur.lastrowid

def insert_peak(conn: sqlite3.Connection,
                imID: int,
                featID: int,
                center: int,
                left_bound: int,
                right_bound: int,
                rt: float,
                height: float,
                area: float,
                sn_ratio: float,
                ion: float,
                fwhh: float,
                tailing_factor: float):
    """
    Inserts into peaks table
    """
    conn.execute(
        """ INSERT INTO peaks (imID, featID, center, left_bound, right_Bound, rt, height, area, sn_ratio, ion, fwhh, tailing_factor)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (imID, featID, center, left_bound, right_bound, rt, height, area, sn_ratio, ion, fwhh, tailing_factor)
    )
    conn.commit()

def insert_run(conn: sqlite3.Connection,
               run_name: str,
               user: str,
               method: str):
    """
    Inserts a run into runs table
    """
    if _run_exists(conn,run_name):
        return ValueError(f"Run {run_name} already in database, choose unique run name")
    created_at = datetime.now().isoformat()
    cur = conn.execute(
        """ INSERT INTO runs (run_name, created_at, user, method)
            VALUES (?, ?, ?, ?)""",
            (run_name, created_at, user, method)
    )
    return cur.lastrowid

# endregion

# region                 ---------- FETCH ----------

def get_proj_samples(conn: sqlite3.Connection):
    """
    Gets rows for all samples associated with this project

    Returns
    -------
    List of row objects, access with rows['sample_name']
    """
    cur = conn.execute(
        "SELECT * FROM samples"
    )
    return cur.fetchall()

def get_sample(conn: sqlite3.Connection, sampleID: int):
    """
    Gets row for a given sampleID

    Returns
    -------
    Row object for 'sample_name'
    """
    cur = conn.execute(
        "SELECT * FROM samples WHERE sampleID = ?", (sampleID,)
    )
    return cur.fetchone()

def get_run_samples(conn: sqlite3.Connection, run_name: str):
    """
    Gets all samples associated with a given run
    """
    cur = conn.execute(
        "SELECT * FROM samples WHERE run_name = ?", (run_name,)
        )
    return cur.fetchall

def get_smaple_im(conn: sqlite3.Connection, sampleID: int):
    """
    Gets IM row for a given sampleID
    """
    cur = conn.execute(
        "SELECT * FROM intensity_matrices WHERE sampleID = ?", (sampleID,)
    )
    return cur.fetchone()

def get_proj_molecules(conn: sqlite3.Connection):
    """
    Gets all molecules for this project
    """
    cur = conn.execute(
        "SELECT * FROM molecules"
        )
    return cur.fetchall

def get_run_molecules(conn: sqlite3.Connection, run_name: str):
    """
    Gets all molecules for this run
    """
    cur = conn.execute(
        "SELECT * FROM molecules WHERE run_name = ?", (run_name)
    )
    return cur.fetchall()

def get_mol(conn: sqlite3.Connection, molID: str):
    """
    Gets a molecule's row based on its ID
    """
    cur = conn.execute(
        "SELECT * FROM molecules WHERE molID = ?", (molID,)
    )
    return cur.fetchone()

def get_im_peaks(conn: sqlite3.Connection, imID: int):
    """
    Gets all peaks for a given intensity matrix
    """
    cur = conn.execute(
        "SELECT * FROM peaks where imID = ?", (imID,)
    )
    return cur.fetchall()

def get_im_feats(conn: sqlite3.Connection, imID: int):
    """
    Gets all features for a given intensity matrix
    """
    cur = conn.execute(
        "SELECT * FROM features WHERE imID = ?", (imID,)
    )
    return cur.fetchall()

def load_run_peak_data(conn: sqlite3.Connection, run_name: str):
    """
    Generates a peak_data dict sample: peak_list structure for recreating a datamatrix
    from stored peak data
    """
    cur = conn.execute(
        "SELECT * FROM peaks where run_name = ?", (run_name)
    )
    peak_rows = cur.fetchall()
    peak_data = {}
    for row in peak_rows:
        sample = row['sample_name']
        if sample not in peak_data:
            peak_data[sample] = []
        peak_data[sample].append(dict(row))
    
    return peak_data

# endregion

# region                 ---------- UTILS ----------

def _run_exists(conn: sqlite3.Connection, run_name: str):
    """
    Returns bool true if a run_name already exists in the databse, false if it does not
    """
    row = conn.execute("SELECT 1 FROM runs WHWERE run_name = ?", (run_name))
    return row is not None

# endregion