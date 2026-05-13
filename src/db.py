"""
Some useful functions for sql database managment
"""

from pathlib import Path
from datetime import datetime
import sqlite3

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
        """ INSERT INTO samples (sample_name, mouseID, group_name, sex, norm_factor, norm_factor_type, injection_order)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (sample_name, mouseID, group_name, sex, norm_factor, norm_factor_type, injection_order)
    )
    conn.commit()
    return cur.lastrowid

def insert_molecule(conn: sqlite3.Connection,
                    molecule_name: str,
                    ion: int,
                    rt: float,
                    std: str):
    """
    Inserts into the molecules table
    """
    cur = conn.execute(
        """ INSERT INTO molecules (molecule_name, ion, rt, std)
            VALUES (?, ?, ?, ?)""",
            (molecule_name, ion, rt, std)
    )
    conn.commit()
    return cur.lastrowid

def insert_feature(conn: sqlite3.Connection,
                   sample_name: str,
                   feat_rt: float,
                   collection_ion: int,
                   identification: str):
    """
    Inserts into the features table
    """
    cur = conn.execute(
        """ INSERT INTO features (sample_name, feat_rt, collection_ion, identification)
            VALUES (?, ?, ?, ?)""",
            (sample_name, feat_rt, collection_ion, identification)
    )
    conn.commit()
    return cur.lastrowid

def insert_im(conn: sqlite3.Connection,
              sample_name: str,
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
        """ INSERT INTO intensity_matrices (sample_name, created_at, sample_type, noise_factor, abundance_threshold, n_ions, n_timepoints)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (sample_name, created_at, sample_type, noise_factor, abundance_threshold, n_ions, n_timepoints)
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
                ion: float):
    """
    Inserts into peaks table
    """
    conn.execute(
        """ INSERT INTO peaks (imID, featID, center, left_bound, right_Bound, rt, height, area, sn_ratio, ion)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (imID, featID, center, left_bound, right_bound, rt, height, area, sn_ratio, ion)
    )
    conn.commit()

# endregion

# region                 ---------- FETCH ----------

def get_samples(conn: sqlite3.Connection):
    """
    Gets rows for all samples associated with this run

    Returns
    -------
    List of row objects, access with rows['sample_name']
    """
    cur = conn.execute(
        "SELECT * FROM samples"
    )
    return cur.fetchall()

def get_sample(conn: sqlite3.Connection, sample_name: str):
    """
    Gets row for a given sample_name

    Returns
    -------
    Row object for 'sample_name'
    """
    cur = conn.execute(
        "SELECT * FROM samples WHERE sample_name = ?", (sample_name,)
    )
    return cur.fetchone()

def get_smaple_im(conn: sqlite3.Connection, sample_name: str):
    """
    Gets IM row for a given sample_name
    """
    cur = conn.execute(
        "SELECT * FROM intensity_matrices WHERE sample_name = ?", (sample_name,)
    )
    return cur.fetchone()

def get_molecules(conn: sqlite3.Connection):
    """
    Gets all molecules for this run
    """
    cur = conn.execute(
        "SELECT * FROM molecules"
    )
    return cur.fetchall()

def get_mol_by_name(conn: sqlite3.Connection, mol_name: str):
    """
    Gets a molecule's row based on its name
    """
    cur = conn.execute(
        "SELECT * FROM molecules WHERE molecule_name = ?", (mol_name,)
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

def load_peak_data(conn: sqlite3.Connection):
    """
    Generates a peak_data dict sample: peak_list structure for recreating a datamatrix
    from stored peak data
    """
    cur = conn.execute(
        "SELECT * FROM peaks"
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
