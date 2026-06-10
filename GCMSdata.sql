-- run_name is included in several tables where it breaks normal form, but it remains there
-- in order to simplify querying for GUI



CREATE TABLE IF NOT EXISTS runs (
    run_name                TEXT PRIMARY KEY,
    created_at              TEXT,
    user                    TEXT,
    method                  TEXT,
    file_type               TEXT,
    norm_type               TEXT
);

CREATE TABLE IF NOT EXISTS samples (
    sample_name             TEXT NOT NULL,
    run_name                TEXT NOT NULL,
    sampleID                TEXT PRIMARY KEY,
    group_name              TEXT,
    sex                     TEXT,
    norm_factor             REAL,
    injection_order         INTEGER,
    FOREIGN KEY (run_name) REFERENCES runs (run_name)
);

CREATE TABLE IF NOT EXISTS molecules (
    molID                   INTEGER PRIMARY KEY,
    molecule_name           TEXT,
    run_name                TEXT NOT NULL,
    ion                     INTEGER,
    rt                      REAL,
    std                     TEXT,
    casNO                   TEXT,
    FOREIGN KEY (run_name) REFERENCES runs (run_name)
);

CREATE TABLE IF NOT EXISTS features (
    featID                  INTEGER PRIMARY KEY,
    sample_name             TEXT NOT NULL,
    run_name                TEXT NOT NULL,
    feat_rt                 REAL,
    collection_ion          INTEGER,
    feat_name               TEXT,
    confidence              REAL,
    FOREIGN KEY (sample_name, run_name) REFERENCES samples (sample_name, run_name),
    FOREIGN KEY (run_name) REFERENCES runs (run_name)
);

CREATE TABLE IF NOT EXISTS intensity_matrices (
    imID                    INTEGER PRIMARY KEY,
    run_name                TEXT NOT NULL,
    sample_name             TEXT NOT NULL,
    created_at              TEXT,
    sample_type             TEXT,
    noise_factor            REAL,
    abundance_threshold     REAL,
    n_ions                  INTEGER,
    n_timepoints            INTEGER,
    FOREIGN KEY (sample_name, run_name) REFERENCES samples (sample_name, run_name),
    FOREIGN KEY (run_name) REFERENCES runs (run_name)
);

CREATE TABLE IF NOT EXISTS peaks (
    peakID                  INTEGER PRIMARY KEY,
    run_name                TEXT NOT NULL,
    imID                    INTEGER NOT NULL,
    featID                  INTEGER NOT NULL,
    center                  INTEGER,
    left_bound              INTEGER,
    right_bound             INTEGER,
    rt                      REAL,
    height                  REAL,
    area                    REAL,
    sn_ratio                REAL,
    ion                     INTEGER,
    fwhh                    REAL,
    tailing_factor          REAL,
    bl_slope                REAL,
    bl_yint                 REAL,
    FOREIGN KEY (imID) REFERENCES intensity_matrices (imID),
    FOREIGN KEY (featID) REFERENCES features (featID),
    FOREIGN KEY (run_name) REFERENCES runs (run_name)
);
