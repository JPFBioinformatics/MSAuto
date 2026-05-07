
CREATE TABLE IF NOT EXISTS samples (
    sample_name             TEXT PRIMARY KEY,
    mouseID                 TEXT,
    group_name              TEXT,
    sex                     TEXT,
    norm_factor             REAL,
    norm_factor_type        TEXT,
    injection_order         INTEGER,
);

CREATE TABLE IF NOT EXISTS molecules (
    moleculeID              INTEGER PRIMARY KEY,
    molecule_name           TEXT,
    ion                     INTEGER,
    rt                      REAL,
    std                     TEXT,
    casNO                   TEXT,
);

CREATE TABLE IF NOT EXISTS features (
    featID                  INTEGER PRIMARY KEY,
    imID                    INTEGER NOT NULL,
    feat_rt                 REAL,
    collection_ion          INTEGER,
    identification          TEXT,
    FOREIGN KEY (imID) REFERENCES intensity_matrices(imID)
);

CREATE TABLE IF NOT EXISTS intensity_matrices (
    imID                    INTEGER PRIMARY KEY,
    sample_name             TEXT NOT NULL,
    created_at              TEXT,
    sample_type             TEXT,
    noise_factor            REAL,
    abundance_threshold     REAL,
    n_ions                  INTEGER,
    n_timepoints            INTEGER,
    FOREIGN KEY (sample_name) REFERENCES samples(sample_name)
);

CREATE TABLE IF NOT EXISTS peaks (
    peakID                  INTEGER PRIMARY KEY,
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
    FOREIGN KEY (imID) REFERENCES intensity_matrices(imID),
    FOREIGN KEY (featID) REFERENCES features(featID)
);
