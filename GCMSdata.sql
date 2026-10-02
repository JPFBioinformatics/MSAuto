-- Global GCMS database: projects -> runs -> samples -> peaks / features
-- Raw data and full peak sets live in <run_dir>/samples/*.pkl, this db holds metadata + results

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS projects (
    project_id          INTEGER PRIMARY KEY,
    project_name        TEXT NOT NULL UNIQUE,
    created_at          TEXT NOT NULL,
    description         TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    run_id              INTEGER PRIMARY KEY,
    project_id          INTEGER NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    run_name            TEXT NOT NULL,
    run_type            TEXT,
    created_at          TEXT NOT NULL,
    user                TEXT,
    method              TEXT,
    norm_type           TEXT,
    input_dir           TEXT,
    cfg_hash            TEXT,
    config_json         TEXT,
    UNIQUE (project_id, run_name)
);

CREATE TABLE IF NOT EXISTS samples (
    sample_id           INTEGER PRIMARY KEY,
    run_id              INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    sample_name         TEXT NOT NULL,
    modelID             TEXT,
    group_name          TEXT,
    sex                 TEXT,
    norm_factor         REAL,
    injection_order     INTEGER,
    matrix_type         TEXT,
    noise_factor        REAL,
    n_ions              INTEGER,
    n_scans             INTEGER,
    scan_interval       REAL,
    UNIQUE (run_id, sample_name)
);

CREATE TABLE IF NOT EXISTS molecules (
    molecule_id         INTEGER PRIMARY KEY,
    run_id              INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    molecule_name       TEXT NOT NULL,
    ion                 INTEGER,
    rt                  REAL,
    std                 TEXT,
    casNo               TEXT,
    UNIQUE (run_id, molecule_name)
);

CREATE TABLE IF NOT EXISTS features (
    feature_id          INTEGER PRIMARY KEY,
    sample_id           INTEGER NOT NULL REFERENCES samples(sample_id) ON DELETE CASCADE,
    feature_key         TEXT NOT NULL,
    feat_rt             REAL,
    collection_ion      INTEGER,
    feat_name           TEXT,
    confidence          REAL,
    UNIQUE (sample_id, feature_key)
);

CREATE TABLE IF NOT EXISTS peaks (
    peak_id             INTEGER PRIMARY KEY,
    sample_id           INTEGER NOT NULL REFERENCES samples(sample_id) ON DELETE CASCADE,
    molecule_id         INTEGER REFERENCES molecules(molecule_id) ON DELETE SET NULL,
    feature_id          INTEGER REFERENCES features(feature_id) ON DELETE SET NULL,
    ion                 INTEGER NOT NULL,
    peak_idx            INTEGER NOT NULL,
    -- location
    center              INTEGER,
    left_bound          INTEGER,
    right_bound         INTEGER,
    rt                  REAL,
    -- size
    raw_height          REAL,
    height              REAL,
    area                REAL,
    sn_ratio            REAL,
    -- shape
    fwhh                REAL,
    tailing_factor      REAL,                  -- USP, 5% height
    bound_symmetry      REAL,
    valley_ratio        REAL,
    conv                REAL,
    flat_top            INTEGER,               -- 0/1
    symmetry_valid      INTEGER,               -- 0/1
    overlap_left        INTEGER,               -- 0/1
    overlap_right       INTEGER,               -- 0/1
    -- detection
    cwt_score           REAL,
    cwt_scale           REAL,
    ridge_span          REAL,
    cluster             INTEGER,
    UNIQUE (sample_id, ion, peak_idx)
);

CREATE TABLE IF NOT EXISTS sample_stats (
    sample_id           INTEGER PRIMARY KEY REFERENCES samples(sample_id) ON DELETE CASCADE,
    n_peaks             INTEGER,
    median_fwhh         REAL,
    median_sn           REAL,
    median_tailing      REAL,
    median_height       REAL,
    frac_overlapped     REAL
);

CREATE INDEX IF NOT EXISTS idx_runs_project   ON runs(project_id);
CREATE INDEX IF NOT EXISTS idx_samples_run    ON samples(run_id);
CREATE INDEX IF NOT EXISTS idx_molecules_run  ON molecules(run_id);
CREATE INDEX IF NOT EXISTS idx_peaks_sample   ON peaks(sample_id);
CREATE INDEX IF NOT EXISTS idx_peaks_molecule ON peaks(molecule_id);
CREATE INDEX IF NOT EXISTS idx_peaks_ion_rt   ON peaks(ion, rt);

-- convenience view: every peak with its project/run/sample/molecule names
CREATE VIEW IF NOT EXISTS v_peaks AS
SELECT p.*, s.sample_name, s.group_name, s.injection_order,
       r.run_name, r.run_type, pr.project_name, m.molecule_name
FROM peaks p
JOIN samples  s  ON s.sample_id  = p.sample_id
JOIN runs     r  ON r.run_id     = s.run_id
JOIN projects pr ON pr.project_id = r.project_id
LEFT JOIN molecules m ON m.molecule_id = p.molecule_id;
