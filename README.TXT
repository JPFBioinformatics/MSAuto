
================================================= Data Storage =================================================

                                     ---------- IMPORTANT NOTES ----------

make sure that each sample has its own, unique name for this run
    samples that are processed in different runs/batches can be called different things

                                    ---------- General Structure ----------

In an effort to produce a reasonable GUI we need to address data structures for storage
our system is going to be a set of partitioned SQL databases linked to h5 files, each run/sample set
will have its own folder under root/databases, which will include a .h5 file and sql database

peaks dict in IntesntiyMatrix is structured as

    Sample1_name: 
        ion1: Peak_List
        ion2: Peak_list
        etc...
    Sample2_name:
        ion1: Peak_list
        etc...
    etc...

 --------------------------------------- .h5 file ---------------------------------------
intensity_matrices:
    sample_name_1:
        intensity_matrix - matrix
        baseline_matrix - matrix                    (bool mas for intensity_matrix for S/N calculation)
        unique_mz - list
        time_map - 2d array                         (scan_idx, scan_time)
        sn_threshold - list                         (per row signal/noise threshold)
        noise_factor - float                        (might remove this one)
        baseline_arrays:
            ion1_centerscan - array
            ion2_cetnerscan - array
            etc...
    sample_name_2:
        etc...
    etc...

data_matrces:
    data_matricx - matrix                           (however many matrices we want keyed on name)
    sample_map - 2d array                           (sample_name, data_matrix idx)
    mol_map - 2d array                              (mol_name, data_matrix idx)
    norm_method - string                            how we normlaized, mass, volume, protein etc...
    imputation_method_method - string               what we used to impute missing data

 ------------------------------------ SQL database  ------------------------------------


    intensity_matrices TABLE
        imID                                        (primary key)
        sample_name                                 (FK to samples)
        sample_type                                 (sim or scan)
        n_ions
        n_timepoints

    peaks TABLE
        peakID                                      (primary key)
        imID                                        (FK to intensity_matrices)
        featID                                      (FK to features)
        center_scan                                 (center, l and r bounds all in scan idx NOT time)
        left_bound
        right_bound
        precise_max_location
        height                                     (baseline corrected)
        area                                       (baseline corrected)
        sn_ratio
        molecule
        ion

    features TABLE      
        featID                                      (primary key, h5 key)
        feat_rt
        collection_ion
        identification

    samples TABLE       
        sample_name                                 (Primary Key)
        mouseID
        group
        norm_factor
        norm_factor_type
        injection_order

    molecules TABLE     
        moleculeID                                  (primary key)
        molecule_name
        ion
        rt
        standard

================================================= QC Metrics =================================================

In an effort to ensure that peaks are picked correctly and chromatography is quality we implement robust
QC metrics/plots and use them to flag peaks of interest for manual review

 --------------------------------- Math Reminders  ---------------------------------

    Symmetry - tailing factor
        T = (a + b) / 2a
        a = distance from peak front to max
        b = distance from peak max to tail
    theoretical plates - measures column efficiency
        N = 5.545 * (RT/FWHH)**2

 ------------------------------------ Meterics  ------------------------------------

Per-Peak:
    area
    height
    FWHH
    fwhh/height
    S/N ratio
    RT
    RT_diff
    symmetry
    conv/sharpness
    width

per molecule
    % CV across samples per group
    % missing
    avg/err RT
    %CV S/N
    avg/err FWHH
    avg/err symmetry
    % intensity outliers
    % RT outliers
    avg/err theoretical plates

per sample
    % missing
    %CV S/N
    injection order vs intensity
    % intensity outliers
    % RT outliers
    avg/err RT
    avg FWHH

global
    ISTD %CV
    RT drift vs inj order
    avg/err theoretical plates vs inj order
    % missing

What each peak needs in peak_dict:
    area                                            (trapz integration)
    height                                          (baseline corrected)
    FWHH
    S/N ratio
    RT
    Symmetry
    sharpness
    left_bound
    right_bound
    center
    flat_top
