=============================================== Project Summary ===============================================

MSAuto is an application that addresses current shortcomings in available data analysis
packages for GC-MS, both for untargeted metabolomics and for targeted data gathering.  Current 
automation packages do not center the quality control of peaks, leaving this up to the user
and treating the data collection as somewhat of a black box, just taking a peak as specified. The
goal of MSAuto is to create a comprehensive QC report with each run, and handle data storage for
various projects made of many individual runs (collections of samples) to allow for easy data
organization and retrevial for paper production.

QC metrics are centered around detecting molecules, samples, and individual features that are 
outliers in the dataset, as well as general dataset peak quality based on standard metrics such 
as signal to noise ration (S/N), full-width at half-height (FWHH), peak symmetry (Tailing factor),
in addition to other metrics undelrined in the QC Metrics section.  RT drift and peak quality is
also analyzed with respect to injection order in order to investigate column integrity and give
users an idea of when it is time to cut, adjust, or replace columns in their machine.

Once peaks are picked and quality is checked, data is stored in a hybrid SQL/h5 file format for
easy visualization and retrevial in the MSAuto GUI, allowing users to investigate the peak and
chromatogram quality manually as well.

Basic data analysis techniques are also available for the output data matrix.  PCA to see how
clusters arise within the dataset help inform wether or not the data is structured as one would
expect (case/control clustering seperately for example). In addition to MSEA (metabolite set
enrichment analyiss), which just applies the concepts of GSEA to metabolite sets defined by users
or by metabolomic pathways informed by KEGG networks.  Pathway analysis as well as network generation
and topological analysis are the end goal of generating these feature matrices, although having
a limited feature set as is common in metabolomics makes this more difficult and potentially less
informative as these same techniques when applied to transcriptomic datasets.

The end goal of MSAuto is to streamline data collection for GC-MS in a way that allows users
to have control over the set of peaks being picked and full access to a robust QC system 
that can quantify any issues with the chromatography, allowing for more consistent results and 
better machine management. The untargeted metabolomics and pathway analysis applications are in
addition to this, but due to the nature of GC-MS being applied more commonly to targeted analysis
the QC and peak picking are centered in this application.

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
        measurments at 10% of peak height (BL adjusted)
            close to 1 is good, >1 is tailing <1 is fronting
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
