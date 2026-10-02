=============================================== LICENCE ===============================================
MSAuto
Copyright (C) 2026 Case Western Reserve University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.


=============================================== DOWNLOAD ===============================================

1. Go to [Releases](https://github.com/JPFBioinformatics/MSAuto/releases) and download the latest `MSAuto-vX.Y.Z-win64.zip`
2. Unzip it anywhere (e.g. Documents)
3. Run `MSAuto.exe` inside the unzipped folder
4. If Windows shows "Windows protected your PC", click **More info → Run anyway**
   (the app is not code-signed yet)

Your projects and results are stored in `%LOCALAPPDATA%\GCMS_Automation\`.
Input files must be `.mzML` (convert Agilent `.D` folders with ProteoWizard msconvert first).

## Run from source

    conda env create -f environment.yaml
    conda activate MSAuto
    python main.py

=============================================== Project Summary ===============================================

MSAuto is an application that addresses current shortcomings in available data analysis
packages for GC-MS, both for untargeted metabolomics and for targeted data gathering.  Current 
automation packages do not center the quality control of peaks, leaving this up to the user
and treating the data collection as somewhat of a black box, just taking a peak as specified. The
goal of MSAuto is to create a comprehensive QC report with each run, and handle data storage for
various projects made of many individual runs (collections of samples) to allow for easy data
organization and retrevial for paper production.

There is a built in, lightweight project management system.  Projects are organized into groups of
runs, where each run is a set of samples processed together/ran on the machine togheter. The current build
works on a targeted set of metabolites specified by a user, but future builds will also enable untargeted
analysis where fetaures are built from groups of peaks matched by retention time across ion traces,
then grouped with most similar features across all samples.  Features can then be identified and 
quantified, building a dataset of metabolites as the samples are processed.  Batch-based analyses
will also be implemented, allowing for the selection of several runs with overlapping groups which can
then be batch-corrected to produce larger, consistent datasets for downstream analyses.

QC metrics are centered around detecting molecules, samples, and individual features that are 
outliers in the dataset, as well as general dataset peak quality based on standard metrics such 
as signal to noise ration (S/N), full-width at half-height (FWHH), peak symmetry (Tailing factor),
in addition to other metrics undelrined in the QC Metrics section.  RT drift and peak quality is
also analyzed with respect to injection order in order to investigate column integrity and give
users an idea of when it is time to cut, adjust, or replace columns in their machine.

Once peaks are picked and quality is checked, data is stored in a hybrid SQL/h5 file format for
easy visualization and retrevial in the MSAuto GUI, allowing users to investigate the peak and
chromatogram quality manually as well.

LIghtweight data analysis techniques are also available for the output data matrix.  PCA to see how
clusters arise within the dataset help inform wether or not the data is structured as one would
expect (case/control clustering seperately for example). In addition to MSEA (metabolite set
enrichment analyiss), which just applies the concepts of GSEA to metabolite sets defined by users
or by metabolomic pathways informed by KEGG networks.  Pathway analysis as well as network generation
and topological analysis are the end goal of generating these feature matrices, although having
a limited feature set as is common in metabolomics makes this more difficult and potentially less
informative as these same techniques when applied to transcriptomic datasets.

The goal of MSAuto is to streamline data collection for GC-MS in a way that allows users
to have control over the set of peaks being picked and full access to a robust QC system 
that can quantify any issues with the chromatography, allowing for more consistent results and 
better machine management. The untargeted metabolomics and pathway analysis applications are in
addition to this, but due to the nature of GC-MS being applied more commonly to targeted analysis
the QC and peak picking are centered in this application.

