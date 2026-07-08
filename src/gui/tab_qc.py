"""

QC plot explorer designed to help inform a user as to wether or not they can trust the data in this run

------------------------------ GUI Plot Explorer ------------------------------

    ISTD area                               Boxplot + area vs injection order per sample
    Missingness                             heatmap
    S/N ratio                               violin plot or histogram per molecule
    Tailing factor                          violin plot or histogram per molecule, boxplot per sample
    Peak area before vs after norm          boxplot per sample
    RT drift                                rt diff vs injection order (avg + err per sample)
    % CV                                    Bar plot per molecule, ordered highest %CV to lowest with cutoff line
    Theoretical plates vs inj order         scatter plot to show column degradation as you go
    bl slope                                per-molecule like % CV, also avg vs RT
    outlier count                           count per sample
    mean abs baseline slope                 per-samle boxplot
    Area Drift vs R^2                       scatter plot with thresholds (we want most points near 0,0)
                                            take peak area vs inj order, fit linear regression, slope of that is
                                            the x value y value is the R^2 value of that fit
    gaussain similarity                     bar chart pr mol ordered high to low, violin/boxplot per moleclule
                                            per-sample boxplot and heatmap

                                            
------------------------------ Full QC Report Info------------------------------

Section 1 — Run Summary
    Sample table - all sample information in addition to:
        % Missing
        % Outliers
        Injection Order
    Molecule table - all mol information in addition to:
        % Missing
        Mean Area
        SD Area
        % CV Area
        % Outliers
        Mean RT
        SD RT
        Mean S/N
        SD S/N

Section 2 — Missingness
    Missingness heatmap (global)
    Missingness count bar chart (per molecule, sorted)

Section 3 — Peak Quality
    Gaussian similarity heatmap
    S/N ratio violin per molecule
    Tailing factor violin per molecule

Section 4 — Instrument Health
    Theoretical plates avg vs injection order
    RT drift per molecule (filtered)
    Area drift vs R² scatter

Section 5 — Variability
    % CV bar chart per molecule
    Outlier count per molecule bar chart
    Outlier count per sample bar chart
    Outlier heatmap
    Outlier Table

Section 6 — Normalization
    Peak normalization heatmap (norm/raw)

"""

# region Imports

import logging
import numpy as np

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
                             QLabel, QComboBox, QGroupBox, QFileDialog, QPushButton,
                             QMessageBox, QListWidget)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.plotting import plot_boxplot, plot_scatter, plot_heatmap, plot_violin, plot_histogram, plot_bar, plot_pdf_table
import src.plotting as plotting_module

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    filename="debug.log"
)
logger = logging.getLogger(__name__)

# endregion

class QCTab(QWidget):
    def __init__(self, run_data, parent=None):
        super().__init__(parent)

        # data
        self.run_data = run_data
        self.data_matrix = run_data.data_matrix

        self.standards = list(self.data_matrix.standards)

        self.mol_map = self.data_matrix.mol_map
        self.mol_list = list(self.mol_map.keys())

        self.sample_map = self.data_matrix.sample_map
        self.sample_list = list(self.sample_map.keys())

        self.group_indices = self.data_matrix.group_indices
        self.group_map = self.data_matrix.group_map

        self.inj_map = {k:row['injection_order'] for k,row in self.data_matrix.samples.items()}

        self.data = self.data_matrix.data
        self.outliers = self.data_matrix.outliers
        self.missing = self.data_matrix.missing

        # combined outlier mask
        self.combined_outliers = np.zeros((self.data_matrix.n_samples, self.data_matrix.n_molecules), dtype=bool)
        for metric in self.outliers:
            self.combined_outliers |= self.outliers[metric]

        self.figure_type = None
        self.figure_level = None
        self.plot_type = None
        self.figure_samples = None
        self.figure_molecules = None

        self.figures = {
            'ISTD Area':{
                'per-molecule':['Boxplot', 'Violin Plot'],
                'per-sample':[],
                'per-group':[],
                'global':['Area vs Injection Order']
            },
            'Missingness':{
                'per-molecule':['Ordered Bar Plot'],
                'per-sample':['Ordered Bar Plot'],
                'per-group':[],
                'global':['Heatmap']
            },
            'S/N Ratio':{
                'per-molecule':['Violin Plot', 'Boxplot'],
                'per-sample':['Boxplot', 'Violin Plot'],
                'per-group':[],
                'global':['Heatmap']
            },
            'Tailing':{
                'per-molecule':['Violin Plot', 'Boxplot'],
                'per-sample':['Boxplot', 'Violin Plot'],
                'per-group':[],
                'global':['Heatmap']
            },
            'Peak Normalization':{
                'per-molecule':[],
                'per-sample':['Boxplot', 'Violin Plot'],
                'per-group':[],
                'global':['Heatmap']
            },
            'RT Drift':{
                'per-molecule':['RT Drift vs Injection Order'],
                'per-sample':[],
                'per-group':[],
                'global':['Heatmap','Avg vs Injection Order']
            },
            '% CV Area':{
                'per-molecule':['Ordered Bar Plot', 'Boxplot', 'Violin Plot'],
                'per-sample':['Boxplot', 'Violin Plot'],
                'per-group':[],
                'global':['Heatmap']
            },
            'Theoretical Plates':{
                'per-molecule':['Violin Plot', 'Boxplot'],
                'per-sample':['Violin Plot', 'Boxplot'],
                'per-group':[],
                'global':['Heatmap', 'Avg vs Injection Order']
            },
            'Baseline Slope':{
                'per-molecule':['Ordered Bar Plot'],
                'per-sample':['Boxplot', 'Violin Plot'],
                'per-group':[],
                'global':['Heatmap', 'Avg Slope vs RT']
            },
            'Outlier Count':{
                'per-molecule':['Ordered Bar Plot'],
                'per-sample':['Ordered Bar Plot'],
                'per-group':[],
                'global':[]
            },
            'Area':{
                'per-molecule':['Area vs Injection Order', 'Boxplot'],
                'per-sample':[],
                'per-group':[],
                'global':['Area Drift vs R^2']
            },
            'Gaussian Similarity':{
                'per-molecule':['Ordered Bar Plot', 'Violin Plot', 'Boxplot'],
                'per-sample':['Boxplot', 'Violin Plot'],
                'per-group':[],
                'global':['Heatmap']
            },
            'Flat Top Prevalance': {
                'per-molecule': ['Ordered Bar Plot'],
                'per-sample': ['Ordered Bar Plot'],
                'per-group':[],
                'global': ['Heatmap']
            },
        }

        self.figure_metrics = {
            'ISTD Area': ['Area'],
            'Missingness': ['missing'],
            'S/N Ratio':['SN_Ratio'],
            'Tailing':['Tailing_Factor'],
            'Peak Normalization':['Area','norm_Area'],
            'RT Drift':['RT_Diff'],
            '% CV Area':['Area', 'norm_Area'],
            'Theoretical Plates':['Theoretical_Plates'],
            'Baseline Slope':['bl_slope','RT'],
            'Outlier Count':['outliers'],
            'Area':['Area'],
            'Gaussian Similarity':['gaussian_similarity'],
            'Flat Top Prevalance':['flat'],
        }
        
        self.plot_dispatch = {
            'Heatmap': plot_heatmap,
            'Ordered Bar Plot': plot_bar,
            'Boxplot': plot_boxplot,
            'Violin Plot': plot_violin,
            'Histogram': plot_histogram,
        }

        # widgets
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas,self)

        self.export_full_btn = QPushButton("Export Full Report")
        self.export_custom_btm = QPushButton("Export Custom Report")
        self.export_fig_btn = QPushButton("Export Figure")
        self.plot_btn = QPushButton("Plot Figure")
        self.save_fig_btn = QPushButton("Save Figure")

        self.figure_label = QLabel("Select Figure")
        self.figure_select = QComboBox()

        self.level_label = QLabel("Analysis Level")
        self.level_select = QComboBox()

        self.plot_label = QLabel("Plot Type")
        self.plot_select = QComboBox()

        self.sample_label = QLabel("Select Sample(s)")
        self.sample_select = QListWidget()

        self.molecule_label = QLabel("Select Molecule(s)")
        self.molecule_select = QListWidget()

        self.initUI()

    def initUI(self):

        self.export_fig_btn.clicked.connect(self.export_fig_clicked)
        self.export_full_btn.clicked.connect(self.export_full_clicked)
        self.plot_btn.clicked.connect(self.plot_figure)

        self.level_select.currentTextChanged.connect(self.change_level)
        self.plot_select.currentTextChanged.connect(self.change_plot)
        
        self.molecule_select.setSelectionMode(QListWidget.MultiSelection)
        self.sample_select.setSelectionMode(QListWidget.MultiSelection)

        self.figure_select.addItems(list(self.figures.keys()))
        self.figure_select.currentTextChanged.connect(self.change_fig)
        self.figure_select.setCurrentIndex(0)

        self.top_toolbar = QHBoxLayout()
        self.top_toolbar.addStretch()
        self.top_toolbar.addWidget(self.export_fig_btn)
        self.top_toolbar.addWidget(self.export_full_btn)

        self.fig_box = QGroupBox()
        self.fig_layout = QVBoxLayout()
        self.fig_layout.addWidget(self.toolbar, alignment=Qt.AlignTop | Qt.AlignCenter)
        self.fig_layout.addWidget(self.canvas, stretch=1)
        self.fig_box.setLayout(self.fig_layout)

        self.sam_mol_layout = QVBoxLayout()
        self.figure_select_layout = QVBoxLayout()
        self.level_select_layout = QVBoxLayout()
        self.plot_select_layout = QVBoxLayout()

        self.box = QGroupBox("Figure Select")
        self.select_layout = QVBoxLayout()
        self.figure_select_layout.addWidget(self.figure_label)
        self.figure_select_layout.addWidget(self.figure_select)
        self.select_layout.addLayout(self.figure_select_layout)
        self.select_layout.addStretch()
        self.level_select_layout.addWidget(self.level_label)
        self.level_select_layout.addWidget(self.level_select)
        self.select_layout.addLayout(self.level_select_layout)
        self.select_layout.addStretch()
        self.select_layout.addLayout(self.sam_mol_layout)
        self.select_layout.addStretch()
        self.plot_select_layout.addWidget(self.plot_label)
        self.plot_select_layout.addWidget(self.plot_select)
        self.select_layout.addLayout(self.plot_select_layout)
        self.select_layout.addStretch()
        self.select_layout.addWidget(self.plot_btn)
        self.box.setLayout(self.select_layout)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.box)
        self.splitter.addWidget(self.fig_box)
        self.splitter.setStretchFactor(0,1)
        self.splitter.setStretchFactor(1,2)

        self.outer_layout = QVBoxLayout()
        self.outer_layout.addLayout(self.top_toolbar)
        self.outer_layout.addWidget(self.splitter, stretch=1)

        self.change_fig()
        self.plot_figure()

        self.setLayout(self.outer_layout)

    def export_full_clicked(self):

        path, _ = QFileDialog.getSaveFileName(self, "Export Full Report PDF", "", "PDF Files (*pdf)")
        if not path:
            return
        if not path.endswith('.pdf'):
            path += '.pdf'

        # mol/sample counts
        n_mols = len(self.mol_list)
        n_samples = len(self.sample_list)

        # set figure size based on data size
        fig_width = max(11, n_mols*0.4)
        fig_height = max(8.5,n_samples*0.4)
        fig_width = min(fig_width, 24)
        fig_height = min(fig_height, 24)

        # swich off dark mode for report
        plotting_module.DARK_MODE = False
        
        # create report
        try:
            with PdfPages(path) as pdf:

                # region 1 - Summary Tables

                samples = self.data_matrix.samples
                molecules = self.data_matrix.molecules

                # Sample table
                per_sample_metrics = {
                                        "% Missing": self.missing.mean(axis=1) * 100,
                                        "% Outliers": self.combined_outliers.mean(axis=1) * 100
                                        }
                
                s_rows = []
                s_row_labels = []
                s_col_labels = []
                for sample_name, vals in samples.items():

                    if not s_col_labels:
                        s_col_labels = [k for k in vals.keys() if k!='sample_name' and k!='run_name'] + ['% Missing', '% Outliers']

                    sam_i = self.sample_map[sample_name]

                    row = [self.fmt(vals[k]) for k in vals.keys() if k!='sample_name' and k!='run_name']
                    row = row + [self.fmt(per_sample_metrics['% Missing'][sam_i]), self.fmt(per_sample_metrics['% Outliers'][sam_i])]
                    
                    s_rows.append(row)
                    s_row_labels.append(sample_name)
                
                # save
                for table in plot_pdf_table(s_rows,s_row_labels,s_col_labels,'Samples'):
                    pdf.savefig(table,bbox_inches='tight')
                    plt.close(table)

                # Molecules table
                per_mol_metrics = {
                                    '% Missing': self.missing.mean(axis=0) * 100,
                                    "Mean Area": np.nanmean(self.data['norm_Area'], axis=0),
                                    "SD Area": np.nanstd(self.data['norm_Area'], axis=0),
                                    "% CV Area": (np.nanstd(self.data['norm_Area'], axis=0) / np.nanmean(self.data['norm_Area'], axis=0)) * 100,
                                    "% Outliers": self.combined_outliers.mean(axis=0) * 100,
                                    "Mean RT": np.nanmean(self.data['RT'], axis=0),
                                    "SD RT": np.nanstd(self.data['RT'], axis=0),
                                    "Mean S/N": np.nanmean(self.data['SN_Ratio'], axis=0),
                                    "SD S/N": np.nanstd(self.data['SN_Ratio'], axis=0)
                                    }
                
                m_rows = []
                m_row_labels = []
                m_col_labels = []
                for mol_name, vals in molecules.items():
                    if not m_col_labels:
                        m_col_labels = [k for k in vals.keys() if k!='molecule_name' and k!='run_name' and k!='molID'] + ['% Missing', 'Mean Area', 'SD Area',
                                                                                                                            '% CV Area', '% Outliers', 'Mean RT',
                                                                                                                            'SD RT', 'Mean S/N', 'SD S/N']
                    mol_i = self.mol_map[mol_name]

                    row = [self.fmt(vals[k]) for k in vals.keys() if k!='molecule_name' and k!='run_name' and k!='molID']
                    row = row + [self.fmt(per_mol_metrics['% Missing'][mol_i]), self.fmt(per_mol_metrics['Mean Area'][mol_i]),
                                self.fmt(per_mol_metrics['SD Area'][mol_i]), self.fmt(per_mol_metrics['% CV Area'][mol_i]),
                                self.fmt(per_mol_metrics['% Outliers'][mol_i]), self.fmt(per_mol_metrics['Mean RT'][mol_i]),
                                self.fmt(per_mol_metrics['SD RT'][mol_i]), self.fmt(per_mol_metrics['Mean S/N'][mol_i]),
                                self.fmt(per_mol_metrics['SD S/N'][mol_i])]
                    
                    m_rows.append(row)
                    m_row_labels.append(mol_name)

                # save
                for table in plot_pdf_table(m_rows,m_row_labels,m_col_labels,'Molecules'):
                    pdf.savefig(table,bbox_inches='tight')
                    plt.close(table)

                # endregion

                # region 2 - Missingness

                # plot heatmap (page 1)
                fig,ax = plt.subplots(figsize=(fig_width,fig_height))

                plot_heatmap(ax, self.missing.astype(float), self.sample_list, self.mol_list,
                            title='Missingness Heatmap', cmap='RdYlGn_r')
                
                pdf.savefig(fig,bbox_inches='tight')
                plt.close(fig)

                # plot bar graphs (page 2)
                fig, (ax1,ax2) = plt.subplots(2,1,figsize=(fig_width,11))
                fig.subplots_adjust(hspace=0.5)

                m_counts = [self.missing[:,self.mol_map[m]] for m in self.mol_list]
                plot_bar(ax1, m_counts, self.mol_list, ylabel='Missing Count',
                        title='Missingness Per Molecule', sorted_desc=True, type='count')
                
                s_counts = [self.missing[self.sample_map[s],:] for s in self.sample_list]
                plot_bar(ax2, s_counts, self.sample_list, ylabel='Missing Count',
                        title='Missingnes Per Sample', sorted_desc=True, type='count')
                
                pdf.savefig(fig,bbox_inches='tight')
                plt.close(fig)

                # endregion

                # region 3 - Peak Quality

                # plot gaussian simliarity heatmap (page 1)
                fig,ax = plt.subplots(figsize=(fig_width,fig_height))
                plot_heatmap(ax, self.data['gaussian_similarity'], self.sample_list, self.mol_list,
                            title='Gaussian Similarity', cmap='RdYlGn')
                pdf.savefig(fig,bbox_inches='tight')
                plt.close(fig)

                # plot S/N and Tailing factor violin plots (per molecule)
                fig, (ax1,ax2) = plt.subplots(2,1, figsize=(fig_width, 11))
                cap = np.nanpercentile(self.data['SN_Ratio'],90) if np.max(self.data['SN_Ratio']) > 1000 else 1000

                sn_data = [self.data['SN_Ratio'][:,self.mol_map[m]] for m in self.mol_list]
                sn_data = [np.clip(arr,None,cap) for arr in sn_data]
                plot_violin(ax1, sn_data, self.mol_list, ylabel='S/N Ratio', title='S/N Ratio Per Molecule')

                tf_data = [self.data['Tailing_Factor'][:,self.mol_map[m]] for m in self.mol_list]
                plot_violin(ax2, tf_data, self.mol_list, ylabel='S/N Ratio', title='Tailing Factor Per Molecule')

                pdf.savefig(fig,bbox_inches='tight')
                plt.close(fig)

                # plot gaussian similarity and FWHH violin plots (per molecule)
                fig, (ax1,ax2) = plt.subplots(2,1, figsize=(fig_width, 11))

                gs_data = [self.data['gaussian_similarity'][:,self.mol_map[m]] for m in self.mol_list]
                plot_violin(ax1, gs_data, self.mol_list, ylabel='Gaussian Similarity', title='Gaussian Similarity Per Molecule')

                tf_data = [self.data['FWHH'][:,self.mol_map[m]] for m in self.mol_list]
                plot_violin(ax2, tf_data, self.mol_list, ylabel='FWHH', title='FWHH Per Molecule')

                pdf.savefig(fig,bbox_inches='tight')
                plt.close(fig)

                # plot S/N and Tailing Factor violin plots (per sample)
                fig, (ax1,ax2) = plt.subplots(2,1, figsize=(fig_width, 11))

                sn_data = [self.data['SN_Ratio'][self.sample_map[s],:] for s in self.sample_list]
                sn_data = [np.clip(arr,None,cap) for arr in sn_data]
                plot_violin(ax1, sn_data, self.sample_list, ylabel='S/N Ratio', title='S/N Ratio Per Sample')

                tf_data = [self.data['Tailing_Factor'][self.sample_map[s],:] for s in self.sample_list]
                plot_violin(ax2, tf_data, self.sample_list, ylabel='Tailing Factor', title='Tailing Factor Per Sample')

                pdf.savefig(fig,bbox_inches='tight')
                plt.close(fig)

                # plot gaussian similarity (per sample)
                fig,ax = plt.subplots(1,1, figsize=(fig_width,6))
                gs_data = [self.data['gaussian_similarity'][self.sample_map[s],:] for s in self.sample_list]
                plot_violin(ax, gs_data, self.sample_list, ylabel='Guassian Similarity', title='Gaussian Similarity Per Sample')

                pdf.savefig(fig,bbox_inches='tight')
                plt.close(fig)

                # endregion

                # region 4 - Instrument Health

                fig = plt.figure(figsize=(fig_width,15))
                gs = gridspec.GridSpec(3,1,hspace=0.6)

                ax1 = fig.add_subplot(gs[0])
                ax2 = fig.add_subplot(gs[1])
                ax3 = fig.add_subplot(gs[2])

                # TP avg vs inj order
                tp_data = [self.data['Theoretical_Plates'][self.sample_map[s],:] for s in self.sample_list]
                inj_order = [np.array([self.inj_map[s]]) for s in self.sample_list]     # list of single element arrays
                plot_scatter(ax1, [tp_data,inj_order], labels=self.mol_list,
                            ylabel='Theoretical Plates', xlabel='Injection Order',
                            title='Theoretical Plates vs Injection Order', avg=True, fit=True,
                            annotate_fit=True)
                
                # rt drift per mol (only display ones where RT drift is significant)
                rt_data = [self.data['RT_Diff'][:,self.mol_map[m]] for m in self.mol_list]
                inj_order = np.array([self.inj_map[s] for s in self.sample_list])       # array of injection order
                inj_list = [inj_order for _ in rt_data]

                # generate labels for significant series
                series_labels = {}
                for i, (mol,arr) in enumerate(zip(self.mol_list,rt_data)):
                    valid = arr[~np.isnan(arr)]
                    if len(valid) >= 2 and np.max(valid) - np.min(valid) > 0.025:
                        series_labels[i] = mol

                # plot all lines, only label ones which are significnalty deviant
                plot_scatter(ax2, [rt_data, inj_list], labels=None,
                            ylabel='RT', xlabel='Injection Order',
                            title='RT Drift vs Injection Order',
                            series_labels=series_labels)
                
                # Area Drift vs R^2
                slopes,r2s = [],[]
                for m in self.mol_list:

                    arr = self.data['Area'][:,self.mol_map[m]]
                    mask = ~np.isnan(arr)

                    if mask.sum() < 3:
                        continue

                    coeffs = np.polyfit(inj_order[mask],arr[mask],1)
                    fit_fn = np.poly1d(coeffs)
                    residuals = arr[mask] - fit_fn(inj_order[mask])
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((arr[mask] - np.mean(arr[mask]))**2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

                    slopes.append(coeffs[0])
                    r2s.append(r2)

                plot_scatter(ax3, [[np.array(slopes)], [np.array(r2s)]], labels=None,
                            ylabel='Area Drift Slope', xlabel='R^2',
                            title='Area Drift vs R^2', fit=False)
                
                pdf.savefig(fig,bbox_inches='tight')
                plt.close(fig)

                # endregion

                # region 5 - Normalization

                fig,ax = plt.subplots(figsize=(fig_width, fig_height))
                norm_data = self.data['Area'] / self.data['norm_Area']
                plot_heatmap(ax, norm_data, self.sample_list, self.mol_list,
                            title='Normalization Ratio (Raw / Norm)', cmap='RdYlGn')
                pdf.savefig(fig,bbox_inches='tight')
                plt.close(fig)

                # endregion

                # region 6 - Variability

                # normalized area % CV per group (page 1)
                if not self.group_indices:
                    fig,ax = plt.subplots(figsize=(fig_width,5))
                    cv_data = [self.data['norm_Area'][:,self.mol_map[m]] for m in self.mol_list]
                    plot_bar(ax, cv_data, self.mol_list, ylabel='% CV',
                            title='% CV Normalized Area', sorted_desc=True, type='%CV')
                    pdf.savefig(fig,bbox_inches='tight')
                    plt.close(fig)
                
                # per-group if gropus are specified
                else:
                    group_names = list(self.group_indices.keys())
                    for i in range(0,len(group_names),2):
                        chunk = group_names[i:i+2]
                        fig,axes = plt.subplots(len(chunk),1, figsize=(fig_width,5*len(chunk)))
                        if len(chunk) == 1:
                            axes = [axes]
                        
                        for ax,group_name in zip(axes,chunk):
                            if not group_name:
                                title = "% CV Per Molecule"
                            else:
                                title = f"% CV Per Molecule - {group_name}"
                            indices = self.group_indices[group_name]
                            cv_data = [self.data['norm_Area'][indices, self.mol_map[m]] for m in self.mol_list]
                            plot_bar(ax, cv_data,self.mol_list, ylabel="% CV",
                                    title=title, sorted_desc=True,
                                    type='%CV')
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)

                # Outlier Heatmap (page 2)
                fig,ax = plt.subplots(figsize=(fig_width,fig_height))
                plot_heatmap(ax, self.combined_outliers.astype(float), self.sample_list, self.mol_list,
                            title='Outlier Heatmap', cmap='RdYlGn_r')
                
                pdf.savefig(fig,bbox_inches='tight')
                plt.close(fig)

                # Outlier counts per mol/sample
                fig,(ax1,ax2) = plt.subplots(2,1, figsize=(fig_width, 11))

                mol_outliers = [self.combined_outliers[:,self.mol_map[m]] for m in self.mol_list]
                plot_bar(ax1, mol_outliers, self.mol_list, ylabel='Outlier Count',
                        title='Outlier Count Per Molecule', sorted_desc=True, type='count')
                
                sam_outliers = [self.combined_outliers[self.sample_map[s],:] for s in self.sample_list]
                plot_bar(ax2, sam_outliers, self.sample_list, ylabel='Outlier Count',
                        title='Outlier Count Per Sample', sorted_desc=True, type='count')
                
                pdf.savefig(fig,bbox_inches='tight')
                plt.close(fig)

                # outlier table
                outlier_col_labels = ['Sample','Molecule','Flagged Metrics']
                outlier_rows = []

                for sample_name,row_i in self.sample_map.items():
                    for mol_name,col_i in self.mol_map.items():
                        if not self.combined_outliers[row_i,col_i]:
                            continue
                        flagged = ', '.join(k for k,v in self.outliers.items() if v[row_i,col_i])
                        outlier_rows.append([sample_name, mol_name, flagged])

                if outlier_rows:
                    tables = plot_pdf_table(outlier_rows, col_labels=outlier_col_labels, title='Outliers')
                else:
                    tables = plot_pdf_table([['No Outliers Detected']],col_labels=['Status'], title='Outliers')
                
                for table in tables:
                    pdf.savefig(table,bbox_inches='tight')
                    plt.close(table)

                # endregion
        
        # turn dark mode back on for plotting
        finally:
            plotting_module.DARK_MODE = True

    def fmt(self, val):
        """
        Formats a float value 
        """
        if isinstance(val,(float,np.floating)):
            if np.isnan(val):
                return 'N/A'
            if val == 0:
                return '0'
            elif abs(val) > 10000 or abs(val) < 0.0001:
                return f'{val:.4e}'
            return f'{val:.5g}'
        return str(val) if val is not None else 'N/A'
    
    def export_fig_clicked(self):

        path, _ = QFileDialog.getSaveFileName(self, "Export Full Report PDF", "", "PDF Files (*pdf)")
        if not path:
            return
        if not path.endswith('.pdf'):
            path += '.pdf'

        self.figure.savefig(path, format='pdf', bbox_inches='tight')

    def change_fig(self):
        
        # get the metric type and save
        text = self.figure_select.currentText()
        if not text:
            return
        self.figure_type = text

        # see what levels (per-mol, per-sample, global)
        levels = self.figures[text]
        levels_list = [k for k,v in levels.items() if v]
        self.level_select.clear()
        self.level_select.addItems(levels_list)
        self.level_select.setCurrentIndex(0)

    def change_level(self):

        # get figure level and save
        level = self.level_select.currentText()
        if not level:
            return
        self.figure_level = level

        # clear layout
        while self.sam_mol_layout.count():
            item = self.sam_mol_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
            elif item.layout():
                while item.layout().count():
                    child = item.layout().takeAt(0)
                    if child.widget():
                        child.widget().setParent(None)

        # list availbe samples or molecules
        if level == 'per-molecule':

            if self.figure_type == 'ISTD Area':
                mols = ['All'] + list(self.standards)
            else:
                mols = ['All'] + self.mol_list

            self.molecule_select.clear()
            self.molecule_select.addItems(mols)

            self.mol_layout = QVBoxLayout()
            self.mol_layout.addWidget(self.molecule_label)
            self.mol_layout.addWidget(self.molecule_select)

            self.sam_mol_layout.addLayout(self.mol_layout)

            self.molecule_select.setCurrentRow(0)

        elif level == 'per-sample':

            sams = ['All'] + self.sample_list

            self.sample_select.clear()
            self.sample_select.addItems(sams)

            self.sam_layout = QVBoxLayout()
            self.sam_layout.addWidget(self.sample_label)
            self.sam_layout.addWidget(self.sample_select)

            self.sam_mol_layout.addLayout(self.sam_layout)

            self.sample_select.setCurrentRow(0)

        # list availbe plots
        plot_types_list = self.figures[self.figure_type][level]
        self.plot_select.clear()
        self.plot_select.addItems(plot_types_list)
        self.plot_select.setCurrentIndex(0)

    def change_plot(self):
        
        # seave plot type
        text = self.plot_select.currentText()
        if not text:
            return
        self.plot_type = text

    def plot_figure(self):

        # check for missing values
        if not self.figure_type or not self.figure_level or not self.plot_type:
            QMessageBox.warning(self,'Error','Please specify figure options')
            return

        # clear figure
        self.figure.clear()

        # flags for data gathering
        outlier = False
        missing = False
        all_flag = False

        # get plot title
        title = f"{self.figure_type} {self.plot_type}"

        # get axis for plotting
        ax = self.figure.add_subplot(111)

        # determine metric to use
        metrics = self.figure_metrics[self.figure_type]
        if metrics[0] == 'outliers':
            outlier = True
        elif metrics[0] == 'missing':
            missing = True

        # Collect list of samples/molecules that we are dealing with 
        level = self.figure_level
        data = []

        if self.plot_type == 'Avg vs Injection Order':
            selected = self.sample_list
            for metric in metrics:
                vals = self.get_array(metric, samples=selected, outliers=outlier, missing=missing)
                data.append(vals)

        elif self.plot_type == 'Avg Slope vs RT':
            selected = self.mol_list
            for metric in metrics:
                vals = self.get_array(metric, samples=selected, outliers=outlier, missing=missing)
                data.append(vals)

        elif self.plot_type == 'Area Drift vs R^2':
            selected = self.mol_list
            for metric in metrics:
                vals = self.get_array(metric, mols=selected, outliers=outlier, missing=missing)
                data.append(vals)
        
        elif self.plot_type == 'Area vs Injection Order' and level == 'global':
            if not self.standards:
                QMessageBox.warning("Error", "No standards specified")
                return
            selected = list(self.standards)
            for metric in metrics:
                vals = self.get_array(metric, mols=selected, outliers=outlier, missing=missing)
                data.append(vals)

        elif level == 'global':
            if outlier:
                data.append(self.outliers)
            elif missing:
                data.append(self.missing)
            else:
                for metric in metrics:
                    data.append(self.data[metric])

        elif level == 'per-sample':
            selected = [item.text() for item in self.sample_select.selectedItems()]
            if not selected:
                QMessageBox.warning(self,'Error','Please select sample(s)')
                return
            if 'All' in selected:
                all_flag = True
                selected = self.sample_list
            for metric in metrics:
                vals = self.get_array(metric, samples=selected, outliers=outlier, missing=missing)
                data.append(vals)
        
        elif level == 'per-molecule':
            selected = [item.text() for item in self.molecule_select.selectedItems()]
            if not selected:
                QMessageBox.warning(self,'Error','Please select molecule(s)')
                return
            if 'All' in selected:
                all_flag = True
                if self.figure_type == 'ISTD Area':
                    selected = self.standards
                else:
                    selected = self.mol_list
            for metric in metrics:
                vals = self.get_array(metric, mols=selected, outliers=outlier, missing=missing)
                data.append(vals)

        # unwrap data from list to just a list of arrays (or a matrix if global)
        if self.figure_type == 'Peak Normalization':
            if self.figure_level == 'global':
                data = [data[0] / data[1]]
            else:
                data = [[a/b for a,b in zip(data[0],data[1])]]
                ax.set_yscale('log')

        # setup kwargs for each plotting method
        if self.plot_type == 'Heatmap':
            rev_types = ['Missingness', 'Outlier Count', 'Tailing', '% CV', 'Flat Top Prevalance']
            if self.figure_type in rev_types:
                cmap = "RdYlGn_r"
            elif self.figure_type in ('Baseline Slope', 'RT Drift'):
                cmap = "RdBu_r"
            else:
                cmap = "RdYlGn"

            kwargs = {
                'matrix': data[0],
                'row_labels': self.sample_list,
                'col_labels': self.mol_list,
                'title': title,
                'cmap': cmap
            }
        elif self.plot_type == 'Ordered Bar Plot':
            if self.figure_type == '% CV':
                bar_type = '%CV'
            elif self.figure_type == 'Outlier Count':
                bar_type = 'count'
            elif self.figure_type == 'Missingness':
                bar_type = 'count'
            elif self.figure_type == 'Flat Top Prevalance':
                bar_type = 'count'
            else:
                bar_type = 'avg'
            kwargs = {
                'data': data[0],
                'labels': selected,
                'ylabel': self.figure_type,
                'title': title,
                'cutoff': None,
                'sorted_desc': True,
                'type': bar_type
            }
        elif self.plot_type == 'Boxplot':
            if self.figure_type == 'S/N Ratio':
                cap = np.nanpercentile(self.data['SN_Ratio'],90) if np.max(self.data['SN_Ratio']) > 1000 else 1000
                data[0] = [np.clip(arr,None,cap) for arr in data[0]]
            kwargs = {
                'data': data[0],
                'labels': selected,
                'ylabel': self.figure_type,
                'title': title,
                'cutoff': None
            }
        elif self.plot_type == 'Violin Plot':
            if self.figure_type == 'S/N Ratio':
                cap = np.nanpercentile(self.data['SN_Ratio'],90) if np.max(self.data['SN_Ratio']) > 1000 else 1000
                data[0] = [np.clip(arr,None,cap) for arr in data[0]]
            kwargs = {
                'data': data[0],
                'labels': selected,
                'ylabel': self.figure_type,
                'title': title,
                'color': True
            }
        elif self.plot_type == 'Histogram':
            kwargs = {
                'data': data[0],
                'labels': selected,
                'xlabel': self.figure_type,
                'title': title,
                'bins': 20
            }
        else:
            # flag on wether or not we need to average values and if we want a linear fit
            fit = True
            avg = False
            legend = True
            annotate = False
        
            # get injection order
            inj_list = []
            inj_order = np.zeros(self.data_matrix.n_samples)
            for sample, i in self.sample_map.items():
                inj_order[i] = self.inj_map[sample]

            # handle different scatter plot types
            if self.plot_type == 'Avg vs Injection Order':
                avg = True
                annotate = True
                ylabel = f"Avg {self.figure_type}"
                xlabel = "Injection Order"
                for sample in selected:
                    inj_list.append(np.array([self.inj_map[sample]]))
                data.append(inj_list)

            elif self.plot_type == 'Area vs Injection Order':
                ylabel = "Area"
                xlabel = "Injection order"

                if self.figure_type == "ISTD Area":
                    title = "ISTD Area vs Injection Order"

                if all_flag:
                    filtered_labels = []
                    filtered_data = []
                    for arr,label in zip(data[0],selected):
                        valid = arr[~np.isnan(arr)]
                        if len(valid) < 2 or np.std(valid) < 1e-6:
                            continue
                        filtered_data.append(arr)
                        filtered_labels.append(label)
                    data[0] = filtered_data
                    selected = filtered_labels

                for i in range(len(data[0])):
                    inj_list.append(inj_order)
                data.append(inj_list)

            elif self.plot_type == 'Avg Slope vs RT':
                avg = True
                annotate = True
                ylabel = "Avg Slope"
                xlabel = "RT"

                if all_flag:
                    filtered_labels = []
                    filtered_data = []
                    filtered_data_1 = []
                    for arr,rt_arr,label in zip(data[0],data[1], selected):
                        valid = arr[~np.isnan(arr)]
                        if len(valid) < 2 or np.std(valid) < 1e-6:
                            continue
                        filtered_data.append(arr)
                        filtered_data_1.append(rt_arr)
                        filtered_labels.append(label)
                    data[0] = filtered_data
                    data[1] = filtered_data_1
                    selected = filtered_labels

            elif self.plot_type == 'Area Drift vs R^2':
                xlabel = 'Area Drift'
                ylabel = 'R^2'
                title = 'Area Drift vs R^2'
                selected = self.mol_list
                fit = False
                legend = False
                
                slopes = []
                r2s = []
                
                for array in data[0]:

                    mask = ~np.isnan(array)

                    if mask.sum() < 3:
                        slopes.append(np.nan)
                        r2s.append(np.nan)
                    
                    x = inj_order[mask]
                    y = array[mask]

                    coeffs = np.polyfit(x,y,1)
                    fit_fn = np.poly1d(coeffs)
                    residuals = y - fit_fn(x)
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((y - np.mean(y))**2)
                    r2 = 1- (ss_res / ss_tot) if ss_tot != 0 else np.nan

                    slopes.append(coeffs[0])
                    r2s.append(r2)

                data = [[np.array(r2s)], [np.array(slopes)]]

            elif self.plot_type == 'RT Drift vs Injection Order':
                ylabel = 'RT Drift'
                xlabel = 'Injection Order'

                if all_flag:
                    print(f"before filter: {len(data[0])} molecules")
                    filtered_data = []
                    filtered_labels = []
                    for arr,label in zip(data[0],selected):
                        valid = arr[~np.isnan(arr)]
                        if len(valid) < 2 or np.std(valid) < 0.01:
                            continue
                        filtered_data.append(arr)
                        filtered_labels.append(label)
                    data[0] = filtered_data
                    selected = filtered_labels

                for i in range(len(data[0])):
                    inj_list.append(inj_order)
                data.append(inj_list)
            
            # build kwargs
            kwargs = {
                'data': data,
                'labels': selected,
                'ylabel': ylabel,
                'xlabel': xlabel,
                'title': title,
                'fit': fit,
                'avg': avg,
                'legend': legend,
                'annotate_fit': annotate
            }
        
        # call plotting handlers and plot samples
        handler = self.plot_dispatch.get(self.plot_type)
        if handler:
            handler(ax, **kwargs)
        else:
            plot_scatter(ax, **kwargs)

        self.canvas.draw()

    def get_array(self, metric, samples=[], mols=[], outliers=False, missing=False):
        """
        Gets the array from the data matrix (outliers=False) or outlier matrices (outliers=True)
        for a given metric for either a molecule or sample of interest
        """

        if outliers:
            data = np.any(np.stack(list(self.outliers.values())), axis=0)
        elif missing:
            data = self.missing
        else:
            data = self.data[metric]

        if samples and mols:
            QMessageBox.warning(self,'Error',"Specify either sample or molecule, not both")
        
        output = []
        if samples:
            for sample in samples:
                row_i = self.sample_map[sample]
                arr = data[row_i,:]
                output.append(arr.astype(float) if (missing or outliers) else arr)
        elif mols:
            for mol in mols:
                col_i = self.mol_map[mol]
                arr = data[:,col_i]
                output.append(arr.astype(float) if (missing or outliers) else arr)
        else:
            QMessageBox.warning(self,'Error',"Must either specify sample or molecule")
            return
        
        return output
    
