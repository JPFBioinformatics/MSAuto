"""

Automatic per-run QC plots designed to inform the user on wether or not they can trust the data from this run

Plots include:
    ISTD area                               Boxplot + area vs injection order per sample
    Missingness                             heatmap
    S/N ratio                               violin plot or histogram per molecule
    Tailing factor                          violin plot or histogram per molecule, boxplot per sample
    Peak area before vs after norm          boxplot per sample
    RT drift                                rt diff vs injection order (avg + err per sample)
    % CV                                    Bar plot per molecule, ordered highest %CV to lowest with cutoff line
    Theoretical plates vs inj order         scatter plot to show column degradation as you go
    bl slope                                per-molecule like % CV, also vs RT
    outlier count                           count per sample
    mean abs baseline slope                 per-samle boxplot
    area/inj order slope vs R^2             scatter plot with thresholds (we want most points near 0,0)
    gaussain similarity                     bar chart pr mol ordered high to low, violin/boxplot per moleclule
                                            per-sample boxplot and heatmap

Plotting methods needed:
    boxplot
    scatter plot
    heatmap
    violin plot
    histogram
    bar plot

"""

# region Imports

import logging
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (QWidget, QTableWidget, QVBoxLayout, QHBoxLayout, QSplitter,
                             QLabel, QComboBox, QHeaderView, QSizePolicy, QTableWidgetItem,
                             QGroupBox, QStyledItemDelegate, QFileDialog, QPushButton, QMenu,
                             QMessageBox, QListWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QBrush

import matplotlib.cm as cm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

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

        self.mol_map = self.data_matrix.mol_map
        self.mol_list = list(self.mol_map.keys())

        self.sample_map = self.data_matrix.sample_map
        self.sample_list = list(self.sample_map.keys())

        self.data = self.data_matrix.data
        self.outliers = self.data_matrix.outliers
        self.missing = self.data_matrix.missing

        self.figure_type = None
        self.figure_level = None
        self.figure_plot = None
        self.figure_samples = None
        self.figure_molecules = None
        
        self.figures = {
            'ISTD Area':{
                'per-molecule':[],
                'per-sample':[],
                'global':[]
            },
            'Mmissingness':{
                'per-molecule':[],
                'per-sample':[],
                'global':[]
            },
            'S/N Ratio':{
                'per-molecule':[],
                'per-sample':[],
                'global':[]
            },
            'Tailing':{
                'per-molecule':[],
                'per-sample':[],
                'global':[]
            },
            'Peak Normalization':{
                'per-molecule':[],
                'per-sample':[],
                'global':[]
            },
            'RT Drift':{
                'per-molecule':[],
                'per-sample':[],
                'global':[]
            },
            '% CV':{
                'per-molecule':[],
                'per-sample':[],
                'global':[]
            },
            'Theoretical Plates':{
                'per-molecule':[],
                'per-sample':[],
                'global':[]
            },
            'Baseline Slope':{
                'per-molecule':[],
                'per-sample':[],
                'global':[]
            },
            'Outlier Count':{
                'per-molecule':[],
                'per-sample':[],
                'global':[]
            },
            'Area / Injection Order':{
                'per-molecule':[],
                'per-sample':[],
                'global':[]
            },
            'Gaussian Similarity':{
                'per-molecule':[],
                'per-sample':[],
                'global':[]
            },
        }

        # widgets
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas,self)

        self.export_full_btn = QPushButton("Export Full Report")
        self.export_fig_btn = QPushButton("Export Figure")

        self.select_label = QLabel("Select Figure")
        self.figure_select = QComboBox()

        self.level_label = QLabel("Analysis Level")
        self.level_select = QComboBox()

        self.plot_label = QLabel("Plot Type")
        self.plot_select = QComboBox()

        self.sample_label = QLabel("Select Sample")
        self.sample_select = QListWidget()

        self.molecule_label = QLabel("Select Molecule")
        self.molecule_select = QListWidget()

        self.initUI()

    def initUI(self):

        self.export_fig_btn.clicked.connect(self.export_fig_clicked)
        self.export_full_btn.clicked.connect(self.export_full_clicked)

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
        self.fig_layout.addWidget(self.canvas)
        self.fig_box.setLayout(self.fig_layout)

        self.sam_mol_layout = QHBoxLayout()

        self.box = QGroupBox()
        self.select_layout = QVBoxLayout()
        self.select_layout.addWidget(self.select_label)
        self.select_layout.addWidget(self.figure_select)
        self.select_layout.addStretch()
        self.select_layout.addLayout(self.sam_mol_layout)
        self.select_layout.addStretch()
        self.select_layout.addWidget(self.plot_label)
        self.select_layout.addWidget(self.plot_select)
        self.select_layout.addStretch()
        self.box.setLayout(self.select_layout)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.box)
        self.splitter.addWidget(self.fig_box)
        self.splitter.setStretchFactor(0,1)
        self.splitter.setStretchFactor(1,1)

        self.outer_layout = QVBoxLayout()
        self.outer_layout.addLayout(self.top_toolbar)
        self.outer_layout.addWidget(self.splitter)

        self.setLayout(self.outer_layout)

    def export_full_clicked(self):

        path, _ = QFileDialog.getSaveFileName(self, "Export Full Report PDF", "", "PDF Files (*pdf)")
        if not path:
            return
        if not path.endswith('.pdf'):
            path += '.pdf'

    def export_fig_clicked(self):

        path, _ = QFileDialog.getSaveFileName(self, "Export Full Report PDF", "", "PDF Files (*pdf)")
        if not path:
            return
        if not path.endswith('.pdf'):
            path += '.pdf'

    def change_fig(self):
        
        # get the metric type and save
        text = self.figure_select.currentText()
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
        self.figure_level = level

        # setup level layout
        self.level_layout = QVBoxLayout()
        self.level_layout.addWidget(self.level_label)
        self.level_layout.addWidget(self.level_select)
        self.sam_mol_layout.addLayout(self.level_layout)

        # list availbe samples or molecules
        if level == 'per-molecule':

            mols = ['All'] + self.mol_list
            self.molecule_select.addItems(mols)

            self.mol_layout = QVBoxLayout()
            self.mol_layout.addWidget(self.molecule_label)
            self.mol_layout.addWidget(self.molecule_select)

            self.sam_mol_layout.addLayout(self.mol_layout)

            self.molecule_select.setCurrentIndex(0)

        elif level == 'per-sample':

            sams = ['All'] + self.sample_list
            self.sample_select.addItems(sams)

            self.sam_layout = QVBoxLayout()
            self.sam_layout.addWidget(self.sample_label)
            self.sam_layout.addWidget(self.mol_layout)

            self.sam_mol_layout.addLayout(self.sam_layout)

            self.sample_select.setCurrentIndex(0)

        # list availbe plots
        plot_types_list = self.figures[self.figure_type][level]
        self.plot_select.clear()
        self.plot_select.addItems(plot_types_list)
        self.plot_select.setCurrentIndex(0)

    def change_sample(self):
        self.figure_molecules = None
        selected = []
        for item in self.sample_select.selectedItems():
            if item == 'All':
                selected = ['All']
            selected.append(item.text())
        self.figure_samples = selected

    def change_molecule(self):
        self.figure_samples = None
        selected = [item.text() for item in self.molecule_select.selectedItems()]

    def change_plot(self):
        
        # seave plot type
        self.plot_type = self.plot_select.currentText()

    def submit_clicked(self):
        pass

    def get_array(self, metric, sample=None, mol=None, outliers=False):
        """
        Gets the array from the data matrix (outliers=False) or outlier matrices (outliers=True)
        for a given metric for either a molecule or sample of interest
        """
        if outliers:
            data = self.outliers[metric]
        else:
            data = self.data[metric]

        if sample and mol:
            raise ValueError("Specify either sample or molecule, not both")
        
        elif sample:
            row_i = self.sample_map[sample]
            return data[row_i,:]
        elif mol:
            col_i = self.mol_map[mol]
            return data[:,col_i]
        else:
            raise ValueError("Must either specify sample or molecule")
