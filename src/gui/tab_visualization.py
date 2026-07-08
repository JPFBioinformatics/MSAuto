"""

Tab for primitive visualization of datasets, allows user to select a plot type (heatmap, histogram, violin plot, etc...),
a metric to plot, and wether it is per-sample/per-molecule/global as makes sense

Also supports metric vs metric hexplots and scatter plots

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

class AnalysisTab(QWidget):
    def __init__(self, run_data, parent=None):
        super().__init__(parent)

        # data
        self.run_data = run_data

        self.data_matrix = self.run_data.data_matrix

        self.n_samples = self.data_matrix.n_samples
        self.sample_map = self.data_matrix.sample_map
        self.sample_list = list(self.sample_map.keys())
        self.samples = self.data_matrix.samples

        self.n_molecules = self.data_matrix.n_molecules
        self.mol_map = self.data_matrix.mol_map
        self.mol_list = list(self.mol_map.keys())
        self.molecules = self.data_matrix.molecules

        self.standards = list(self.data_matrix.standards)

        self.group_map = self.data_matrix.group_map
        self.group_indices = self.data_matrix.group_indices
        self.group_list = list(self.group_indices.keys())

        # widgets
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas,self)

        self.export_fig_btn = QPushButton("Export Figure")
        self.plot_btn = QPushButton("Plot Figure")

        self.plot_label = QLabel("Analysis Type")
        self.plot_select = QComboBox()

        self.level_label = QLabel("Analysis Level")
        self.level_select = QComboBox()

        self.sample_label = QLabel("Select Sample(s)")
        self.sample_select = QListWidget()

        self.molecule_label = QLabel("Select Molecule(s)")
        self.molecule_select = QListWidget()

        # analysis handling
        self.plot_types = [
            'Scatter',
            'Boxplot',
            'Violin Plot',
            'Heatmap',
            'Histogram',
            'Hexplot',
            'Bar Plot'
        ]
        
        self.initUI()

    def initUI(self):
        pass
