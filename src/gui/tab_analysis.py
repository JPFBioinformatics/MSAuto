"""

Tab for running analyses on the datasets and creating custom figures.

For advanced analysis techniques, not basic data visualization

General data cleaning workflow:
    Drop Sparse Features -> Impute NANs -> log2 transform -> autoscale


"""

# region Imports

import logging
import numpy as np

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QScrollArea, QSpinBox,
                             QLabel, QComboBox, QGroupBox, QFileDialog, QPushButton, QLineEdit,
                             QMessageBox, QListWidget, QDoubleSpinBox,QStackedWidget)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from src.plotting import plot_pca, plot_scree_bar
import src.plotting as plotting_module

from src.analysis import pca

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
        self.matrix = self.data_matrix.data['clean_Area']

        self.n_samples = self.data_matrix.n_samples
        self.sample_map = self.data_matrix.sample_map
        self.sample_list = self.data_matrix.sample_list
        self.samples = self.data_matrix.samples

        self.n_molecules = self.data_matrix.n_molecules
        self.mol_map = self.data_matrix.mol_map
        self.mol_list = self.data_matrix.mol_list
        self.molecules = self.data_matrix.molecules

        self.clean_mol_map = self.data_matrix.clean_mol_map
        self.clean_mol_list = self.data_matrix.clean_mol_list

        self.standards = list(self.data_matrix.standards)

        self.group_map = self.data_matrix.group_map
        self.group_indices = self.data_matrix.group_indices
        self.group_list = list(self.group_indices.keys())

        # figure layout
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas,self)

        self.scroll_area = QScrollArea()

        self.data_type_label = QLabel("Data Type")
        self.data_select = QComboBox()
        self.dtype = 'clean_Area'

        self.export_analysis_btn = QPushButton("Export Analysis")
        self.analyze_btn = QPushButton("Analyze")

        # analysis handling
        self.analysis_types = [
            'PCA',
            'PLS-DA',
            'Correlation Heatmap',
            'T-Test',
            'Mann-Whitney U Test',
            'One-Way ANOVA',
            'Kruskal-Wallis',
            'Clustering'
        ]
        self.analysis_selector = QComboBox()
        self.stack = QStackedWidget()

        # pca
        self.pca_data = None
        self.pca_exp_var = 0.80
        
        self.initUI()

    def initUI(self):

        # setup data type selector
        self.data_select.addItems(['Area','Height'])
        self.data_select.currentTextChanged.connect(self.set_matrix)
        self.set_matrix()

        self.analyze_btn.clicked.connect(self.run_analysis)

        # setup tab stack
        self.stack.addWidget(self.build_pca_controls())
        self.stack.addWidget(self.build_plsda_controls())
        self.stack.addWidget(self.build_correlation_controls())
        self.stack.addWidget(self.build_ttest_controls())
        self.stack.addWidget(self.build_mannwihtney_controls())
        self.stack.addWidget(self.build_anova_controls())
        self.stack.addWidget(self.build_kruskal_controls())
        self.stack.addWidget(self.build_clustering_controls())

        # set analysis selector list
        self.analysis_selector.addItems(self.analysis_types)
        self.analysis_selector.currentIndexChanged.connect(self.stack.setCurrentIndex)

        # scroll area
        self.scroll_area.setWidget(self.canvas)
        self.scroll_area.setWidgetResizable(True)

        # top toolbar layout
        toolbar_layout = QHBoxLayout()
        toolbar_layout.addWidget(self.data_select)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.export_analysis_btn)

        # layout for plot contorls
        plot_box = QGroupBox("Plot Controls")
        plot_control_layout = QVBoxLayout()
        plot_control_layout.addWidget(self.stack)
        plot_control_layout.addWidget(self.analyze_btn)
        plot_box.setLayout(plot_control_layout)

        # layout for figure
        fig_box = QGroupBox()
        fig_layout = QVBoxLayout()
        fig_layout.addWidget(self.toolbar, alignment=Qt.AlignTop | Qt.AlignCenter)
        fig_layout.addWidget(self.scroll_area, stretch=1)
        fig_box.setLayout(fig_layout)

        # splitter for plot controls/figure
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(plot_box)
        splitter.addWidget(fig_box)
        splitter.setStretchFactor(0,1)
        splitter.setStretchFactor(1,2)

        # global layout
        global_layout = QVBoxLayout()
        global_layout.addLayout(toolbar_layout)
        global_layout.addWidget(splitter)

        self.setLayout(global_layout)

    def set_matrix(self):
        label = self.data_select.currentText()
        if label == 'Area':
            self.dtype = 'clean_Area'
        else:
            self.dtype = 'clean_Height'
        self.matrix = self.data_matrix.data[self.dtype]

    def run_analysis(self):
        idx = self.stack.currentIndex()
        runners = [
            self.run_pca,
            self.run_plsda,
            self.run_correlation,
            self.run_ttest,
            self.run_mannwhitney,
            self.run_anova,
            self.run_kruskal,
            self.run_clustering
        ]
        runners[idx]()

    # region Layout Methods

    def build_pca_controls(self):

        # parent widget/layout
        self.pca_controls = QWidget()
        layout = QVBoxLayout(self.pca_controls)

        # PCA plot selections
        self.pca_label = QLabel("PCA Plot Controls")
        layout.addWidget(self.pca_label)
        layout.addStretch()

        self.select_pcx = QSpinBox()
        self.select_pcx.setMinimumWidth(80)
        self.select_pcx.setValue(1)
        self.select_pcy = QSpinBox()
        self.select_pcy.setMinimumWidth(80)
        self.select_pcy.setValue(2)
        pcx_pcy_layout = QHBoxLayout()
        pcx_pcy_layout.addWidget(self.select_pcx)
        pcx_pcy_layout.addWidget(self.select_pcy)

        layout.addLayout(pcx_pcy_layout)
        layout.addStretch()

        self.pca_title = QLineEdit()
        self.pca_title.setPlaceholderText("Title (optional)")
        layout.addWidget(self.pca_title)
        layout.addStretch()

        self.pca_exp_var_select = QDoubleSpinBox()
        self.pca_exp_var_select.setRange(0.0,1.0)
        self.pca_exp_var_select.setSingleStep(0.05)
        self.pca_exp_var_select.setValue(0.80)
        layout.addWidget(self.pca_exp_var_select)
        layout.addStretch()

        return self.pca_controls

    def run_pca(self):

        pcx = self.select_pcx.value()
        pcy = self.select_pcy.value()
        exp_var = self.pca_exp_var_select.value()
        title = self.pca_title.text()

        if pcx == pcy:
            QMessageBox.warning("Error", "Must Plot different PCs against each other")
            return

        # if pca not calculated then recaculate
        if self.pca_data is None:
            self.pca_data = pca(self.matrix,exp_var)

        # recalculate cutoff point if needed
        if exp_var != self.pca_exp_var:
            idx = int(np.searchsorted(self.pca_data['cumulative_variance'], exp_var))+1
            self.pca_exp_var = exp_var
            self.pca_data['n_for_exp_var'] = idx

        # plot figures
        self.figure.clear()
        gs = GridSpec(2,2,figure=self.figure,hspace=0.4)
        ax_scores = self.figure.add_subplot(gs[0,:])        # top row
        ax_scree_exp = self.figure.add_subplot(gs[1,0])     # bottom left
        ax_scree_cum = self.figure.add_subplot(gs[1,1])     # bottom right

        # adjust scroll bar size
        n_plots = 2
        fig_height = n_plots * 5
        self.figure.set_size_inches(self.figure.get_size_inches()[0],fig_height)
        self.canvas.setMinimumHeight(int(fig_height * self.figure.dpi))

        plot_pca(ax_scores, self.pca_data, self.group_indices, title=title, pc_x=pcx, pc_y=pcy)
        plot_scree_bar(ax_scree_exp,self.pca_data['explained_variance'], 
                       title="Explained Variance")
        plot_scree_bar(ax_scree_cum, self.pca_data['cumulative_variance'], 
                       title='Cumulative Variance', cutoff=self.pca_exp_var)

    def build_plsda_controls(self):
        return QWidget()

    def run_plsda(self):
        pass

    def build_correlation_controls(self):
        return QWidget()

    def run_correlation(self):
        pass

    def build_ttest_controls(self):
        return QWidget()

    def run_ttest(self):
        pass

    def build_mannwihtney_controls(self):
        return QWidget()

    def run_mannwhitney(self):
        pass

    def build_anova_controls(self):
        return QWidget()

    def run_anova(self):
        pass

    def build_kruskal_controls(self):
        return QWidget()

    def run_kruskal(self):
        pass

    def build_clustering_controls(self):
        return QWidget()

    def run_clustering(self):
        pass


    # endregion
