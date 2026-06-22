"""

Allows a user to investigate a run's data matrix, displays area/height matrices and color
codes cells if they are outliers with respect to the different QC metrics as follows:

    | METRIC                      | COLOR                       | Type
----|-----------------------------|-----------------------------|-------------
    | area/height                 | Orange                      | Molecule
    | FWHH                        | Blue                        | Molecule
    | RT                          | Red                         | Molecule
    | Tailing Factor              | Yellow                      | Molecule
    | Conv/Sharpness              | Purple                      | Molecule
    | S/N Ratio                   | Green                       | Molecule
    | Theoretical Plates          | Grey                        | Molecule                   

"""

# region Imports

import sys, logging
import numpy as np

from PyQt5.QtWidgets import (QWidget, QTableWidget, QVBoxLayout, QHBoxLayout, QSplitter,
                             QLabel, QPushButton, QMessageBox, QComboBox, QApplication, 
                             QHeaderView, QSizePolicy, QTableWidgetItem, QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.cm as cm

from src.plotting import plot_chromatogram, plot_peak, plot_spectrum
from src.run_data import RunData

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    filename="debug.log"
)
logger = logging.getLogger(__name__)

# endregion

# Main data table
class DataTab(QWidget):
    def __init__(self, run_data, chrom_tab, parent):
        super().__init__(parent)

        # save relevant run data
        self.chrom_tab = chrom_tab
        self.run_data = run_data
        self.data_matrix = run_data.data_matrix
        self.n_samples = self.data_matrix.n_samples
        self.sample_map = self.data_matrix.sample_map
        self.n_molecules = self.data_matrix.n_molecules
        self.mol_map = self.data_matrix.mol_map
        self.data_metrics = list(self.data_matrix.data.keys())
        self.group_map = self.data_matrix.group_map
        self.group_indices = self.data_matrix.group_indices
        groups = list(set(self.group_indices.keys()))
        self.cmap = cm.get_cmap('rainbow', len(groups))
        self.group_colormap = {group:self.cmap(i) for i,group in enumerate(groups)}
        self.std_color = self.cmap(len(groups))

        # get standards
        self.stds = []
        for row in self.data_matrix.molecules.values():
            if row['std'] not in self.stds:
                self.stds.append(row['std'])

        # build combined outlier mask
        self.combined_outliers = np.zeros((self.n_samples, self.n_molecules), dtype=bool)
        for metric in self.data_matrix.outliers:
            self.combined_outliers |= self.data_matrix.outliers[metric]
        
        # setup per molcule metrics
        self.per_mol_metrics = {
                                '% Missing': self.data_matrix.missing.mean(axis=0) * 100,
                                "Mean Area": np.nanmean(self.data_matrix.data['Area'], axis=0),
                                "SD Area": np.nanstd(self.data_matrix.data['Area'], axis=0),
                                "% CV Area": (np.nanstd(self.data_matrix.data['Area'], axis=0) / np.nanmean(self.data_matrix.data['Area'], axis=0)) * 100,
                                "% Outliers": self.combined_outliers.mean(axis=0) * 100,
                                "Mean RT": np.nanmean(self.data_matrix.data['RT'], axis=0),
                                "SD RT": np.nanstd(self.data_matrix.data['RT'], axis=0),
                                "Mean S/N": np.nanmean(self.data_matrix.data['SN_Ratio'], axis=0),
                                "SD S/N": np.nanstd(self.data_matrix.data['SN_Ratio'], axis=0)
                                }

        # setup per sample metrics
        injection_order = np.zeros(self.n_samples)
        for row in self.data_matrix.samples.values():
            i = self.sample_map[row['sample_name']]
            injection_order[i] = row['injection_order']
        self.per_sample_metrics = {
                                    "% Missing": self.data_matrix.missing.mean(axis=1) * 100,
                                    "% Outliers": self.combined_outliers.mean(axis=1) * 100,
                                    "Injection Order": injection_order
                                    }

        # setup widgets
        self.data_table = QTableWidget()

        self.data_type_label = QLabel()
        self.data_type_dropdown = QComboBox()

        self.outlier_label = QLabel()

        self.initUI()

    def initUI(self):
        
        # data table columns
        col_count = self.n_molecules + len(list(self.per_mol_metrics))
        self.data_table.setColumnCount(col_count)
        col_labels = list(self.sample_map.keys())
        col_labels.extend(list(self.per_sample_metrics))
        for i,sample_name in enumerate(col_labels):
            item = QTableWidgetItem(sample_name)
            try:
                group = self.group_map[sample_name]
                r,g,b,_ = [int(x*225) for x in self.group_colormap[group]]
                item.setBackground(QColor(r,g,b,100))
            except:
                pass
            self.data_table.setVerticalHeaderItem(i,item)
        # fill in per mol metrics
        for i,mol_metric in enumerate(self.per_mol_metrics):
            i += self.n_molecules
            for j,value in enumerate(self.per_mol_metrics[mol_metric]):
                self.data_table.setItem(i,j,QTableWidgetItem(f"{value:.2f}"))
        

        # data table rows
        row_count = self.n_samples + len(list(self.per_sample_metrics))
        self.data_table.setRowCount(row_count)
        header_labels = list(self.mol_map.keys())
        header_labels.extend(self.per_mol_metrics)
        for i,mol_name in enumerate(header_labels):
            item = QTableWidgetItem(mol_name)
            if mol_name in self.stds:
                r,g,b,_ = [int(x*225) for x in self.std_color]
                item.setBackground(QColor(r,g,b,100))
            self.data_table.setHorizontalHeaderItem(i,item)
        # fill in per sample metrics
        for i,sample_metric in enumerate(self.per_sample_metrics):
            i += self.n_samples
            for j,value in enumerate(self.per_sample_metrics[sample_metric]):
                self.data_table.setItem(i,j,QTableWidgetItem(f"{value:.2f}"))

        # siz policy
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.data_table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.data_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # click responses
        self.data_table.cellClicked.connect(self.on_cell_single_click)
        self.data_table.cellDoubleClicked.connect(self.on_cell_double_click)
        
        # data type selection
        self.data_type_label.setText("Metric:")
        self.data_type_dropdown.addItems(self.data_metrics)
        self.data_type_dropdown.currentTextChanged.connect(self.data_type_changed)
        self.data_type_dropdown.setCurrentIndex(self.data_metrics.index('Area'))

        # setup side label
        self.outlier_label.setText("None")
        self.legend_layout = QVBoxLayout()
        for group,color in self.group_colormap.items():
            r,g,b,_ = [int(x*225) for x in color]
            label = QLabel(f"   {group}")
            label.setStyleSheet(f"background-color: rgb{r},{g},{b}; color: white; padding: 2px;")
            self.legend_layout.addWidget(label)
        self.info_layout = QVBoxLayout()
        self.info_layout.addWidget(self.outlier_label)
        self.info_layout.addLayout(self.legend_layout)

        # setup whole layout
        self.top_layout = QHBoxLayout()
        self.top_layout.addWidget(self.data_table)
        self.top_layout.addLayout(self.info_layout)

        self.setLayout(self.top_layout)

    def data_type_changed(self):
        dtype = self.data_type_dropdown.currentText()
        for sample in self.data_matrix.samples:
            i = self.sample_map[sample]
            for mol in self.data_matrix.molecules:
                j = self.mol_map[mol]
                item = QTableWidgetItem(f"{self.data_matrix.data[dtype][i,j]:.2f}")
                if self.combined_outliers[i,j]:
                    item.setBackground(QColor(180,0,0,100))
                self.data_table.setItem(i,j,item)

    def on_cell_single_click(self, i, j):
        if i >= self.n_samples or j >= self.n_molecules:
            return
        
        out_types = []
        for metric in self.data_matrix.outliers:
            if self.data_matrix.outliers[metric][i,j]:
                out_types.append(metric)
            
        self.outlier_label.setText(f"Outlier Metrics:\n" + '\n'.join(out_types) 
                                   if out_types else "Outlier Metrics:\nNone")

    def on_cell_double_click(self, i, j):
        if i >= self.n_samples or j >= self.n_molecules:
            return
        
        sample = list(self.sample_map.keys())[i]
        molecule = list(self.mol_map.keys())[j]

        self.chrom_tab.sample_dropdown.setCurrentText(sample)
        self.chrom_tab.mol_dropdown.setCurrentText(molecule)
        
        dashboard = self.parent().parent()
        dashboard.tabs.setCurrentWidget(self.chrom_tab)
