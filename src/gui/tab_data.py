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

import logging
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (QWidget, QTableWidget, QVBoxLayout, QHBoxLayout, QSplitter,
                             QLabel, QComboBox, QHeaderView, QSizePolicy, QTableWidgetItem,
                             QGroupBox, QStyledItemDelegate, QFileDialog, QPushButton, QMenu,
                             QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QBrush

import matplotlib.cm as cm

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
        self.inv_sample_map = {v:k for k,v in self.sample_map.items()}
        self.n_molecules = self.data_matrix.n_molecules
        self.mol_map = self.data_matrix.mol_map
        self.inv_mol_map = {v:k for k,v in self.mol_map.items()}
        self.data_metrics = list(self.data_matrix.data.keys())
        self.group_map = self.data_matrix.group_map
        self.group_indices = self.data_matrix.group_indices
        groups = list(set(self.group_indices.keys()))
        self.cmap = cm.get_cmap('rainbow', len(groups))
        self.group_colormap = {group:self.cmap(i) for i,group in enumerate(groups)}
        self.std_color = self.cmap(len(groups)) if groups else self.cmap(0)

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
                                "Mean Area": np.nanmean(self.data_matrix.data['norm_Area'], axis=0),
                                "SD Area": np.nanstd(self.data_matrix.data['norm_Area'], axis=0),
                                "% CV Area": (np.nanstd(self.data_matrix.data['norm_Area'], axis=0) / np.nanmean(self.data_matrix.data['norm_Area'], axis=0)) * 100,
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

        self.export_btn = QPushButton("Export to Excel")

        print(f"Stds:\n{self.stds}")
        print(f"Groups:\n{self.group_indices}")

        self.initUI()

    def initUI(self):
        
        # data table columns
        self.data_table.setHorizontalHeader(ColoredHeader(Qt.Horizontal, self.data_table))
        col_count = self.n_molecules + len(list(self.per_sample_metrics)) + 1
        self.data_table.setColumnCount(col_count)
        col_labels = list(self.mol_map.keys())
        col_labels.append('')
        col_labels.extend(list(self.per_sample_metrics))
        for i,name in enumerate(col_labels):
            item = QTableWidgetItem(name)
            if name in self.stds:
                r,g,b,_ = [int(x*225) for x in self.std_color]
                item.setBackground(QColor(r,g,b,100))

            self.data_table.setHorizontalHeaderItem(i,item)
        # data table rows
        self.data_table.setVerticalHeader(ColoredHeader(Qt.Vertical, self.data_table))
        row_count = self.n_samples + len(list(self.per_mol_metrics)) + 1
        self.data_table.setRowCount(row_count)
        header_labels = list(self.sample_map.keys())
        header_labels.append('')
        header_labels.extend(self.per_mol_metrics)
        for i,name in enumerate(header_labels):
            item = QTableWidgetItem(name)
            try:
                group = self.group_map[name]
                r,g,b,_ = [int(x*225) for x in self.group_colormap[group]]
                item.setBackground(QColor(r,g,b,100))
            except:
                pass
            self.data_table.setVerticalHeaderItem(i,item)

        # fill in per mol metrics
        for i,mol_metric in enumerate(self.per_mol_metrics):
            i += self.n_samples + 1
            for j,value in enumerate(self.per_mol_metrics[mol_metric]):
                self.data_table.setItem(i,j,QTableWidgetItem(self.format_val(value)))
        # fill in per sample metrics
        for j,sample_metric in enumerate(self.per_sample_metrics):
            j += self.n_molecules + 1
            for i,value in enumerate(self.per_sample_metrics[sample_metric]):
                self.data_table.setItem(i,j,QTableWidgetItem(self.format_val(value)))

        # table policies
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.data_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.data_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.data_table.setItemDelegate(BackgroundDelegate(self.data_table))
        self.data_table.setFocusPolicy(Qt.ClickFocus)
        self.data_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.data_table.customContextMenuRequested.connect(self.on_cell_right_click)
        self.data_table.cellClicked.connect(self.on_cell_single_click)
        self.export_btn.clicked.connect(self.export_to_excel)
        
        # data type selection
        self.data_type_label.setText("Metric:")
        self.data_type_dropdown.addItems(self.data_metrics)
        self.data_type_dropdown.currentTextChanged.connect(self.data_type_changed)
        self.data_type_dropdown.setCurrentIndex(self.data_metrics.index('Area'))

        # setup table
        table_box = QGroupBox("Data")
        table_layout = QVBoxLayout()
        table_layout.addWidget(self.data_table)
        table_box.setLayout(table_layout)

        # setup side label
        info_box = QGroupBox("Info")

        outlier_section = QGroupBox("Outlier Metrics")
        self.outlier_layout = QVBoxLayout()
        none_label = QLabel("None")
        none_label.setStyleSheet("color: rgba(255,255,255,0.4); padding: 4px;")
        self.outlier_layout.addWidget(none_label)
        self.outlier_layout.addStretch()
        outlier_section.setLayout(self.outlier_layout)

        colors_section = QGroupBox("Group Colors")
        legend_layout = QVBoxLayout()
        for group,color in self.group_colormap.items():
            r,g,b,_ = [int(x*225) for x in color]
            label = QLabel(f"   {group}   ")
            label.setStyleSheet(f"background-color: rgb({r},{g},{b}); color: white; padding: 4px; border-radius: 3px;")
            legend_layout.addWidget(label)
        legend_layout.addStretch()
        colors_section.setLayout(legend_layout)

        info_layout = QVBoxLayout()
        info_layout.addWidget(outlier_section)
        info_layout.addWidget(colors_section)
        info_layout.addStretch()
        info_box.setLayout(info_layout)

        # splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(table_box)
        splitter.addWidget(info_box)
        splitter.setStretchFactor(0,4)
        splitter.setStretchFactor(1,1)

        toolbar_layout = QHBoxLayout()
        toolbar_layout.addWidget(self.data_type_label)
        toolbar_layout.addWidget(self.data_type_dropdown)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.export_btn)

        layout = QVBoxLayout()
        layout.addLayout(toolbar_layout)
        layout.addWidget(splitter)
        self.setLayout(layout)

        self.data_type_changed()

    def data_type_changed(self):
        dtype = self.data_type_dropdown.currentText()
        self.data_table.setUpdatesEnabled(False)
        for sample in self.data_matrix.samples:
            i = self.sample_map[sample]
            for mol in self.data_matrix.molecules:
                j = self.mol_map[mol]
                item = QTableWidgetItem(self.format_val(self.data_matrix.data[dtype][i,j]))
                if self.combined_outliers[i,j]:
                    item.setBackground(QColor(180,0,0,75))
                self.data_table.setItem(i,j,item)
        self.data_table.setUpdatesEnabled(True)

    def on_cell_single_click(self, i, j):
        if i >= self.n_samples or j >= self.n_molecules:
            return
        
        # clear previous layout
        while self.outlier_layout.count():
            item = self.outlier_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        out_types = [metric for metric in self.data_matrix.outliers
                     if self.data_matrix.outliers[metric][i,j]]
    
        if out_types:
            for metric in out_types:
                label = QLabel(f"   {metric}   ")
                label.setStyleSheet("background-color: rgb(180,60,60); color: white; padding: 4px; border-radius: 3px;")
                self.outlier_layout.addWidget(label)
        else:
            label = QLabel("None")
            label.setStyleSheet("color: rgba(255,255,255,0.4); padding: 4px;")
            self.outlier_layout.addWidget(label)
        
        self.outlier_layout.addStretch()
            
    def view_chrom(self, i, j):
        
        sample = self.inv_sample_map[i]
        molecule = self.inv_mol_map[j]

        if self.data_matrix.missing[i,j]:
            QMessageBox(self, "Error", f"No peak found for {molecule} in {sample}")
            return

        self.chrom_tab.navigate_to(sample, molecule)
        
        self.parent().parent().setCurrentWidget(self.chrom_tab)

    def on_cell_right_click(self, pos):

        index = self.data_table.indexAt(pos)
        i,j = index.row(), index.column()

        if i >= self.n_samples or j >= self.n_molecules:
            return
        
        menu = QMenu(self)
        view_chromatogram = menu.addAction("Chromatogram View")

        if menu.exec_(self.data_table.viewport().mapToGlobal(pos)) == view_chromatogram:
            self.view_chrom(i,j)

    def format_val(self, val):
        if abs(val) > 1000000:
            return f"{val:.2e}"
        return f"{val:.2f}"

    def export_to_excel(self):

        path, _ = QFileDialog.getSaveFileName(self, "Export to Excel", "", "Excel Files (*xlsx)")

        if not path:
            return
        if not path.endswith('.xlsx'):
            path += '.xlsx'
        
        row_labels = [self.data_table.verticalHeaderItem(i).text()
                      for i in range(self.data_table.rowCount())]
        col_labels = [self.data_table.horizontalHeaderItem(i).text()
                      for i in range(self.data_table.columnCount())]
        
        data = []
        for i in range(self.data_table.rowCount()):
            row = []
            for j in range(self.data_table.columnCount()):
                item = self.data_table.item(i,j)
                try:
                    row.append(float(item.text()))
                except:
                    row.append(item.text() if item else '')
            data.append(row)

        df = pd.DataFrame(data, index=row_labels, columns=col_labels)
        df.to_excel(path, index=True)

class BackgroundDelegate(QStyledItemDelegate):
    """
    Allows for highlighting of table cells 
    """
    def paint(self, painter, option, index):
        bg = index.data(Qt.BackgroundRole)
        if bg and isinstance(bg,QBrush) and bg.color().isValid():
            painter.fillRect(option.rect, bg)
        super().paint(painter,option,index)

class ColoredHeader(QHeaderView):
    """
    Allows for highlighting of table headers
    """
    def paintSection(self, painter, rect, logical_index):
        super().paintSection(painter, rect, logical_index)
        table = self.parent()
        if self.orientation() == Qt.Horizontal:
            item = table.horizontalHeaderItem(logical_index)
        else:
            item = table.verticalHeaderItem(logical_index)
        if item:
            bg = item.background()
            if bg.color().isValid() and bg.color().alpha() > 0:
                painter.fillRect(rect, bg)

