"""

Loads a sample's IntensityMatrix object and allows display of TIC or given Ion chromatograms

collected peaks are marked and annotated with baseline, L/R bounds, and RT as well as shading
over the region given for the are of the peak

Peaks can be selected and their QC metrics displayed, along with their Mass Spectrum so a user
can manually look through and determine if the peak was collected correctly.

Functionalities:
    1.  Manual baseline redrawing, allows user to select two points to draw a linear baseline
        through, recalculates peak metrics based on this baseline and overrides saved values.
        Users can hit a "reset" button to fall back to origonal baseline
    2.  Change the peak identified as this "molecule" to another peak detected in the 
        IntensityMatrix, recalculating all metrics and overriding origonal peakn in the 
        SQL db and current DataMatrix object

"""

# region Imports

import sys, logging
import numpy as np

from PyQt5.QtWidgets import (QWidget, QTableWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QMessageBox, QComboBox, QApplication, 
                             QHeaderView, QSizePolicy, QTableWidgetItem, QFrame)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from src.plotting import plot_chromatogram, plot_peak, plot_spectrum
from src.run_data import RunData

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    filename="debug.log"
)

# endregion

# Main Peak review
class ChromatogramTab(QWidget):
    def __init__(self, run_data, parent=None):
        super().__init__(parent)

        self.setWindowState(Qt.WindowMaximized)

        self.run_data = run_data
        self.data_matrix = run_data.data_matrix

        self.sample_list = list(run_data.samples.keys())
        self.sample = self.sample_list[0]

        self.molecule_list = list(run_data.molecules.keys())
        self.molecule_list.append("None")
        self.molecule = "None"

        self.intensity_matrix = self.run_data.intensity_matrices[self.sample]
        self.ion_list = [str(x) for x in self.intensity_matrix.unique_mzs]
        self.ion_list[-1] = "TIC"
        self.ion = "TIC"
        
        self.peak_idx = 0
        self.peak_list = self.intensity_matrix.peak_dict[9999]
        self.peak = self.peak_list[self.peak_idx]

        self.ion_idx = self.intensity_matrix.unique_mzs.index(9999)
        self.time_array = np.array(list(self.intensity_matrix.time_map.values()))
        self.int_array = np.array(self.intensity_matrix.intensity_matrix[self.ion_idx])

        self.ion_label = QLabel("Ion:")
        self.ion_dropdown = QComboBox()

        self.mol_label = QLabel("Molecule:")
        self.mol_dropdown = QComboBox()
        self.next_peak = QPushButton("Next")
        self.prev_peak = QPushButton("Prev")

        self.sample_label = QLabel("Sample:")
        self.sample_dropdown = QComboBox()
        
        self.peak_view = PeakViewWidget(self.time_array, self.int_array, self.peak, parent=self)
        self.spectrum_view = SpectrumViewWidget(self.intensity_matrix, self.peak)
        self.trace_view = TraceViewWidget(self.time_array, self.int_array, self.ion, self.peak_list, parent=self)

        self.initUI()

    def initUI(self):
        
        # ion selection
        self.ion_dropdown.addItems(self.ion_list)
        self.ion_dropdown.blockSignals(True)
        self.ion_dropdown.setCurrentIndex(self.ion_list.index(str(self.ion)))
        self.ion_dropdown.blockSignals(False)
        self.ion_dropdown.currentTextChanged.connect(self.on_ion_changed)

        # molecule selection
        self.mol_dropdown.addItems(self.molecule_list)
        self.mol_dropdown.blockSignals(True)
        self.mol_dropdown.setCurrentIndex(self.molecule_list.index(self.molecule))
        self.mol_dropdown.blockSignals(False)
        self.mol_dropdown.currentTextChanged.connect(self.on_mol_changed)

        # sample selection
        self.sample_dropdown.addItems(self.sample_list)
        self.sample_dropdown.blockSignals(True)
        self.sample_dropdown.setCurrentIndex(self.sample_list.index(self.sample))
        self.sample_dropdown.blockSignals(False)
        self.sample_dropdown.currentTextChanged.connect(self.on_sample_changed)

        # peak navigation buttons
        self.next_peak.clicked.connect(self.next_clicked)
        self.prev_peak.clicked.connect(self.prev_clicked)

        # layout
        toolbar_layout = QHBoxLayout()
        toolbar_layout.addWidget(self.sample_label)
        toolbar_layout.addWidget(self.sample_dropdown)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.ion_label)
        toolbar_layout.addWidget(self.ion_dropdown)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.prev_peak)
        toolbar_layout.addWidget(self.next_peak)
        toolbar_layout.addWidget(self.mol_label)
        toolbar_layout.addWidget(self.mol_dropdown)

        chrom_spectra_layout = QVBoxLayout()
        chrom_spectra_layout.addWidget(self.trace_view)
        chrom_spectra_layout.addWidget(self.spectrum_view)

        view_layout = QHBoxLayout()
        view_layout.addLayout(chrom_spectra_layout)
        view_layout.addWidget(self.peak_view)

        layout = QVBoxLayout()
        layout.addLayout(toolbar_layout)
        layout.addLayout(view_layout)

        self.setLayout(layout)

    def on_ion_changed(self):

        # get new ion
        ion = self.ion_dropdown.currentText()
        self.ion = ion

        # reset molecule to None
        self.molecule = 'None'
        self.mol_dropdown.blockSignals(True)
        self.mol_dropdown.setCurrentIndex(self.molecule_list.index(self.molecule))
        self.mol_dropdown.blockSignals(False)

        # reset to first peak in this trace
        self.peak_idx = 0
        if ion != 'TIC':
            peak_list = self.intensity_matrix.peak_dict[np.int64(self.ion)]
        else:
            peak_list = self.intensity_matrix.peak_dict[9999]
        
        # get peak list
        if peak_list:
            self.peak = peak_list[self.peak_idx]
        else:
            self.peak = None

        # update views
        self.ion_dropdown.blockSignals(True)
        self.ion_dropdown.setCurrentIndex(self.ion_list.index(str(self.ion)))
        self.ion_dropdown.blockSignals(False)
        
        self.peak_view.update(self.peak)
        self.trace_view.update(self.ion)
        self.spectrum_view.update(self.intensity_matrix, self.peak)

    def on_mol_changed(self):
        # get molecule
        self.molecule = self.mol_dropdown.currentText()

        # get molecule's ion for trace if applicable
        if self.molecule != 'None':
            self.ion = self.data_matrix.molecules[self.molecule]['ion']

        # find peak
        self.peak_idx = self.get_peak_idx()
        self.peak = self.intensity_matrix.peak_dict[self.ion][self.peak_idx]

        # update
        self.ion_dropdown.blockSignals(True)
        self.ion_dropdown.setCurrentIndex(self.ion_list.index(str(self.ion)))
        self.ion_dropdown.blockSignals(False)
        self.trace_view.update(self.ion)
        self.peak_view.update(self.peak)
        self.spectrum_view.update(self.intensity_matrix, self.peak)

    def on_sample_changed(self):
        # get sample and update intesntiy matrix
        self.sample = self.sample_dropdown.currentText()
        self.intensity_matrix = self.run_data.intensity_matrices[self.sample]

        # update ion list
        self.ion_list = [str(x) for x in self.intensity_matrix.unique_mzs]
        self.ion_list[-1] = 'TIC'
        
        # reset to no molecule
        self.molecule = 'None'
        self.mol_dropdown.blockSignals(True)
        self.mol_dropdown.setCurrentIndex(self.molecule_list.index('None'))
        self.mol_dropdown.blockSignals(False)

        # display TIC
        self.ion = 'TIC'
        self.ion_dropdown.blockSignals(True)
        self.ion_dropdown.setCurrentIndex(self.ion_dropdown.count() - 1)
        self.ion_dropdown.blockSignals(False)

        # take first peak of TIC
        self.peak_idx = 0
        self.peak_list = self.intensity_matrix.peak_dict[9999]
        if self.peak_list:
            self.peak = self.peak_list[0]
        else:
            self.peak = None

        # update views
        self.trace_view.update(self.ion)
        self.peak_view.update(self.peak)
        self.spectrum_view.update(self.intensity_matrix, self.peak)

    def get_peak_idx(self):
        if self.molecule != 'None':
            row_i = self.data_matrix.sample_map[self.sample]
            col_i = self.data_matrix.mol_map[self.molecule]
            peak_idx = self.data_matrix.data['peak_idx'][row_i,col_i]
            if peak_idx == -1:
                QMessageBox.warning(self, "Warning", f"No peak found for {self.molecule} in {self.sample} ")
                return 0
            return peak_idx
        else:
            return 0

    def next_clicked(self):
        try:
            self.peak_idx += 1
            self.peak = self.peak_list[self.peak_idx]
            self.peak_view.update(self.peak)
        except Exception as e:
            QMessageBox(self, "Error", "No Next Peak")
    
    def prev_clicked(self):
        try:
            self.peak_idx -= 1
            self.peak = self.peak_list[self.peak_idx]
            self.peak_view.update(self.peak)
        except Exception as e:
            QMessageBox(self, "Error", "No Previous Peak")

class TraceViewWidget(QWidget):
    def __init__(self, time_array, int_array, ion, peak_list, parent=None):
        super().__init__(parent)

        # data
        self.time_array = time_array
        self.int_array = int_array
        self.peak_list = peak_list
        self.title = f"Ion: {ion}"
        self.ion = ion

        # figure
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas,self)

        # buttons
        self.redraw_btn = QPushButton("Reset")

        self.initUI()

    def initUI(self):
        
        frame = QFrame()
        frame_layout = QVBoxLayout()
        frame_layout.addStretch()
        frame_layout.addWidget(self.toolbar)
        frame_layout.addWidget(self.canvas)
        frame_layout.addStretch()
        frame.setLayout(frame_layout)

        layout = QVBoxLayout()
        layout.addWidget(frame)

        self.setLayout(layout)

        self.plot()

    def plot(self):
        self.figure.clear()
        plot_chromatogram(self.time_array, self.int_array, self.peak_list, self.title,
                          ax=self.figure.add_subplot(111))
        self.canvas.draw()

    def update(self, ion):
        im = self.parent().intensity_matrix
        self.ion = ion
        if ion != 'TIC':
            ion_idx = im.unique_mzs.index(int(ion))
            self.peak_list = im.peak_dict[np.int64(ion)]
        else:
            ion_idx = im.unique_mzs.index(9999)
            self.peak_list = im.peak_dict[np.int64(9999)]
        
        self.int_array = im.intensity_matrix[ion_idx]
        self.time_array = np.array(list(im.time_map.values()))

        self.title = f"Ion: {self.ion}"

        self.plot()

class PeakViewWidget(QWidget):
    def __init__(self, time_array, int_array, peak, parent=None):
        super().__init__(parent)

        self.peak = peak
        self.time_array = time_array
        self.int_array = int_array
        self.peak_idx = peak['peak_idx']
        self.title = f"Peak {self.peak_idx}"

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.table = QTableWidget()

        self.next_btn = QPushButton('Next')
        self.prev_btn = QPushButton('Prev')

        self.initUI()

    def initUI(self):
        
        table_rows = ['RT', 'Ion', 'Molecule', 'Area', 'Height', 'FWHH', 'Taling', 'S/N Ratio', 'Sharpness', 'Theoretical Plates']
        self.table.setRowCount(len(table_rows))
        self.table.setColumnCount(1)
        self.table.setVerticalHeaderLabels(table_rows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        frame = QFrame()
        frame_layout = QVBoxLayout()
        frame_layout.addWidget(self.canvas)
        frame_layout.addWidget(self.table)
        frame.setLayout(frame_layout)

        layout = QVBoxLayout()
        layout.addWidget(frame)

        self.update(self.peak)

        self.setLayout(layout)

    def plot(self):
        self.figure.clear()
        if self.peak is not None:
            plot_peak(self.time_array, self.int_array, self.peak, self.title,
                    ax=self.figure.add_subplot(111))
        self.canvas.draw()

    def update(self, peak):

        self.peak = peak
        self.title = f"Peak {peak['peak_idx']}"
        im = self.parent().intensity_matrix
        ion_idx = im.unique_mzs.index(peak['ion'])
        self.int_array = im.intensity_matrix[ion_idx]
        self.plot()

        if self.peak is not None:
            rt = peak.get('rt', 0) or 0
            ion = peak.get('ion', 'None')
            if ion == 9999:
                ion = 'TIC'
            molecule = peak.get('molecule', 'None')
            area = peak.get('area', 0) or 0
            height = peak.get('height', 0) or 0
            fwhh = peak.get('fwhh', 0) or 0
            tf = peak.get('tailing_factor', 0) or 0
            sn_ratio = peak.get('sn_ratio', 0) or 0
            conv = peak.get('conv', 0) or 0
            if fwhh == 0:
                tp = 0
            else:
                tp = 5.545 * (rt / fwhh)**2

            self.table.setItem(0,0,QTableWidgetItem(f"{rt:.2f}"))
            self.table.setItem(1,0,QTableWidgetItem(str(ion)))
            self.table.setItem(2,0,QTableWidgetItem(str(molecule)))
            self.table.setItem(3,0,QTableWidgetItem(f"{area:.2f}"))
            self.table.setItem(4,0,QTableWidgetItem(f"{height:.2f}"))
            self.table.setItem(5,0,QTableWidgetItem(f"{fwhh:.2f}"))
            self.table.setItem(6,0,QTableWidgetItem(f"{tf:.2f}"))
            self.table.setItem(7,0,QTableWidgetItem(f"{sn_ratio:.2f}"))
            self.table.setItem(8,0,QTableWidgetItem(f"{conv:.2f}"))
            self.table.setItem(9,0,QTableWidgetItem(f"{tp:.2f}"))

        else:
            for row_i in range(self.table.rowCount()):
                self.table.setItem(row_i,0,QTableWidgetItem("0"))

class SpectrumViewWidget(QWidget):
    def __init__(self, intensity_matrix, peak, parent = None):
        super().__init__(parent)

        mzs, abundances = intensity_matrix.generate_spectra(peak)
        self.peak = peak
        self.mzs = mzs
        self.abundances = abundances

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        self.initUI()

    def initUI(self):

        frame = QFrame()
        frame_layout = QVBoxLayout()
        frame_layout.addWidget(self.canvas)
        frame.setLayout(frame_layout)

        layout = QVBoxLayout()
        layout.addWidget(frame)

        self.setLayout(layout)
        self.plot()

    def plot(self):
        self.figure.clear()
        if self.peak:
            plot_spectrum(self.mzs, self.abundances, ax=self.figure.add_subplot(111))
        self.canvas.draw()

    def update(self, intensity_matrix, peak):
        mzs,abundances = intensity_matrix.generate_spectra(peak)
        self.peak = peak
        self.mzs = mzs
        self.abundances = abundances
        self.plot()

if __name__ == "__main__":

    app = QApplication(sys.argv)

    with open("style.css", "r") as f:
        app.setStyleSheet(f.read())

    run_data = RunData("run_name", "test")

    w = ChromatogramTab(run_data)
    w.show()

    sys.exit(app.exec_())