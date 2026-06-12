"""

Main starting point for the application, projectselect runselect then mainwindow

First we select/create the project, create the data directory in appdir / databases / projects / project_name
project_name also saved to config.yaml here

Then we select/create the run, either loading data that is saved in the database or creating a new run
where you must specify the run_name, and fill in sample and molecule tables which are all saved to 
database right when it is created
In this process you also specify input_dir and input_type which are saved to config.yaml along with run_name

"""

# region Imports

import sys, shutil, logging, traceback
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import (QMainWindow, QWidget, QStackedWidget, QTabWidget, QTableWidget,
                             QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QLineEdit, QListWidget,
                             QDialog, QFileDialog, QMessageBox, QComboBox, QProgressDialog,
                             QApplication, QHeaderView, QSizePolicy, QTableWidgetItem)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

from src.db import (connect, init_db, run_exists, get_run_names, insert_sample, insert_run,
                    insert_molecule, get_run_molecules, insert_peak)
from src.config_loader import ConfigLoader
from src.utils import get_app_dir, sanitize_name, get_proj_db, get_run_dir, get_proj_dir, get_run_cfg_path
from src.mzml_processor import full_bulk_convert
from src.intensity_matrix import IntensityMatrix as IM
from src.run_data import RunData as RD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    filename="debug.log"
)

# endregion

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.project_name = None
        self.run_names = []
        self.run_data = []
        self.active_run_idx = 0
        self.cfg = None
        self.sample_data = None
        self.mol_data = None
        self.input_dir = None
        self.input_type = None

        self.stack = QStackedWidget()
        self.project_select = ProjectSelectWidget(self)
        self.run_select = RunSelectWidget(self)
        self.confirm_window = ConfirmConfigWidget(self)
        self.dashboard = MainDashboard(self)

        self.setWindowTitle("MSAuto")
        screen = QApplication.primaryScreen().geometry()
        self.move(screen.center() - self.rect().center())

        self.initUI()

    def initUI(self):

        self.stack.addWidget(self.project_select)
        self.stack.addWidget(self.run_select)
        self.stack.addWidget(self.confirm_window)
        self.stack.addWidget(self.dashboard)
        self.setCentralWidget(self.stack)

        return

# region                 ---------- Project Select ----------

class ProjectSelectWidget(QWidget):
    """
    Manages entry into app and selects project to work within
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("MSAuto > Project")

        self.new_proj_btn = QPushButton("New Project")
        self.load_proj_btn = QPushButton("Load Project")
        self.mng_proj_btn = QPushButton("Manage Projects")
        self.title = QLabel("GCMS Automation - Project Select")
        
        self.setFixedSize(800,500)

        screen = QApplication.primaryScreen().geometry()
        self.move(screen.center() - self.rect().center())

        self.initUI()

    def initUI(self):

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(25)

        self.title.setAlignment(Qt.AlignCenter)
        self.title.setFont(QFont("Roboto", 18, QFont.Bold))

        self.new_proj_btn.setFixedSize(400,100)
        self.new_proj_btn.setFont(QFont("Roboto", 12, QFont.Bold))
        self.new_proj_btn.clicked.connect(self.new_clicked)

        self.load_proj_btn.setFixedSize(400,100)
        self.load_proj_btn.setFont(QFont("Roboto", 12, QFont.Bold))
        self.load_proj_btn.clicked.connect(self.load_clicked)

        self.mng_proj_btn.setFixedSize(400,100)
        self.mng_proj_btn.setFont(QFont("Roboto", 12, QFont.Bold))
        self.mng_proj_btn.clicked.connect(self.mng_clicked)

        layout.addWidget(self.title)
        layout.addWidget(self.new_proj_btn, alignment=Qt.AlignHCenter)
        layout.addWidget(self.load_proj_btn, alignment=Qt.AlignHCenter)
        layout.addWidget(self.mng_proj_btn, alignment=Qt.AlignHCenter)
        self.setLayout(layout)

    def new_clicked(self):
        dialog = NewProjectDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            self.window().project_name = dialog.project_name
            self.window().stack.setCurrentIndex(1)

    def load_clicked(self):
        dialog = LoadProjectDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            self.window().project_name = dialog.project_name
            self.window().stack.setCurrentIndex(1)

    def mng_clicked(self):
        self.mng_proj_btn.setText("Not done yet")

class NewProjectDialog(QDialog):
    """
    Creates a new project
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Project")
        self.setFixedSize(500,300)

        self.name_input = QLineEdit()
        self.submit_btn = QPushButton("Submit")
        self.back_btn = QPushButton("Back")
        self.back_btn.setObjectName("smallBtn")
        self.title = QLabel("Enter Project Name:")
        self.project_name = None

        screen = QApplication.primaryScreen().geometry()
        self.move(screen.center() - self.rect().center())

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignHCenter)
        layout.setSpacing(10)

        header_layout = QHBoxLayout()
        header_layout.setAlignment(Qt.AlignTop)

        self.title.setFont(QFont("Roboto", 12, QFont.Bold))

        self.name_input.setPlaceholderText("Project Name...")
        self.name_input.setFixedSize(300, 50)
        self.name_input.setFont(QFont("Roboto", 12))

        self.submit_btn.setFixedSize(300,50)
        self.submit_btn.setFont(QFont("Roboto", 12))
        self.submit_btn.clicked.connect(self.submit_clicked)

        self.back_btn.setFixedSize(60,30)
        self.back_btn.clicked.connect(self.reject)
        
        header_layout.addWidget(self.back_btn)
        header_layout.addStretch()

        layout.addLayout(header_layout)
        layout.addWidget(self.title, alignment=Qt.AlignHCenter)
        layout.addWidget(self.name_input, alignment=Qt.AlignHCenter)
        layout.addWidget(self.submit_btn, alignment=Qt.AlignHCenter)
        layout.addSpacing(50)

        self.setLayout(layout)

    def submit_clicked(self):
        
        # get directoires
        appdir = get_app_dir()
        projects_dir = appdir / "databases" / "projects"
        projects_dir.mkdir(exist_ok=True, parents=True)

        # check that project name is entered
        project_name = sanitize_name(self.name_input.text())
        if not project_name:
            QMessageBox.warning(self, "Error", "Please Enter a project name")
            return

        projects = [p.name for p in projects_dir.iterdir() if p.is_dir()]

        if project_name not in projects:
            # create proj dir
            project = projects_dir / project_name
            project.mkdir(exist_ok=True, parents=True)

            # create sql database
            init_db(project / f"{project_name}.db", appdir / "GCMSdata.sql")

            self.project_name = project_name

            self.accept()
        else:
            QMessageBox.warning(self, "Error", f"Project name {project_name} already exists, choose unique name")
            return
        
        return

class LoadProjectDialog(QDialog):
    """
    Loads an existing project
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(500,400)
        self.setWindowTitle("Load Project")
        self.project_name = None

        self.title = QLabel("Select Project:")
        self.back_btn = QPushButton("Back")
        self.back_btn.setObjectName("smallBtn")
        self.search_input = QLineEdit()
        self.proj_list = QListWidget()
        self.submit_btn = QPushButton("Submit")

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignHCenter)
        layout.setSpacing(10)

        header_layout = QHBoxLayout()
        header_layout.setAlignment(Qt.AlignTop)

        self.title.setFont(QFont("Roboto", 12, QFont.Bold))

        self.search_input.setPlaceholderText("Project Name...")
        self.search_input.setFixedSize(300, 50)
        self.search_input.setFont(QFont("Roboto", 12))
        self.search_input.textChanged.connect(self.populate_list)

        self.proj_list.setFixedSize(300,100)

        self.back_btn.setFixedSize(60,30)
        self.back_btn.clicked.connect(self.reject)

        self.submit_btn.setFixedSize(300,50)
        self.submit_btn.setFont(QFont("Roboto", 12, QFont.Bold))
        self.submit_btn.clicked.connect(self.submit_clicked)

        header_layout.addWidget(self.back_btn)
        header_layout.addStretch()

        layout.addLayout(header_layout)
        layout.addWidget(self.title, alignment=Qt.AlignHCenter)
        layout.addWidget(self.search_input, alignment=Qt.AlignHCenter)
        layout.addWidget(self.proj_list, alignment=Qt.AlignHCenter)
        layout.addWidget(self.submit_btn, alignment=Qt.AlignHCenter)
        layout.addSpacing(50)

        self.setLayout(layout)
        self.populate_list("")

    def populate_list(self, text=""):

        appdir = get_app_dir()
        projects_dir = appdir / "databases" / "projects"
        projects_dir.mkdir(exist_ok=True,parents=True)

        projects = sorted([p.name for p in projects_dir.iterdir() if p.is_dir()])
        self.proj_list.clear()
        for p in projects:
            if text.lower() in p.lower():
                self.proj_list.addItem(p)
        
    def submit_clicked(self):
        selected = self.proj_list.currentItem()
        if not selected:
            QMessageBox.warning(self, "Error", "Please select a project")
            return
        self.project_name = selected.text()
        self.accept()

# endregion

# region                 ---------- Run Select ----------

class RunSelectWidget(QWidget):
    """
    Manages which run to investigate within a project
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        self.project_name = self.window().project_name

        self.setWindowTitle("MSAuto > Project > Run")
        self.setFixedSize(850,525)

        self.new_proj_btn = QPushButton("New Run")
        self.load_proj_btn = QPushButton("Load Run")
        self.mng_proj_btn = QPushButton("Manage Runs")
        self.back_btn = QPushButton("Back")
        self.back_btn.setObjectName("smallBtn")
        self.title = QLabel(f"Project: {self.project_name} - Run Select")

        screen = QApplication.primaryScreen().geometry()
        self.move(screen.center() - self.rect().center())

        self.initUI()

    def initUI(self):

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(25)

        self.title.setAlignment(Qt.AlignCenter)
        self.title.setFont(QFont("Roboto", 18, QFont.Bold))

        self.new_proj_btn.setFixedSize(400,100)
        self.new_proj_btn.setFont(QFont("Roboto", 12, QFont.Bold))
        self.new_proj_btn.clicked.connect(self.new_clicked)

        self.load_proj_btn.setFixedSize(400,100)
        self.load_proj_btn.setFont(QFont("Roboto", 12, QFont.Bold))
        self.load_proj_btn.clicked.connect(self.load_clicked)

        self.mng_proj_btn.setFixedSize(400,100)
        self.mng_proj_btn.setFont(QFont("Roboto", 12, QFont.Bold))
        self.mng_proj_btn.clicked.connect(self.mng_clicked)

        layout.addWidget(self.title)
        layout.addWidget(self.new_proj_btn, alignment=Qt.AlignHCenter)
        layout.addWidget(self.load_proj_btn, alignment=Qt.AlignHCenter)
        layout.addWidget(self.mng_proj_btn, alignment=Qt.AlignHCenter)
        self.setLayout(layout)

    def showEvent(self, event):
        proj_name = self.window().project_name
        if proj_name is not self.project_name:
            self.project_name = proj_name
        self.title.setText(f"Project: {proj_name} - Run Select")
        super().showEvent(event)

    def new_clicked(self):
        dialog = NewRunDialog(self, self.project_name)
        if dialog.exec_() == QDialog.Accepted:
            self.window().run_names.append(dialog.run_name)
            self.window().input_dir = dialog.input_dir
            self.window().input_type = dialog.input_type
            self.window().sample_data = dialog.sample_data
            self.window().mol_data = dialog.mol_data
            self.window().stack.setCurrentIndex(2)

    def load_clicked(self):
        dialog = LoadRunDialog(self, self.project_name)
        if dialog.exec_() == QDialog.Accepted:
            self.window().run_names.append(dialog.run_name)
            self.window().cfg = dialog.cfg
            self.window().run_data.append(dialog.run_data)
            self.window().stack.setCurrentIndex(3)

    def mng_clicked(self):
        self.mng_proj_btn.setText("Not done yet")

class NewRunDialog(QDialog):
    """
    handles creation of a new run
    """
    def __init__(self, parent=None, project_name = None):
        super().__init__(parent)
        self.setWindowTitle("New Run")
        self.setFixedSize(600,400)

        self.name_input = QLineEdit()
        self.submit_btn = QPushButton("Submit")
        self.back_btn = QPushButton("Back")
        self.back_btn.setObjectName("smallBtn")
        self.title = QLabel("Enter File Type, Run Dir, and Run Name")
        self.browse_btn = QPushButton("Browse")
        self.path_input =QLineEdit()
        self.file_type_combo = QComboBox()
        self.file_type_label = QLabel("File Type:")

        self.input_dir = None
        self.project_name = project_name
        self.run_name = None
        self.cfg = None
        self.sample_data = None
        self.mol_data = None
        self.input_type = None

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignHCenter)
        layout.setSpacing(10)

        header_layout = QHBoxLayout()
        header_layout.setAlignment(Qt.AlignTop)

        path_layout = QHBoxLayout()

        file_layout = QHBoxLayout()

        self.title.setFont(QFont("Roboto", 12, QFont.Bold))

        self.path_input.setPlaceholderText("Sample dir path...")
        self.path_input.setFixedSize(290,50)
        self.browse_btn.clicked.connect(self.browse_clicked)
        self.browse_btn.setFixedSize(100,50)

        self.name_input.setPlaceholderText("Run Name...")
        self.name_input.setFixedSize(400, 50)
        self.name_input.setFont(QFont("Roboto", 12))

        self.submit_btn.setFixedSize(400,50)
        self.submit_btn.setFont(QFont("Roboto", 12))
        self.submit_btn.clicked.connect(self.submit_clicked)

        self.back_btn.setFixedSize(60,30)
        self.back_btn.clicked.connect(self.reject)

        self.file_type_combo.addItems([".D",".mzML"])
        self.file_type_combo.setCurrentText(".D")
        self.file_type_combo.setFixedSize(200,50)
        self.file_type_label.setFont(QFont("Roboto", 12))
        
        file_layout.addStretch()
        file_layout.addWidget(self.file_type_label)
        file_layout.addWidget(self.file_type_combo)
        file_layout.addStretch()

        path_layout.addStretch()
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.browse_btn)
        path_layout.addStretch()

        header_layout.addWidget(self.back_btn, alignment=Qt.AlignTop)
        header_layout.addStretch()
        header_layout.addWidget(self.title)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        layout.addLayout(file_layout)
        layout.addLayout(path_layout)
        layout.addWidget(self.name_input, alignment=Qt.AlignHCenter)
        layout.addWidget(self.submit_btn, alignment=Qt.AlignHCenter)
        layout.addSpacing(50)

        self.setLayout(layout)

    def submit_clicked(self):
        
        # get directoires
        projects_dir = get_proj_dir(self.project_name)
        projects_dir.mkdir(exist_ok=True, parents=True)

        # save sample directory
        self.input_dir = Path(self.path_input.text().strip())
        if not self.input_dir.exists():
            QMessageBox.warning(self, "Error", "Directory does not exist, please reselect")
            return

        # get project name
        project_name = self.project_name
        if not project_name:
            QMessageBox.warning(self, "Error", "No project name, please return to project manager")
            return

        # get run name
        given_name = sanitize_name(self.name_input.text())
        if not given_name:
            QMessageBox.warning(self, "Error", "Please enter a run name")
            return
        run_name = f"{datetime.now().strftime('%Y_%m_%d')}_{given_name}"

        # make sure DB exists
        db_path = projects_dir / f"{project_name}.db"
        try:
            conn = connect(db_path) 
            # make sure run is uniquely named
            if run_exists(conn, run_name):
                QMessageBox.warning(self, "Error", "Run already exists, choose a unique name")
                return
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
            return
        finally:
            conn.close()

        # save run name
        self.run_name = run_name
        self.input_type = self.file_type_combo.currentText()
        
        # save sample and molecule data
        sample_dialog = SampleTableDialog(self, self.project_name, self.input_dir, self.input_type)
        if sample_dialog.exec_() != QDialog.Accepted:
            QMessageBox.warning(self, "Error", "No Sample Data Saved")
            return
        mol_dialog = MoleculeTableDialog(self, self.project_name)
        if mol_dialog.exec_() != QDialog.Accepted:
            QMessageBox.warning(self, "Error", "No Molecule Data Saved")
            return
        
        self.sample_data = sample_dialog.data
        self.mol_data = mol_dialog.data
        self.accept()

    def browse_clicked(self):
        selected = QFileDialog.getExistingDirectory(self, "Select Sample Directory")
        if selected:
            self.path_input.setText(selected)

class SampleTableDialog(QDialog):
    """
    Handles entry of sample metadata
    """
    def __init__(self, parent=None, project_name = None, sample_dir = None, input_type = None):
        super().__init__(parent)
        self.setWindowTitle("Sample Metadata")
        self.setWindowFlags(Qt.Window | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.resize(1000,500)

        self.project_name = project_name
        self.input_dir = sample_dir
        self.input_type = input_type

        self.title = QLabel("Sample Metadata")
        self.table = QTableWidget()
        self.back_btn = QPushButton("Back")
        self.back_btn.setObjectName("smallBtn")
        self.submit_btn = QPushButton("Submit")
        self.paste_btn = QPushButton("Paste")
        self.add_row_btn = QPushButton("Add Row")
        self.remove_row_btn = QPushButton("Remove Row")

        self.data = []                      # list of dicts with header:value for each entry/row
        
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        header_layout = QHBoxLayout()
        header_layout.setAlignment(Qt.AlignTop)

        footer_layout = QHBoxLayout()
        footer_layout.setAlignment(Qt.AlignBottom)

        columns = ['sample_name', 'modelID', 'group', 'sex', 'norm_factor', 'injection_order']
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.title.setFont(QFont("Roboto", 12, QFont.Bold))

        self.paste_btn.clicked.connect(self.paste_clicked)
        self.submit_btn.clicked.connect(self.submit_clicked)
        self.add_row_btn.clicked.connect(self.add_row_clicked)
        self.remove_row_btn.clicked.connect(self.remove_row_clicked)
        self.back_btn.setFixedSize(60,30)
        self.back_btn.clicked.connect(self.reject)

        header_layout.addWidget(self.back_btn)
        header_layout.addStretch()
        header_layout.addWidget(self.title, alignment=Qt.AlignHCenter)
        header_layout.addStretch()

        footer_layout.addWidget(self.paste_btn)
        footer_layout.addWidget(self.add_row_btn)
        footer_layout.addWidget(self.remove_row_btn)
        footer_layout.addStretch()
        footer_layout.addWidget(self.submit_btn)
        
        layout.addLayout(header_layout)
        layout.addWidget(self.table)
        layout.addLayout(footer_layout)

        self._populate_samples()

        self.setLayout(layout)

    def paste_clicked(self):

        clipboard = QApplication.clipboard().text()
        rows = clipboard.strip().split('\n')

        start_row = self.table.currentRow()
        if start_row < 0:
            start_row = 0
        rows_needed = start_row + len(rows)
        if rows_needed > self.table.rowCount():
            self.table.setRowCount(rows_needed)

        start_col = self.table.currentColumn()
        if start_col < 0:
            start_col = 0
        if len(rows[0].split('\t')) + start_col > self.table.columnCount():
            QMessageBox.warning(self,"Error", "Pasting too many columns, check copied values")
            return
        
        for i,row in enumerate(rows):
            cells = row.split('\t')
            for j,cell in enumerate(cells):
                if j < self.table.columnCount():
                    self.table.setItem(i+start_row,j+start_col,QTableWidgetItem(cell.strip()))

    def submit_clicked(self):
        for i in range(self.table.rowCount()):
            item = self.table.item(i,0)
            if not item or not item.text().strip():
                QMessageBox(self,"Error", f"Sample Name missing for row {i}")
                return
            row = {}
            for j in range(self.table.columnCount()):
                cell = self.table.item(i,j)
                val = cell.text().strip() if cell else None
                row[self.table.horizontalHeaderItem(j).text()] = val or None
            self.data.append(row)
        if not self.data:
            QMessageBox.warning(self, "Error", "No Data Entered")
            return
        self.accept()

    def remove_row_clicked(self):
        selected = self.table.currentRow()
        if selected >= 0:
            self.table.removeRow(selected)
        else:
            self.table.removeRow(self.table.rowCount() - 1)

    def add_row_clicked(self):
        self.table.insertRow(self.table.rowCount())

    def _populate_samples(self):
        if not self.input_dir:
            QMessageBox.warning(self, "Error", "No sample directory specified, return to Run Select")
            return
        if not self.input_type:
            QMessageBox.warning(self, "Erro", "No file type specified, return to New Run")
            return
        files = sorted(Path(self.input_dir).glob(f"*{self.input_type}"))
        self.table.setRowCount(len(files))
        for i,f in enumerate(files):
            self.table.setItem(i,0,QTableWidgetItem(f.stem))

class MoleculeTableDialog(QDialog):
    """
    Handles entry of molecule metadata
    """
    def __init__(self, parent=None, project_name=None):
        super().__init__(parent)

        self.project_name = project_name

        self.setWindowTitle("Molecule Metadata")
        self.setWindowFlags(Qt.Window | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.resize(1000,500)

        self.title = QLabel("Molecule Metadata")
        self.table = QTableWidget()
        self.back_btn = QPushButton("Back")
        self.back_btn.setObjectName("smallBtn")
        self.submit_btn = QPushButton("Submit")
        self.paste_btn = QPushButton("Paste")
        self.add_row_btn = QPushButton("Add Row")
        self.remove_row_btn = QPushButton("Remove Row")
        self.search_input = QLineEdit()
        self.run_list = QListWidget()
        self.load_btn = QPushButton("Load Table")

        self.data = []                      # list of dicts with header:value for each entry/row
        
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        header_layout = QHBoxLayout()
        header_layout.setAlignment(Qt.AlignTop)

        footer_layout = QHBoxLayout()
        footer_layout.setAlignment(Qt.AlignBottom)

        columns = ['molecule_name', 'ion', 'rt', 'std', 'casNo']
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table.setRowCount(20)
        

        self.title.setFont(QFont("Roboto", 12, QFont.Bold))

        self.paste_btn.clicked.connect(self.paste_clicked)
        self.submit_btn.clicked.connect(self.submit_clicked)
        self.add_row_btn.clicked.connect(self.add_row_clicked)
        self.remove_row_btn.clicked.connect(self.remove_row_clicked)
        self.back_btn.setFixedSize(60,30)
        self.back_btn.clicked.connect(self.reject)
        self.load_btn.clicked.connect(self.load_clicked)

        header_layout.addWidget(self.back_btn)
        header_layout.addStretch()
        header_layout.addWidget(self.title, alignment=Qt.AlignHCenter)
        header_layout.addStretch()
        header_layout.addWidget(self.search_input)
        header_layout.addWidget(self.load_btn)

        footer_layout.addWidget(self.paste_btn)
        footer_layout.addWidget(self.add_row_btn)
        footer_layout.addWidget(self.remove_row_btn)
        footer_layout.addStretch()
        footer_layout.addWidget(self.submit_btn)
        
        layout.addLayout(header_layout)
        layout.addWidget(self.table)
        layout.addLayout(footer_layout)

        self.setLayout(layout)

    def paste_clicked(self):

        clipboard = QApplication.clipboard().text()
        rows = clipboard.strip().split('\n')

        start_row = self.table.currentRow()
        if start_row < 0:
            start_row = 0
        rows_needed = start_row + len(rows)
        if rows_needed > self.table.rowCount():
            self.table.setRowCount(rows_needed)

        start_col = self.table.currentColumn()
        if start_col < 0:
            start_col = 0
        if len(rows[0].split('\t')) + start_col > self.table.columnCount():
            QMessageBox.warning(self,"Error", "Pasting too many columns, check copied values")
            return
        
        for i,row in enumerate(rows):
            cells = row.split('\t')
            for j,cell in enumerate(cells):
                if j < self.table.columnCount():
                    self.table.setItem(i+start_row,j+start_col,QTableWidgetItem(cell.strip()))

    def submit_clicked(self):
        for i in range(self.table.rowCount()):
            item = self.table.item(i,0)
            if not item or not item.text().strip():
                QMessageBox(self,"Error", f"Sample Name missing for row {i}")
                return
            row = {}
            for j in range(self.table.columnCount()):
                cell = self.table.item(i,j)
                val = cell.text().strip() if cell else None
                row[self.table.horizontalHeaderItem(j).text()] = val or None
            self.data.append(row)
        if not self.data:
            QMessageBox.warning(self, "Error", "No Data Entered")
            return
        self.accept()

    def remove_row_clicked(self):
        selected = self.table.currentRow()
        if selected >= 0:
            self.table.removeRow(selected)
        else:
            self.table.removeRow(self.table.rowCount() - 1)

    def add_row_clicked(self):
        self.table.insertRow(self.table.rowCount())
    
    def load_clicked(self):
        selected = self.run_list.currentItem()
        if not selected:
            QMessageBox.warning(self, "Error", "Please select a run")
            return

        run_name = selected.text()
        project_name = self.project_name
        project_dir = get_proj_dir(project_name)
        db_path = project_dir / f'{project_name}.db'
        conn = connect(db_path)
        rows = get_run_molecules(conn, run_name)
        self.table.setRowCount(len(rows))
        columns = [self.table.horizontalHeaderItem(j).text().lower() for j in range(self.table.columnCount())]
        for i,row in enumerate(rows):
            for key in row.keys():
                if key.lower() in columns:
                    j = columns.index(key.lower())
                    val = row[key]
                    self.table.setItem(i,j,QTableWidgetItem(str(val) if val is not None else '')) 

class LoadRunDialog(QDialog):
    """
    handles loading a run
    """
    def __init__(self, parent=None, project_name=None):
        super().__init__(parent)
        self.setFixedSize(500,400)
        self.setWindowTitle("Load Run")
        self.project_name = project_name
        self.run_name = None
        self.cfg = None
        self.run_data = None

        self.title = QLabel("Select Run:")
        self.back_btn = QPushButton("Back")
        self.back_btn.setObjectName("smallBtn")
        self.search_input = QLineEdit()
        self.run_list = QListWidget()
        self.submit_btn = QPushButton("Submit")

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignHCenter)
        layout.setSpacing(10)

        header_layout = QHBoxLayout()
        header_layout.setAlignment(Qt.AlignTop)

        self.title.setFont(QFont("Roboto", 12, QFont.Bold))

        self.search_input.setPlaceholderText("Run Name...")
        self.search_input.setFixedSize(300, 50)
        self.search_input.setFont(QFont("Roboto", 12))
        self.search_input.textChanged.connect(self.populate_list)

        self.run_list.setFixedSize(300,100)

        self.back_btn.setFixedSize(60,30)
        self.back_btn.clicked.connect(self.reject)

        self.submit_btn.setFixedSize(300,50)
        self.submit_btn.setFont(QFont("Roboto", 12, QFont.Bold))
        self.submit_btn.clicked.connect(self.submit_clicked)

        header_layout.addWidget(self.back_btn)
        header_layout.addStretch()

        layout.addLayout(header_layout)
        layout.addWidget(self.title, alignment=Qt.AlignHCenter)
        layout.addWidget(self.search_input, alignment=Qt.AlignHCenter)
        layout.addWidget(self.run_list, alignment=Qt.AlignHCenter)
        layout.addWidget(self.submit_btn, alignment=Qt.AlignHCenter)
        layout.addSpacing(50)

        self.setLayout(layout)
        self.populate_list("")

    def populate_list(self, text=""):
        project_name = self.project_name
        projects_dir = get_proj_dir(project_name)
        projects_dir.mkdir(exist_ok=True,parents=True)

        db_path = projects_dir / f"{project_name}.db"
        try:
            conn = connect(db_path)
        except FileNotFoundError as e:
            QMessageBox.warning(self, "Error", "No database found, please return to project manager")
            return
        
        run_names = sorted(get_run_names(conn))
        self.run_list.clear()
        for r in run_names:
            if text.lower() in r.lower():
                self.run_list.addItem(r)

        conn.close()
 
    def submit_clicked(self):
        selected = self.run_list.currentItem()
        if not selected:
            QMessageBox.warning(self, "Error", "Please select a run")
            return
        run_name = selected.text()
        cfg = ConfigLoader(get_run_cfg_path(self.project_name, run_name))
        self.run_name = run_name
        self.cfg = cfg

        self.progress = QProgressDialog("Processing...",None,0,0,self)
        self.progress.setWindowTitle("Please Wait")
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.show()

        self.worker = ProcessingLoader(self.project_name, self.run_name)
        self.worker.finished.connect(self.on_processing_done)
        self.worker.error.connect(self.on_processing_error)
        self.worker.run_data.connect(self.on_run_data)
        self.worker.start()

    def on_run_data(self, rd):
        self.run_data = rd

    def on_processing_done(self):
        self.progress.close()
        self.accept()

    def on_processing_error(self, msg):
        self.progress.close()
        QMessageBox.critical(self,"Error",f"Processing failed:\n{msg}")

class ProcessingLoader(QThread):
    """
    popup box that signals program is loading a run
    """
    finished = pyqtSignal()
    run_data = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, project_name, run_name):
        super().__init__()
        self.project_name = project_name
        self.run_name = run_name

    def run(self):
        try:
            rd = RD(self.run_name, self.project_name)
            self.run_data.emit(rd)
            self.finished.emit()
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))

# endregion

# region                 ---------- Confirm Data ----------

class ConfirmConfigWidget(QWidget):
    """
    displays all input data for confirmation before processing
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        self.sample_data = None
        self.mol_data = None
        self.project_name = None
        self.run_name = None
        self.input_dir = None
        self.input_type = None
        self.db_path = None

        self.setWindowTitle("Confrim Data")
        self.resize(800,500)

        self.tabs = QTabWidget()

        self.config_tab = QWidget()
        self.samples_tab = QWidget()
        self.molecules_tab = QWidget()

        self.config_table = QTableWidget()
        self.samples_table = QTableWidget()
        self.molecules_table = QTableWidget()

        self.back_btn = QPushButton("Back")
        self.back_btn.setObjectName("smallBtn")
        self.confirm_btn = QPushButton("Confirm")

        screen = QApplication.primaryScreen().geometry()
        self.move(screen.center() - self.rect().center())

        self.initUI()

    def initUI(self):
        
        # config tab
        config_layout = QVBoxLayout()
        self.config_table.setColumnCount(2)
        self.config_table.setHorizontalHeaderLabels(['Key','Value'])
        self.config_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        config_layout.addWidget(self.config_table)
        self.config_tab.setLayout(config_layout)

        # samples tab
        samples_layout = QVBoxLayout()
        sample_columns = ['sample_name', 'modelID', 'group', 'sex', 'norm_factor', 'injection_order']
        self.samples_table.setColumnCount(len(sample_columns))
        self.samples_table.setHorizontalHeaderLabels(sample_columns)
        self.samples_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.samples_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        samples_layout.addWidget(self.samples_table)
        self.samples_tab.setLayout(samples_layout)

        # molecules tab
        molecules_layout = QVBoxLayout()
        mol_columns = ['molecule_name', 'ion', 'rt', 'std', 'casNo']
        self.molecules_table.setColumnCount(len(mol_columns))
        self.molecules_table.setHorizontalHeaderLabels(mol_columns)
        self.molecules_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.molecules_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        molecules_layout.addWidget(self.molecules_table)
        self.molecules_tab.setLayout(molecules_layout)

        # add tabs
        self.tabs.addTab(self.config_tab, "Config")
        self.tabs.addTab(self.samples_tab, "Samples")
        self.tabs.addTab(self.molecules_tab, "Molecules")

        # header for all tabs
        header = QHBoxLayout()
        self.back_btn.setFixedSize(60,30)
        self.back_btn.clicked.connect(self.back_clicked)
        header.addWidget(self.back_btn)
        header.addStretch()
        
        # footer for all tabs
        footer = QHBoxLayout()
        self.confirm_btn.setFixedSize(120,40)
        self.confirm_btn.clicked.connect(self.confirm_clicked)
        footer.addStretch()
        footer.addWidget(self.confirm_btn)

        layout = QVBoxLayout()
        layout.addLayout(header)
        layout.addWidget(self.tabs)
        layout.addLayout(footer)
        self.setLayout(layout)

    def showEvent(self,event):
        self.sample_data = self.window().sample_data
        if not self.sample_data:
            super().showEvent(event)
            return
        self.mol_data = self.window().mol_data
        self.project_name = self.window().project_name
        self.run_name = self.window().run_names[-1]
        self.input_dir = self.window().input_dir
        self.input_type = self.window().input_type
        self.db_path = get_proj_db(self.project_name)
        cfg = ConfigLoader.load_default_config(
            get_run_dir(self.project_name, self.run_name) / 'config.yaml')
        cfg.set("run_name", value=self.run_name)
        cfg.set("input_dir", value=str(self.input_dir))
        cfg.set("project_name", value=self.project_name)
        cfg.set("input_type", value=self.input_type)
        self.cfg = cfg
        self._populate_config()
        self._populate_samples()
        self._populate_molecules()
        super().showEvent(event)

    def _populate_config(self):
        config = self.cfg.config
        self.config_table.setRowCount(len(config))
        for i,(key,value) in enumerate(config.items()):
            self.config_table.setItem(i,0,QTableWidgetItem(str(key)))
            self.config_table.setItem(i,1,QTableWidgetItem(str(value)))

    def _populate_samples(self):
        sample_data = self.window().sample_data
        self.samples_table.setRowCount(len(sample_data))
        for i,row in enumerate(sample_data):
            self.samples_table.setItem(i,0,QTableWidgetItem(str(row['sample_name'])))
            self.samples_table.setItem(i,1,QTableWidgetItem(str(row['modelID'])))
            self.samples_table.setItem(i,2,QTableWidgetItem(str(row['group'])))
            self.samples_table.setItem(i,3,QTableWidgetItem(str(row['sex'])))
            self.samples_table.setItem(i,4,QTableWidgetItem(str(row['norm_factor'])))
            self.samples_table.setItem(i,5,QTableWidgetItem(str(row['injection_order'])))

    def _populate_molecules(self):
        mol_data = self.window().mol_data
        self.molecules_table.setRowCount(len(mol_data))
        for i,row in enumerate(mol_data):
            self.molecules_table.setItem(i,0,QTableWidgetItem(str(row['molecule_name'])))
            self.molecules_table.setItem(i,1,QTableWidgetItem(str(row['ion'])))
            self.molecules_table.setItem(i,2,QTableWidgetItem(str(row['rt'])))
            self.molecules_table.setItem(i,3,QTableWidgetItem(str(row['std'])))
            self.molecules_table.setItem(i,4,QTableWidgetItem(str(row['casNo'])))

    def back_clicked(self):
        run_dir = get_run_dir(self.window().project_name, self.window().run_names[-1])
        if run_dir.exists():
            shutil.rmtree(run_dir)
        self.window().run_names.pop()
        self.window().cfg = None
        self.window().stack.setCurrentIndex(1)

    def confirm_clicked(self):

        self.progress = QProgressDialog("Processing...",None,0,0,self)
        self.progress.setWindowTitle("Please Wait")
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.show()

        self.worker = ProcessingWorker(self.sample_data, self.mol_data, self.project_name, self.run_name, self.cfg)
        self.worker.finished.connect(self.on_processing_done)
        self.worker.error.connect(self.on_processing_error)
        self.worker.run_data.connect(self.on_run_data)
        self.worker.start()

    def on_run_data(self, rd):
        self.window().run_data.append(rd)

    def on_processing_done(self):
        self.progress.close()
        self.window().sample_data = None
        self.window().mol_data = None
        self.window().stack.setCurrentIndex(3)

    def on_processing_error(self, msg):
        self.progress.close()
        QMessageBox.critical(self,"Error",f"Processing failed:\n{msg}")

class ProcessingWorker(QThread):
    """
    popup box that signals to a user that program is processing and did not crash
    """
    finished = pyqtSignal()
    error = pyqtSignal(str)
    run_data = pyqtSignal(object)

    def __init__(self, sample_data, mol_data, proj_name, run_name, cfg):
        super().__init__()
        self.sample_data = sample_data
        self.mol_data = mol_data
        self.project_name = proj_name
        self.run_name = run_name
        self.input_dir = cfg.get('input_dir')
        self.input_type = cfg.get('input_type')
        self.cfg = cfg

    def run(self):
        conn = None
        try:
            # make run dir
            run_dir = get_run_dir(self.project_name, self.run_name)
            run_dir.mkdir(parents=True,exist_ok=True)

            # save config
            self.cfg.save()

            conn = connect(get_proj_db(self.project_name))
            mols = []
            mzs = []
            rts = []
            with conn:
                # insert run to table
                insert_run(conn,self.run_name)

                # insert sample/molecule data to DB
                for row in self.sample_data:
                    insert_sample(conn,
                                    row['sample_name'],
                                    self.run_name,
                                    row['modelID'],
                                    row['group'],
                                    row['sex'],
                                    row['norm_factor'],
                                    row['injection_order'])

                for row in self.mol_data:
                    mols.append(row['molecule_name'])
                    mzs.append(row['ion'])
                    rts.append(row['rt'])
                    insert_molecule(conn,
                                    row['molecule_name'],
                                    self.run_name,
                                    row['ion'],
                                    row['rt'],
                                    row['std'],
                                    row['casNo'])

                # generate intensity mtrices, save, collect data and save
                ims = full_bulk_convert(self.input_dir,self.input_type,self.cfg)
                id_map = {}
                for im in ims:
                    #save intensityMatrix objects
                    imID = im.save_sql_im(conn,self.run_name)
                    id_map[im.sample_name] = imID

                peak_data = IM.collect_data(ims, mols, mzs, rts)
                for im_name, peak_list in peak_data.items():
                    for peak in peak_list:
                        insert_peak(conn,
                                    id_map[im_name],
                                    self.run_name,
                                    peak['molecule'],
                                    peak['center'],
                                    peak['left_bound'],
                                    peak['right_bound'],
                                    peak['rt'],
                                    peak['height'],
                                    peak['area'],
                                    peak['sn_ratio'],
                                    peak['ion'],
                                    peak['fwhh'],
                                    peak['tailing_factor'],
                                    peak['bl_slope'],
                                    peak['bl_yint'],
                                    peak['conv'],
                                    peak['valley_ratio']
                                    )

            for im in ims:
                im.save_h5_object(self.project_name,self.run_name)

            rd = RD(self.run_name, self.project_name)
            self.run_data.emit(rd)
            self.finished.emit()

        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))

        finally:
            if conn:
                conn.close()

# endregion

# region                 ---------- Main Dashboard ----------

class MainDashboard(QWidget):
    """
    Entry point for analysis/visualization
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tabs = QTabWidget()
        self.title = QLabel("Main Dashboard")
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        layout.addWidget(self.title)
        layout.addWidget(self.tabs)
        self.setLayout(layout)

# endregion

# testing block
if __name__ == "__main__":

    app = QApplication(sys.argv)

    with open("style.css", "r") as f:
        app.setStyleSheet(f.read())

    w = MainWindow()
    w.show()

    sys.exit(app.exec_())