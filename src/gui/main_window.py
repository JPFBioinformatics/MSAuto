"""

Main starting point for the application, projectselect runselect then mainwindow

ProjectSelect
-------------
    two buttons: Load Project and New Project

    Load Project:
        pulls up a file explorer of self_get_app_dir() / databases / projects
        allows selecting of project from here, open button to confirm
        move on to main dashboard
    
    New Project:
        pulls up a window with a textbox and submit button
        name project, submit name
        checks if the name is available, if it is then creates a directory for it, initializes
        the project database
        move on to main dashboard

RunSelect
---------

    two buttons: Load Run and New Run

    Load Run:
        queries sql database for names of all runs
        select run name, Load button to confirm selection
        Loads all run data to run_data object
        takes you to main dashboard with run_data loaded

    New Run:
        oepns file explorer to select a directory of files to include in this run
        once files are selected opens another window with tables to fill out 
        Sample Table:
            names already filled in from files in selected dir
            fill in the rest of information needed in SQL samples table
        Molecules Table:
            fill in all information needed in SQL molecules table
        Confirm/continue/save? button saves this informatino to samples/molecules tables
        then program uses these tables to collect data from the files in the directory
        saves data, loads it to a run_data object
        takes you to main dashboard with run_data loaded

MainDashboard
-------------

    Entry point for run-based data inquiry
    Tabs lead to other root/gui/ *.py files
        File
            Load Run
            New Run
            Export
                Report
                QC
                Chromatogram
                Peak
                Spectrum
            Exit
                To Desktop
                To Main Menu
        Analysis
        QC
        Report
        Info
        Chromatogram
        Peak

"""

# region Imports

import sys

from PyQt5.QtWidgets import (QMainWindow, QWidget, QStackedWidget,
                             QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QLineEdit, QListWidget,
                             QDialog, QFileDialog, QMessageBox,
                             QApplication)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from qt_material import apply_stylesheet

from src.db import connect, get_run_samples, init_db
from src.run_data import RunData
from src.config_loader import ConfigLoader
from src.utils import get_app_dir, sanitize_name

# endregion

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.project_name = None
        self.run_name = []
        self.run_data = []
        self.active_run_idx = 0
        self.setWindowTitle("MSAuto")

        self.initUI()

    def initUI(self):
        return


class ProjectSelectWidget(QWidget):
    """
    Manages entry into app and selects project to work within
    """
    def __init__(self):
        super().__init__()

        self.setWindowTitle("MSAuto")

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
            self.parent().project_name = dialog.proj_name
            self.parent().stack.setCurrentIndex(1)

    def load_clicked(self):
        dialog = LoadProjectDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            self.parent().project_name = dialog.proj_name
            self.parent().stack.setCurrentIndex(1)

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
        self.title = QLabel("Enter Project Name:")
        self.proj_name = None

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
        print(projects_dir)
        projects_dir.mkdir(exist_ok=True, parents=True)

        # check that project name is entered
        proj_name = sanitize_name(self.name_input.text())
        if not proj_name:
            QMessageBox.warning(self, "Error", "Please Enter a project name")
            return

        projects = [p.name for p in projects_dir.iterdir() if p.is_dir()]

        if proj_name not in projects:
            # create proj dir
            project = projects_dir / proj_name
            project.mkdir(exist_ok=True, parents=True)

            # create sql database
            init_db(project / f"{proj_name}.db", appdir / "GCMSdata.sql")

            self.proj_name = proj_name

            self.accept()
        else:
            QMessageBox.warning(self, "Error", f"Project name {proj_name} already exists, choose unique name")
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
        self.proj_name = None

        self.title = QLabel("Select Project:")
        self.back_btn = QPushButton("Back")
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
        self.search_input.textChanged.connect(self.filter_list)

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

    def filter_list(self, text):
        self.populate_list(text)

    def populate_list(self, text):
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
        self.proj_name = selected.text()
        self.accept()

class RunSelectWidget(QWidget):
    """
    Manages which run to investigate within a project
    """



# testing block
if __name__ == "__main__":

    app = QApplication(sys.argv)
    apply_stylesheet(app, theme="dark_teal.xml", extra={"density_scale": "-2", "font_size": "12px"})

    w = ProjectSelectWidget()
    #w = NewProjectDialog()
    w.show()

    sys.exit(app.exec_())