"""
GCMS Automation desktop app entry point
"""
import sys
import logging
import multiprocessing


def main():
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QDir

    from src.main_pipeline.utils import get_log_dir, get_stylesheet_path, get_theme_dir
    from src.main_pipeline.db import ensure_db
    from src.gui.main_window import MainWindow

    # one log file for the whole app, in the writable data folder
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        filename=get_log_dir() / "debug.log",
    )

    app = QApplication(sys.argv)

    # stylesheet + the 'icon:' prefix it uses (fixes the qt.svg warnings)
    QDir.addSearchPath('icon', str(get_theme_dir()))
    with open(get_stylesheet_path(), "r") as f:
        app.setStyleSheet(f.read())

    # create/upgrade the database before any dialog touches it
    ensure_db()

    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    multiprocessing.freeze_support()      # must be first: lets ProcessPool workers start in a frozen app
    main()
