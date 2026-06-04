"""

Main starting point for the application, projectselect runselect then mainwindow

ProjectSelect
-------
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
--------------

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

