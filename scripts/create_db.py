import sqlite3,sys
from pathlib import Path

# location of pipeline root dir
root_dir = Path(__file__).resolve().parent.parent
# tell python to look here for modules
sys.path.insert(0, str(root_dir))

from src.utils import get_app_dir

def init_database(db_path: Path, schema_path: Path):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    with open(schema_path) as f:
        conn.executescript(f.read())
    conn.close()

def main():
    appdir = get_app_dir()
    db_dir = appdir / 'database'
    db_path = db_dir / "GCMSdata.db"
    db_dir.mkdir(exist_ok=True,parents=True)
    schema_path = 'GCMSdata.sql'

    init_database(db_path, schema_path)

if __name__ == "__main__":
    main()