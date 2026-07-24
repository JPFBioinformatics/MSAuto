from pathlib import Path
from src.mzml_processor import create_scan_matrix
from src.config_loader import ConfigLoader
from src.plotting import plot_heatmap
import matplotlib.pyplot as plt
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    filename=Path(__file__).parent / "cwt_test.log"
)
logger = logging.getLogger(__name__)

file_path = Path(r"C:\Jack\Projects\IlyaAura Mouse Labelling\7_10_int1\mzML_files\Int 1.mzML")

cfg_path = Path(__file__).parent / 'config.yaml'
cfg = ConfigLoader(cfg_path)

im = create_scan_matrix(file_path, cfg=cfg)

ion = 218

row_i = im.ion_map[ion]
times = im.time_map.values()

coefficients = im.find_maxima_cwt(im.intensity_matrix[row_i,:],ion)

fig,ax = plt.subplots(1,1)
plot_heatmap(ax,coefficients,row_labels=times)
plt.show()