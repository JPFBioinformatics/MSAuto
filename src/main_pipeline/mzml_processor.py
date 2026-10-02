"""

Handles converting agilant .D directories to mzml files and then converting those into IntensityMatrix
objects.

"""

# region Imports

import base64, zlib, os, psutil, random
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.main_pipeline.intensity_matrix import IntensityMatrix
from src.main_pipeline.utils import get_run_dir, configure_run_logging
from src.main_pipeline.im_store import write_sample

# logging
import logging
logger = logging.getLogger(__name__)

# endregion

def full_bulk_convert(mzml_dir: Path, cfg, store, serial=False, detect_peaks=False):
    """
    Builds an IntensityMatrix for every .mzML file in mzml_dir and saves each to `store` as it
    finishes. Samples already saved with the current config are skipped.
    (.D -> .mzML conversion is done by the user beforehand, e.g. with msconvert)

    Returns:
        names                           sample names available in the store, in file order
    """
    mzml_dir = Path(mzml_dir)

    # setup logger
    run_dir = get_run_dir(cfg.get("project_name"), cfg.get("run_name"))
    configure_run_logging(run_dir)

    # find mzML files
    if not mzml_dir.is_dir():
        raise FileNotFoundError(f"mzML directory not found: {mzml_dir}")
    files = sorted(mzml_dir.glob("*.mzML"), key=lambda f: f.stem)
    if not files:
        raise FileNotFoundError(f"No .mzML files found in {mzml_dir}")
    logger.info(f"Found {len(files)} mzML files in {mzml_dir}")

    # skip samples already saved with this config
    to_build = [f for f in files if not store.is_valid(f.stem)]
    logger.info(f"{len(files) - len(to_build)} samples loaded from saved state, {len(to_build)} to build")

    # determine max workers for this system (calibration builds + saves a few samples)
    max_workers, built, success_count, fail_count = choose_max_workers(
        to_build, cfg, store.samples_dir, calibration_n=3, headroom_gb=2,
        serial=serial, detect_peaks=detect_peaks)
    remaining_files = [f for f in to_build if f not in built]

    # build the rest in parallel, each worker writes its own IM to disk
    with ProcessPoolExecutor(max_workers=max_workers, initializer=configure_run_logging,
                             initargs=(run_dir,)) as executor:
        futures = {executor.submit(build_and_save, file, cfg, store.samples_dir, detect_peaks): file
                   for file in remaining_files}
        for future in as_completed(futures):
            file = futures[future]
            try:
                name, _ = future.result()
                built[file] = name
                logger.info(f"Created {file.name} IntensityMatrix")
                success_count += 1
            except Exception as e:
                logger.info(f"Failed to process {file.name}: {e}", exc_info=True)
                fail_count += 1

    names = [f.stem for f in files if store.is_valid(f.stem)]

    logger.info(
        "\n\n-------------------- Conversion to IntensityMatrix --------------------\n"
        f"Total Files: {len(files)}\nLoaded from saved state: {len(files) - len(to_build)}\n"
        f"Successful Builds: {success_count}\nFailed Builds: {fail_count}\nSamples Available: {len(names)}\n\n"
    )

    return names

def decode_binary_data(encoded_data, dtype, max_signal=None):
    """
    Decodes base64, decompresses zlib, and converts to a NumPy array with an associated m/z and time lists.
    Params:
        encoded_data                base64 data to be decoded
        dtype                       type of data that is converted
        max_signal                  maximum signal expected, if signal exceeds this it is set to 0
    """
    try:
        decoded = base64.b64decode(encoded_data)
        decompressed = zlib.decompress(decoded)
    except Exception as e:
        print(f"Exception:\n{e}")
        return None
    
    # generate result array, removing nan and clipping to max signal
    result = np.frombuffer(decompressed,dtype=dtype)                        # generate result array
    mask = result > 1e9
    if mask.any():
        indices = np.where(mask)[0]
        logger.info(f"Large Raw Values:{result[indices]}\nIndex values:{indices}")
    
    result =np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)          # replace nan with 0
    if max_signal is not None:
        result[result > max_signal] = 0                                     # remove excessively high signals
    return result

def bin_masses(unique_mzs, intensity_matrix, max_mz, min_mz):
    """
    Bins masses -0.3 to +0.7 of integer values
    Params:
        unique_mzs                      list of m/z values to bin
        intensity_matrix                intensity matrix that corrosponds to unbinned m/z values for processing
        min/max mz                      mz range for this run, used to prevent corrupted data from entering analysis
    
    Returns:
        binned_mzs                      list of binned mz values
        binned_matrix                   intensity matrix that has been binned
    """

    if min_mz is None:
        min_mz = 50
    if max_mz is None:
        max_mz = 1000
    
    # change unique mz list to array
    mz_array = np.asarray(unique_mzs)

    # filter out invalid entries
    valid = (mz_array >= min_mz) & ~np.isnan(mz_array) & (mz_array <= max_mz)
    mz_array = mz_array[valid]
    intensity_matrix = intensity_matrix[valid,:]

    # get bin assignments
    bin_assignments = (mz_array + 0.3).astype(int)

    # get unique bins and inverse
    binned_mzs, inverse = np.unique(bin_assignments, return_inverse = True)
    
    # prepare output matrix (rows = bins cols = time points)
    num_bins = len(binned_mzs)
    _, num_cols = intensity_matrix.shape
    binned_matrix = np.zeros((num_bins,num_cols), dtype = intensity_matrix.dtype)

    # bin masses using inverse to map each origional mz to its bin
    for src_row, bin_row in enumerate(inverse):
        binned_matrix[bin_row] += intensity_matrix[src_row]

    return list(binned_mzs), binned_matrix 

def create_scan_matrix(mzml_path, cfg, apply_threshold = False, detect_peaks=False):
    """
    Extracts spectra metadata and builds a matrix where each spectrum is
    represented by a column and each unique m/z is represented by a row from a SCAN file
    Params:
        mzml_path                   path to the mzml file to process
    Returns:
        output_matrix               intensitymatrix object based on input mzml file
    """
    tree = ET.parse(mzml_path)
    root = tree.getroot()

    namespaces = {
        '': 'http://psi.hupo.org/ms/mzml'
    }

    time_map = {}
    intensity_list = []
    unique_mzs = set()
    skipped = 0

    max_mz = cfg.get('max_mz')
    min_mz = cfg.get('min_mz')
    max_signal = cfg.get('max_signal')

    # get name of sample
    name = mzml_path.stem
    logger.info(f"Sample: {name} Identified")

    # Iterate over each <spectrum> element and get scan information
    for spectrum in root.findall('.//spectrum', namespaces):
        scan_id = spectrum.get('id')
        if scan_id:
            scan_id = scan_id.split('=')[-1]

        scan_start_time = None
        scan_list = spectrum.find('scanList', namespaces)
        if scan_list is not None:
            scan = scan_list.find('scan', namespaces)
            if scan is not None:
                cv = scan.find('.//cvParam[@name="scan start time"]', namespaces)
                if cv is not None:
                    scan_start_time = float(cv.get('value'))

        # find binary data arrays
        mz_encoded = None
        intensity_encoded = None

        for bda in spectrum.findall('./binaryDataArrayList/binaryDataArray', namespaces):
            array_type = None

            for cv in bda.findall('cvParam',namespaces):
                acc = cv.get('accession')
                if acc == "MS:1000514":
                    array_type = "mz"
                elif acc == "MS:1000515":
                    array_type = "intensity"

            binary = bda.find('binary',namespaces)
            if binary is None or not binary.text:
                continue
        
            # save encoded binary data
            if array_type == 'mz':
                mz_encoded = binary.text
            elif array_type == 'intensity':
                intensity_encoded = binary.text

        # if bianry data is not present, then skip this scan
        if mz_encoded is None or intensity_encoded is None:
            skipped += 1
            continue

        # decode the data
        mz_array = decode_binary_data(mz_encoded,dtype=np.float64, max_signal=max_signal)
        intensity_array = decode_binary_data(intensity_encoded,dtype=np.float32, max_signal=max_signal)

        # make sure array values are valid
        if mz_array is None or intensity_array is None:
            skipped += 1
            continue

        # Save metadata for the spectrum in the list
        time_map[len(intensity_list)] = float(scan_start_time)
        
        # ensure consistent lengths before zipping
        if len(mz_array) != len(intensity_array):
            skipped += 1
            continue

        # Create a dictionary for each spectrum (m/z -> intensity)
        spectrum_intensity_dict = dict(zip(mz_array, intensity_array))

        # Add the spectrum dictionary to the intensity_list
        intensity_list.append(spectrum_intensity_dict)

        # Add the m/z values to the set of unique m/z values
        unique_mzs.update(mz_array)

    logger.info(f"Sample: {name} Finsihed mzML parse")

    # show how many spectra have beens skipped
    if skipped > 0:
        logger.warning(f"File: {mzml_path.stem}\nSkipped: {skipped}\n")

    # sort unnizue mz lit
    unique_mzs = sorted(unique_mzs)

    # check max/min
    if min_mz is None:
        min_mz = 50
    if max_mz is None:
        max_mz = 1000

    # remove invalid mz values from list
    mz_array  = np.asarray(unique_mzs)
    valid = (mz_array >= min_mz) & ~np.isnan(mz_array) & (mz_array <= max_mz)
    mz_array = mz_array[valid]

    # assign bins
    bin_assignments = (mz_array + 0.3).astype(int)
    binned_mzs_arr, inverse = np.unique(bin_assignments, return_inverse=True)
    mz_to_bin_row = dict(zip(mz_array, inverse))

    # generate binnd matrix
    binned_matrix = np.zeros((len(binned_mzs_arr), len(intensity_list)))

    # fill in matrix from bins
    for col_idx, spectrum_intensity_dict in enumerate(intensity_list):
        for mz,intensity in spectrum_intensity_dict.items():
            bin_row = mz_to_bin_row.get(mz)
            if bin_row is not None:
                binned_matrix[bin_row, col_idx] += intensity
    binned_mzs = list(binned_mzs_arr)

    # add TIC row to end of matrix
    sum_row = np.sum(binned_matrix, axis=0)
    final_matrix = np.vstack((binned_matrix,sum_row))

    # add 9999 value to end of binned_mzs to represent the TIC
    binned_mzs.append(9999)

    # create intensity matrix object
    output_matrix = IntensityMatrix(intensity_matrix=final_matrix,
                                    unique_mzs=binned_mzs,
                                    cfg=cfg,
                                    sample_name=name,
                                    time_map=time_map,
                                    matrix_type="SCAN",
                                    detect_peaks=detect_peaks,
                                    apply_threshold=apply_threshold)
    logger.info(f"Sample: {name} IntensityMatrix Created")

    time_vals = output_matrix.get_time_per_scan()
    logger.info(f"\nTotal Scans: {len(time_vals['array'])}\nAvg Time per Scan: {time_vals['avg']}\nStdev: {time_vals['stdev']}\nPct Err: {(100*time_vals['stdev']/time_vals['avg']):.2f}")

    return output_matrix

def create_sim_matrix(mzml_path, cfg, detect_peaks=False):
    """
    Extracts spectra metadata and builds a matrix where each spectrum is
    represented by a column and each unique m/z is represented by a row, from a SIM file
    Params:
        mzml_path                   path to the mzml file to process
    Returns:
        output_matrix               intensitymatrix object based on input mzml file
    """
    tree = ET.parse(mzml_path)
    root = tree.getroot()

    namespaces = {
        '': 'http://psi.hupo.org/ms/mzml'
    }

    time_map = {}
    ion_map = {}

    # get name of sample
    file_name = mzml_path.stem

    # generate empty matrix for data storage
    chrom_list = root.find('.//chromatogramList', namespaces)
    num_chroms = int(chrom_list.get('count'))
    first_chrom = chrom_list.find('chromatogram',namespaces)
    num_time_points = int(first_chrom.get('defaultArrayLength'))
    matrix = np.zeros((num_chroms,num_time_points))

    int_count = 0
    time_count = 0

    # iterate over chroms and gather data
    for idx,chrom in enumerate(chrom_list):

        # get ion and add to map
        iso = chrom.find(
            './/precursor/isolationWindow/cvParam[@accession="MS:1000827"]',
            namespaces
        )
        # handle TIC ion value
        if iso is None:
            ion = 9999
            ion_map[ion] = num_chroms-1
        else:
            ion = int(0.3+float(iso.attrib["value"]))
            ion_map[ion] = int(idx)-1

        # now parse the binary data arrays, getting the time and intensity arrays
        for bda in chrom.findall('.//binaryDataArray', namespaces):
            cvparams = [child for child in list(bda) if child.tag.endswith('cvParam')]

            # handle cv blocks
            array_type = None
            dtype = None
            for cv in cvparams:
                acc = cv.attrib.get('accession')
                if acc == "MS:1000523":
                    dtype = np.float64
                elif acc == "MS:1000521": 
                    dtype = np.float32
                if acc == "MS:1000515":
                    array_type = "intensity_array"
                    int_count += 1
                elif acc == "MS:1000595":
                    array_type = "time_array"
                    time_count += 1
                elif acc == "MS:1000786":
                    array_type = "nonstandard"
            
            # grab encoded data
            if array_type == "time_array" or array_type == "intensity_array":
                encoded = bda.find('binary', namespaces).text
                decoded = decode_binary_data(encoded,dtype)

                # generate time_map (col_idx: time)
                if array_type == "time_array" and idx == 1:
                    for i,time in enumerate(decoded):
                        time_map[i] = float(time)

                # add intensity data to array
                elif array_type == "intensity_array":
                    if idx != 0:
                        matrix[idx-1] = decoded
                    # add TIC to the end of the matrix
                    else:
                        matrix[-1] = decoded

    # convert ion map to sorted list
    mzs = [ion for ion,_ in sorted(ion_map.items(), key=lambda x: x[1])]

    # get row varainces for time matrix and see if it looks good

    # create intensity matrix object and return
    output_matrix = IntensityMatrix(intensity_matrix=matrix,
                                    unique_mzs=mzs,
                                    cfg=cfg,
                                    sample_name=file_name,
                                    time_map=time_map,
                                    matrix_type="SIM",
                                    detect_peaks=detect_peaks)
    logger.info(f"Produced inntensity matrix for sample: {file_name}")
    return output_matrix

def create_intensity_matrix(mzml_path, cfg, apply_threshold=False, detect_peaks=False):
    """
    Generatews intensity matrix from mzml object, automatically detecting if it is SCAN or SIM
    Params:
        mzml_path                       Path to mzml object to analyze
    """

    # get aquisition type (SCAN or SIM)
    type = aq_type(mzml_path)

    if type == "SIM":
        matrix = create_sim_matrix(mzml_path, cfg, detect_peaks=detect_peaks)
    elif type == "SCAN":
        matrix = create_scan_matrix(mzml_path, cfg, apply_threshold=apply_threshold, detect_peaks=detect_peaks)

    return matrix

def aq_type(mzml_path: Path):
    """
    Determines if the mzML file supplied is from a SIM or SCAN run
    Params:
        mzml_path                       Path to the mzML object to be analyzed
    """

    tree = ET.parse(mzml_path)
    root = tree.getroot()

    namespaces = {
        '': 'http://psi.hupo.org/ms/mzml'
    }

    # get filecontent information
    try:
        content = root.find('.//fileDescription/fileContent',namespaces)
    except Exception as e:
        raise ValueError(f"No file content found at {mzml_path}\nError:\n{e}", exec_info=True)

    # get cvParams
    cvparams = [child for child in list(content) if child.tag.endswith('cvParam')]

    # iterate and save accession values
    for cv in cvparams:
        acc = cv.attrib.get("accession")
        if acc == "MS:1001472":
            return "SIM"
        elif acc == "MS:1000579":
            return "SCAN"

def build_and_save(mzml_path, cfg, samples_dir, detect_peaks):
    """
    worker: builds an IM, writes it to samples_dir, returns only its name + peak memory
    (the IM itself never travels back to the main process)
    """
    name = Path(mzml_path).stem
    matrix = create_intensity_matrix(mzml_path, cfg, detect_peaks=detect_peaks)
    write_sample(matrix, samples_dir, cfg, name=name)
    peak_mem = psutil.Process(os.getpid()).memory_info().peak_wset
    return name, peak_mem

def choose_max_workers(files, cfg, samples_dir, calibration_n=3, headroom_gb=2.0, serial=False, detect_peaks=True):
    """
    chooses a random number of files to use to test system and calibrate how many workers
    we can use to process samples based on availbe cpu cores and memory
    """
    if serial or not files:
        return 1, {}, 0, 0
    
    cpu_ceiling = max(1, os.cpu_count()-1)

    calibration_files = random.sample(files, min(calibration_n,len(files)))
    
    peak_mem_bytes = 0
    calibration_results  = {}
    success_count = 0
    fail_count = 0

    run_dir = get_run_dir(cfg.get("project_name"), cfg.get("run_name"))

    with ProcessPoolExecutor(max_workers=1, initializer=configure_run_logging, initargs=(run_dir,)) as executor:
        for file in calibration_files:
            try:
                name,mem = executor.submit(build_and_save, file, cfg, samples_dir, detect_peaks).result()
                peak_mem_bytes = max(peak_mem_bytes, mem)
                calibration_results[file] = name
                success_count += 1
            except Exception as e:
                logger.warning(f"Failed to process {file.name} during calibration: {e}")
                fail_count += 1
    
    if peak_mem_bytes == 0:
        logger.warning(f"Calbiration failed for all sampled files, defaulting to max_workers = 1")
        return 1, calibration_results, success_count, fail_count
        
    available_bytes = psutil.virtual_memory().available - headroom_gb * 1e9
    mem_ceiling = max(1, int(available_bytes // peak_mem_bytes))
    max_workers = min(cpu_ceiling, mem_ceiling)

    logger.info(
        "\n------------------------------ CPU Optimization ------------------------------\n"
        f"Total CPU Cores: {os.cpu_count()} | CPU Cores Available: {cpu_ceiling}\n"
        f"Memory Available (GB): {available_bytes/1e9:.2f} | Max Workers: {min(cpu_ceiling,mem_ceiling)}\n"
    )

    return max_workers, calibration_results, success_count, fail_count
