"""

Class that stores an mzml file's data as a matrix for peak identification and searching

"""

# region Imports

import h5py, copy
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from src.config_loader import ConfigLoader
from src.utils import get_run_dir, get_proj_dir, get_run_cfg_path
from src.db import insert_im

# logging
import logging
logger = logging.getLogger(__name__)

# endregion

# Class for storage and cleaning of intensity matrix extracted by mzml_processor
class IntensityMatrix:

    def __init__(self, intensity_matrix: np.ndarray,
                 unique_mzs: list,
                 cfg: ConfigLoader,
                 sample_name: str = None,
                 time_map: dict = None,
                 matrix_type: str = None,
                 detect_peaks: bool = False):
        self.intensity_matrix = intensity_matrix
        self.unique_mzs = unique_mzs
        self.time_map = time_map
        self.noise_factor = None
        self.abundance_threshold = None
        self.peak_dict = None
        self.collected_peaks = None
        self.baseline_mask = None
        self.molecule_map = None
        self.cfg = cfg
        self.sample_name = sample_name
        self.matrix_type = matrix_type

        # calculate and apply abundnace threshold transformation to intensity matrix
        self.calculate_threshold()
        self.apply_threshold()
        # calculate noise factor for this intensity matrix
        self.calculate_noise_factor()
        # identify peaks in this intensity matrix
        if detect_peaks:
            self.identify_peaks(self.intensity_matrix)

    # region                 ---------- Abundance Threshold ----------

    def calculate_threshold(self):        
        """
            Counts the number of zero to nonzero transtions for each m/z in 10 approximately equally sized time segments then takes the square root
            of these values and multiplies it by the minimum abundance measured in the entire intensity matrix, use this value to replace 0 values

            Parameters:
                intensity_matrix (np.ndarray): 2D numpy array where each row corrosponds to a m/z chromatogram intensity profile and each column
                                            corrosponds to a scan
            Returns:
                threshold_values (np.ndarray): 2D numpy array with 10 columns (1 per segment) and a row for each unique m/z in the input matrix
                                            each entry corrosponds to the calculated threshold value for that m/z in that segment
            """
        
        intensity_matrix = self.intensity_matrix
        min_value = np.min(intensity_matrix[intensity_matrix > 0])
        threshold_values = np.empty((len(self.unique_mzs), 10))
        segments = np.array_split(intensity_matrix, 10, axis=1)

        segment_starts = []
        start_idx = 0
        for seg_idx, segment in enumerate(segments):
            transitions = (segment[:, :-1] == 0) & (segment[:, 1:] > 0)
            threshold_values[:, seg_idx] = transitions.sum(axis=1) / segment.shape[1]
            segment_starts.append(start_idx)
            start_idx += segment.shape[1]

        threshold_values **= 0.5
        threshold_values *= min_value

        self.abundance_threshold = {'start_idxs': segment_starts, 'values': threshold_values}

    # takes any value in the array that is below At for that segment for that m/z value and 
    def apply_threshold(self):
        matrix = self.intensity_matrix
        starts = self.abundance_threshold['start_idxs']
        values = self.abundance_threshold['values']  # (n_ions, 10)

        for seg_idx in range(10):
            start = starts[seg_idx]
            end = starts[seg_idx + 1] if seg_idx < 9 else matrix.shape[1]
            seg_thresh = values[:, seg_idx][:, None]
            segment = matrix[:, start:end]
            matrix[:, start:end] = np.where(segment < seg_thresh, seg_thresh, segment)

        self.intensity_matrix = matrix

    # endregion

    # region                 ---------- Noise Factor Calculation ----------

    # calculates the noise factor (Nf) for the entire intensity_matrix
    def calculate_noise_factor(self):

        matrix = self.intensity_matrix

        num_segments = matrix.shape[1] // 13
        segments = []
        noise_factors = []

        #loop over the number of segments creating each segment in segments as we go
        for i in range(num_segments):
            start = i*13
            end = (i+1)*13
            segment = matrix[:, start:end]
            
            # filters out any rows that contain 0 values
            nonzero_rows = segment[~np.any(segment == 0, axis = 1)]
            
            # filter rows that cross less than 7 times
            crossing_filtered = []
            for row in nonzero_rows:
                if row.size == 0:
                    continue
                avg = np.mean(row)

                # skip rows with 0 variation
                if avg == 0:
                    continue

                crossings = self.count_crossings(row,avg)
                if crossings > 6:
                    crossing_filtered.append(row)
                
            segments.append(np.array(crossing_filtered))

        # iterate through each segment
        for segment in segments:
            for row in segment:
                current_nf = self.calculate_row_nf(row)
                if not np.isnan(current_nf):
                    noise_factors.append(current_nf)

        # fallback if no noise factors calculated:
        if len(noise_factors) == 0:
            self.noise_factor = np.nan
        else:
            self.noise_factor = np.median(noise_factors)
    
    # counts the number of times the values of an array "cross" a given average value
    def count_crossings(self,row,avg):
        crossings = 0
        for i in range(len(row)-1):
            if (row[i] < avg and row[i+1] > avg) or (row[i] > avg and row[i+1] < avg):
                crossings += 1
        return crossings

    # calculates and returns the median deviation for a given 1D array
    def calculate_row_nf(self, row):

        # calculate the mean of the row
        mean = np.mean(row)
        if mean == 0:
            return 0
        
        # calculate rest of row nf 
        sqrt_of_mean = mean ** 0.5

        # calculate deviation from the mean for all members of row
        deviations = np.abs(row-mean)

        # calculat noise factor
        nf = np.median(deviations)/sqrt_of_mean
        
        # return the median of the deviations / sqrt of the mean (Nf for that row)
        return nf

    # calculates per-row masks to use for S/N calculations
    def noise_mask(self, peak_list: list):
        """
        Calculates per-row mask to use for S/N calculations

        Params
        ------
        peak_list                   list of peaks for this row from find_maxima
        
        Returns
        -------
        masks                       bool row masks for a list of peaks (1=true 2=false)
        """
        mask = np.ones(len(self.time_map))

        for peak in peak_list:
            l = peak["left_bound"]
            r = peak["right_bound"]
            mask[l:r+1] = 0

        return mask
    
    # endregion

    # region                 ---------- Finding Maxima ----------

    # finds the peaks (maxima and bounds) for each row of a given intensity matrix and the tic, last row is TIC
    def identify_peaks(self, matrix):

        # dict to hold the lists of peak values m/z : peak_list
        peaks = {}
        masks = []

        for row_idx,row in enumerate(matrix):

            ion = self.unique_mzs[row_idx]
            row_peaks, row_nm = self.find_maxima(row,ion)
            peaks[ion] = row_peaks
            masks.append(row_nm)

        self.peak_dict = peaks
        self.baseline_mask = np.vstack(masks)

        return peaks

    def find_maxima(self, array, ion):
        """
        Uses a 2 pass appraoch to determine bounds and peak features for each detected maxima point, defines
        baseline using valley-to-valley baseline calculation

        Params
        ------
        array                           row from intensity matrix
        ion                             ion label from row of intensity matrix

        Returns
        -------
        maxima                          list of peaks for this ion's row array
        """
        sn_threshold = self.cfg.get('sn_threshold')
        prom_mult = self.cfg.get('prominance_multiplier')
        vr = self.cfg.get('valley_ratio')

        # set prominance
        median = np.nanmedian(array)
        mad = np.nanmedian(np.abs(array - median))
        prom = (median +  mad) * prom_mult

        # Excludes the first and last 12 points from the search to prevent bounding errors
        array_range = array[12:-12]

        # finds the local maxima of the given array, stores their index
        max_idxs, _ = find_peaks(array_range, prominence=prom)

        # Shifts indices found in the range for use in the original array
        max_idxs += 12

        # list to hold dictionary entries containing left_bound, right_bound and center for each maxima
        maxima = []

        # first pass, finds bounds of peaks and saves to maxima
        for peak_max in max_idxs:

            # get basic info
            left_bound = self.find_bound(array, peak_max, -1)
            right_bound = self.find_bound(array, peak_max, 1)
            fit = self.quadratic_fit(array, peak_max)

            # detect flat top peaks
            max_val = array[peak_max]
            if peak_max == left_bound or peak_max == right_bound:
                flat_top = True
            else:
                l_val = array[peak_max-1]
                r_val = array[peak_max+1]
                tol = max_val * 0.01
                flat_top = False
                if abs(l_val-max_val) <= tol or abs(r_val-max_val) <= tol:
                    flat_top = True

            maxima.append({
                'center': peak_max,
                'left_bound': left_bound,
                'right_bound': right_bound,
                'rt': fit['x_values'][1],
                'raw_height': fit['y_values'][1],
                'ion': ion,
                'flat_top': flat_top,
                'cluster': None,
                'valid': False,
                'processed': False,
                'fwhh': np.nan,
                'feature': None,
                'valley_ratio': None,
                'tailing_factor': np.nan,
                'sn_ratio': np.nan,
                'height': np.nan,
                'baseline': None,
                'bl_slope': np.nan,
                'bl_yint': np.nan,
                'conv': np.nan,
                'peak_idx': -1,
                'molecule': None,
                'cluster': None
            })

        # get noise mask for this row
        row_nm = self.noise_mask(maxima)
        bl_indices = np.where(row_nm)[0]

        # second pass, finds baseline and other relevant features
        n_clusters = 1
        time_map = self.time_map
        for i,peak in enumerate(maxima):

            # if baseline already set then skip this peak, it's already been handled
            if peak['processed']:
                continue

            # set initial anchors
            l_anchor = peak['left_bound']
            r_anchor = peak['right_bound']

            # check right bound because peaks are processed left to right
            j = i
            min_vr = 1.0
            while j < len(maxima) - 1:
                
                next_peak = maxima[j+1]

                # exit loop if peak bounds do not overlap
                if maxima[j]['right_bound'] != next_peak['left_bound']:
                    break
                
                # caluclate valley ratio
                valley_height = array[maxima[j]['right_bound']]
                valley_ratio = valley_height / min(array[peak['center']], array[next_peak['center']])

                # break if the valley ratio is small (cluster ends)
                if valley_ratio < vr:
                    break

                # get min_vr for cluster
                min_vr = min(min_vr, valley_ratio)

                # if still overlapping then extend right anchor and keep walking
                r_anchor = next_peak['right_bound']
                j += 1

            # get baseline for culster
            baseline = self.tentative_baseline(l_anchor, r_anchor, array)
            bl_array = baseline['baseline_array']

            # handle cluster value assignments
            cluster_peaks = maxima[i:j+1]
            for entry in cluster_peaks:

                # assign valley ratio
                entry['valley_ratio'] = min_vr if j > i else None

                # calculate sharpness/conv value
                conv = self.convolution_value(entry,array)
                entry['conv'] = conv

                # assign cluster identity
                if j > i:
                    entry['cluster'] = n_clusters

                # assign baseline array
                bl_start = entry['left_bound'] - l_anchor
                bl_end = entry['right_bound'] - l_anchor +1
                bl = bl_array[bl_start:bl_end]
                entry['baseline'] = bl
                entry['bl_slope'] = (bl[-1] - bl[0]) / (len(bl) - 1) if len(bl) > 1 else 0.0
                entry['bl_yint'] = bl[0]

                # find which two scans the percise max lives between
                center = entry['center']
                rt = entry['rt']
                c_time = time_map[center]
                if rt > c_time:
                    time1 = c_time
                    time2 = time_map[center+1]
                    scan1 = center
                    scan2 = center+1
                elif rt < c_time:
                    time1 = time_map[center-1]
                    time2 = c_time
                    scan1 = center-1
                    scan2 = center
                else:
                    time1 = time2 = c_time
                    scan1 = scan2 = center

                # if maximizes directly on scan then handle, else linear impute
                if scan1 == scan2:
                    bl_i = center - entry['left_bound']
                    bl_norm = bl[bl_i]
                else:
                    frac = (rt - time1) / (time2 - time1)
                    bl_i1 = scan1 - entry['left_bound']
                    bl_i2 = scan2 - entry['left_bound']
                    bl_norm = bl[bl_i1] + frac * (bl[bl_i2] - bl[bl_i1])
                
                # assign height
                entry['height'] = entry['raw_height'] - bl_norm

                # calculate signal to noise ratio
                sn_ratio = self.calculate_sn(entry, array, bl_indices)
                entry['sn_ratio'] = sn_ratio

                # S/N check
                sn_count = 0
                if entry['sn_ratio'] < sn_threshold or np.isnan(entry['sn_ratio']):
                    entry['valid'] = False
                    sn_count +=1
                else:
                    entry['valid'] = True
                
                entry['processed'] = True

            # update cluster counter
            if j > i:
                n_clusters += 1

        # filter valid/invalid peaks, return only valid peaks and count for logs
        valid_peaks = []
        for peak in maxima:
            if peak.get('valid', True):
                self.integrate_peak(peak,array)
                valid_peaks.append(peak)
                peak['tailing_factor'] = self.calculate_tailing(peak,array)
                peak['fwhh'] = self.calculate_fwhh(peak,array)
                
        # get peak indices and finish analysis
        for idx, peak in enumerate(valid_peaks):
            peak['peak_idx'] = idx
            peak['feature'] = None

        # reccalculate noise mask based on valid peaks
        valid_nm = self.noise_mask(valid_peaks)

        if len(valid_peaks) == 0 or len(maxima) == 0:
            count_valid = len(valid_peaks)
            count_invalid = len(maxima) - count_valid
            logger.debug(f"Sample: {self.sample_name} | Ion: {ion} | Valid: {count_valid} | Invalid: {count_invalid} | Total: {len(maxima)}")
        
        return valid_peaks, valid_nm

    # finds the left or right deconvolution bound for a given maxima, step = 1 for right bound step = -1 for left bound
    def find_bound(self, array, center, step, frac: float = 0.01, max_width: int = 25):

        nf = self.noise_factor
        max_value = array[center]
        counter = step
        n = len(array)

        # walk along flat-topped peaks
        while(
            abs(counter) <= max_width
            and 0 <= center + counter < n
            and array[center + counter] == max_value
        ):
            counter += step

        # force a bound if plateau is not escaped
        if abs(counter) > max_width or not (0 <= center + counter < n):

            pos = center + step * max_width

            if pos < 0:
                pos = 0
            elif pos >= n:
                pos = n-1

            return pos

        # handle normal peaks
        min_value = array[center + counter]

        # iterate up to 12 setps in given direction from center
        while(abs(counter) <= max_width and 0 <= center + counter < n):

            value = array[center + counter]
            
            # if the value at this step is less than the current min, set the min to this value
            if value < min_value:
                min_value = value

            # if the value at this step is less than frac of max close window here
            if value < frac * max_value:
                return center + counter
                
            
            # if the value at this step is more than 5 nf greater than the minimum close the window at the previous step
            if value > 5 * nf + min_value:
                return center + counter - step
            
            # increment counter
            counter += step

        # if no previous checks returned a value close window at 25 steps from the max
        pos = center + step * max_width

        if pos < 0:
            pos = 0
        elif pos >= n:
            pos = n-1

        return pos

    # finds a quadratic fit for a set of 3 points in an array
    def quadratic_fit(self, array, center):
        """
        Calculates quadratic fit through 3 points using closd-form quadradic approach
        """

        """
        Old function
        # get map to convert scans to minutes
        scan_map = self.time_map
        # x values for fit, the center index and its two direct neighbors (in minutes)
        x_points = np.array([scan_map[center-1], scan_map[center], scan_map[center+1]])
        # y values for fit, from row corrosponding to x values
        y_points = array[[center-1,center,center+1]]

        # perform quadratic numpy polyfit, returning coefficients in a,b,c for ax^2 + bx + c form
        coeffs = np.polyfit(x_points, y_points, 2)
        a,b,c = coeffs

        # calcluate the precise maxima of the fit
        max_x = -b/(2*a)

        # array of left point(max_x-1), max, and right point (max_x +1)
        x_values = np.array([max_x-1, max_x, max_x+1]).astype(float)
        # array of y values corrosponding to x_values 
        y_values = a*x_values**2 + b*x_values + c

        fit_result = {
            'x_values' : x_values,
            'y_values' : y_values,
            'coeffs' : coeffs
        }

        return fit_result
        """

        scan_map = self.time_map
        
        # get x,y values
        x0,x1,x2 = scan_map[center-1],scan_map[center],scan_map[center+1]
        y0,y1,y2 = array[center-1],array[center],array[center+1]

        # closed-from quadradic through 3 points
        f01 = (y1-y0) / (x1-x0)
        f12 = (y2-y1) / (x2-x1)
        a = (f12 - f01) / (x2-x0)

        # check for flat top (no curvature to interpolate)
        if a == 0:
            return {
                'x_values': np.array([x1-1,x1,x1+1]),
                'y_values': np.array([y1,y1,y1]),
                'coeffs': np.array([0.0,0.0,y1])
            }

        # calcualte rest
        b = f01 - a * (x0+x1)
        c = y0 - f01 * x0 + a * x0 * x1
        max_x = -b / (2*a)

        # apply the fit
        x_values = np.array([max_x-1,max_x,max_x+1])
        y_values = a*x_values**2 + b*x_values + c

        return {
            'x_values': x_values,
            'y_values': y_values,
            'coeffs': np.array([a,b,c])
        }

    # checks if peak is above rejection threshold (4 noise units is base but we can adjust in the future)
    def threshold_check(self, row, peak_idx, height):

        threshold = 4 * self.noise_factor * row[peak_idx]**0.5 

        if height < threshold:
            return False
        else:
            return True

    # calculates convolution value for a single peak, used to see if peak is singlet or not
    def convolution_value(self, peak, array):
        
        # get values from the peak dict
        peak_max = peak["center"]
        left = peak["left_bound"]
        right = peak["right_bound"]
        row = array

        # holds the sum of all rates of sharpness calculated for the peak
        rate_sum = 0

        # value to prevent divide by 0 errors
        eps = 1e-12

        # if the peak is not wide enough then return None
        if peak_max - left < 4 or right - peak_max < 4:
            return None

        # loop over all the scans in the 3 scan window and calculate rate for each
        for i in range(1,4):
            term1 = (row[peak_max+(i+1)] - row[peak_max+i]) / (row[peak_max+i] + eps)
            term2 = (row[peak_max-(i+1)] - row[peak_max-i]) / (row[peak_max-i] + eps)

            rate_sum += term1+term2

        return rate_sum

    def calculate_sn(self, peak: dict, row_array: np.ndarray, bl_indices: np.ndarray, n_closest: int = 20):
        """
        Calculates S/N ratio for a given peak using the baseline mask to determine local
        noise. If this calculation fails then falls back to avereage noise level for
        that ion.
        
        Params
        ------
        peak                            peak to calculate S/N for
        row_array                       array for this row of intensity matrix
        bl_indices                      indices where row noise mask is valid (noise point indices)
        n_closest                       how far in each direction from center to use for calculation

        Returns
        -------
        sn_ratio                        signal to noise ratio for this peak
        """
        
        center = peak['center']

        # determine n_closest points from baseline mask on either side of our peak
        left_bl = bl_indices[bl_indices < center][-n_closest:]
        right_bl = bl_indices[bl_indices > center][:n_closest]
        local_bl = np.concatenate([left_bl,right_bl])

        if len(local_bl) == 0:
            return np.nan
        
        # calculate RMS deviation
        bl_signal = row_array[local_bl]
        noise = np.max(bl_signal) - np.min(bl_signal)       # peak to peak noise, not stdev

        # fallback if baseline for noise has no variation
        if noise == 0:
            noise = 1.0

        return peak['height'] / noise

    def calculate_fwhh(self, peak: dict, row_array: np.ndarray):
        """
        Calculates FWHH for a peak
        """

        # get peak values
        half_height = peak['height'] / 2
        baseline = peak['baseline']
        left = peak['left_bound']
        right = peak['right_bound']

        # calculate signal array (bl corrected) and corrected center
        signal = row_array[left:right+1] - baseline
        center = peak['center'] - left

        # find left and right index directly after the half height value
        left_i = center
        while left_i > 0 and signal[left_i] > half_height:
            left_i -= 1
        right_i = center
        while right_i < len(signal)-1 and signal[right_i] > half_height:
            right_i += 1

        # bailout checks
        denom_l = signal[left_i+1] - signal[left_i]
        denom_r = signal[right_i-1] - signal[right_i]
        bad_left = (left_i == 0 and signal[0] > half_height) or denom_l == 0
        bad_right = (right_i == len(signal) and signal[right_i] > half_height) or denom_r == 0
        if bad_left or bad_right:
            logger.debug(
                f"Sample {self.sample_name} | Ion {peak['ion']} | RT {peak['rt']:.3f} "
                f"FWHH undefined - peak bounds do not cross half height cleanly"
            )
            return np.nan

        # interpolate precise HH time
        frac_l = (half_height - signal[left_i]) / denom_l
        frac_r = (half_height -signal[right_i]) / denom_r
        time_l = self.time_map[left+left_i] + frac_l * (self.time_map[left+left_i+1] - self.time_map[left+left_i])
        time_r = self.time_map[left+right_i] - frac_r * (self.time_map[left+right_i] - self.time_map[left+right_i-1])
        
        return time_r - time_l

    def calculate_tailing(self, peak: dict, row_array: np.ndarray):
        """
        Calculates tailing factor (ratio of center to right/center to left distances)
        calucalted as the quotiont of the distance from peak center to left and peak center
        to right over twice the distance from center to left
        """
        # get peak values
        tf_height = peak['height'] * 0.1
        baseline = peak['baseline']
        left = peak['left_bound']
        right = peak['right_bound']

        # calculate signal array (bl corrected) and corrected center
        signal = row_array[left:right+1] - baseline
        center = peak['center'] - left
        center_time = peak['rt']

        # find left and right index directly after the tailing factor height value
        left_i = center
        while left_i > 0 and signal[left_i] > tf_height:
            left_i -= 1
        right_i = center
        while right_i < len(signal)-1 and signal[right_i] > tf_height:
            right_i += 1

        # bailout checks
        denom_l = signal[left_i+1] - signal[left_i]
        denom_r = signal[right_i-1] - signal[right_i]
        bad_left = (left_i == 0 and signal[0] > tf_height) or denom_l == 0
        bad_right = (right_i == len(signal)-1 and signal[right_i] > tf_height) or denom_r == 0
        if bad_left or bad_right:
            logger.debug(
                f"Sample {self.sample_name} | Ion {peak['ion']} | RT {peak['rt']:.3f} "
                f"Tailing factor undefined - peak bounds do not cross 10% height cleanly"
            )
            return np.nan

        # interpolate precise times when tf height is reached
        frac_l = (tf_height - signal[left_i]) / denom_l
        frac_r = (tf_height -signal[right_i]) / denom_r
        time_l = self.time_map[left+left_i] + frac_l * (self.time_map[left+left_i+1] - self.time_map[left+left_i])
        time_r = self.time_map[left+right_i] - frac_r * (self.time_map[left+right_i] - self.time_map[left+right_i-1])
        
        # calculate tailing factor
        a = center_time - time_l
        b = time_r - center_time

        return (a+b)/(2*a)

    # endregion

    # region                 ---------- Baseline Calculation ----------

    # calculates a tentative baseline for a percieved component (minutes not scans)
    def tentative_baseline(self,left_bound,right_bound,array):

        # create componenet array
        component_array = array[left_bound:right_bound+1]
        if len(component_array) != right_bound - left_bound + 1:
            raise ValueError(f"Baseline bounds [{left_bound}, {right_bound}] exceed array length {len(array)}")

        # get the index of the peak maximum
        max_idx = np.argmax(component_array)
        # if peak is flat top then assign the max to the midpoint
        if max_idx == 0 or max_idx == (right_bound - left_bound):
            max_idx = (right_bound - left_bound) // 2 
        
        # get the index values of the minimum on the left and on the right of the max
        left_idx = np.argmin(component_array[:max_idx])
        right_idx = np.argmin(component_array[max_idx:]) + max_idx

        # get the intensity values associated with both these minimums
        left_val = component_array[left_idx]
        right_val = component_array[right_idx]

        # get linear baseline variables
        m = (right_val - left_val) / (right_idx - left_idx)
        b = left_val - m * left_idx

        # generate tentative baseline array
        baseline_array = m * np.arange(len(component_array)) + b

        # shfit baseline down if any of its values are greater than the value of input array at same index
        diffs = []
        for idx, element in enumerate(baseline_array):
            if element > component_array[idx]:
                diff = element - component_array[idx]
                diffs.append(diff)

        # handle y-int/baseline array correction (shifting up/down)
        if len(diffs) > 0:
            y_correct = max(diffs)
            baseline_array -= y_correct
        else:
            y_correct = 0

        # save and return values
        output = {
            'baseline_array' : baseline_array,
            'slope': m,
            'y_int': b-y_correct,
            'left_bound': left_bound,
            'right_bound': right_bound
        }
        return output

    # endregion

    # region                 ---------- Data Collection ----------

    def generate_spectra(self, peak: dict, label: str = "Unknown", n_closest: int = 10, save_spectra: bool = False):
        """
        Generates a spectra for a given peak, takes signal from all ions at the peak's center scan, subtracts
        local noise and generates a spectra.  Spectra is then normalized to relative abundance, and any peaks
        less than 5% of the largest peak are dropped
        """
        center = peak['center']

        if peak is None:
            return None, None

        if np.isnan(center):
            logger.info(f"Sample {self.sample_name} peak {label} not found for spectra geneartion")
            return

        # exclude TIC from spectrum
        mask = np.ones(len(self.unique_mzs), dtype=bool)
        tic_idx = self.unique_mzs.index(9999)
        mask[tic_idx] = False
        
        # build raw values for spectra, removing baseline with noise mask
        raw_vals = []
        real_indices = np.where(mask)[0]
        for row_idx in real_indices:
            row = self.intensity_matrix[row_idx]
            row_nm = self.baseline_mask[row_idx]
            bl_indices = np.where(row_nm)[0]

            left = bl_indices[bl_indices < center][-n_closest:]
            right = bl_indices[bl_indices > center][:n_closest]
            local_bl = np.concatenate([left,right])

            mean_noise = np.mean(row[local_bl]) if len(local_bl) > 0 else 0

            raw_vals.append(max(0, row[center] - mean_noise))
        raw_vals = np.array(raw_vals)

        # convert to relative abundance
        max_val = np.nanmax(raw_vals)
        rel_vals = 100 * raw_vals / max_val

        # filter peaks less than 5% rel abundance
        rel_vals[rel_vals < 5] = 0

        mzs = np.array(self.unique_mzs)[mask]

        spectra = {
            'mzs': mzs,
            'abundances': rel_vals
        }

        if save_spectra:
            peak['spectrum'] = spectra

        return mzs, rel_vals

    def closest_peak(self, mz: int, rt: float):
        """
        finds the peak closest to the given retention time (rt) value in a given ion chromatogram for ion mz
        Params:
            mz                      M/Z ion chromatogram to search
            rt                      retention time of the peak of interest
        Returns:
            closest_peak            peak from that ion list that is closest to the supplied RT
        """

        cfg = self.cfg
        threshold = cfg.get("rt_threshold")
        mz = np.int64(mz)
        rt = float(rt)

        try:
            # get the peak list for this row
            peaks = self.peak_dict[mz]

        except Exception as e:
            logger.warning(f"Error locating ion chromatogram for ion: {mz}\n{e}\nUnique mzs:\n {self.unique_mzs}")
            return None

        # if no peaks are found raise error
        if len(peaks) == 0:
            logger.debug(f"No peaks found for {self.sample_name} Ion {mz}")
            return None
        
        # find the peak closest to specified RT
        try:
            closest_peak = min(
                peaks,
                key = lambda p: abs(float(p['rt']) - rt)
            )
        except Exception as e:
            logger.debug(f"No peaks availbe in ion chromatogram for ion: {mz}\n{e}")
            return None
        
        # copy peak for collection
        result = copy.copy(closest_peak)

        # find rt difference (positive if RT > real value negative if RT < real value)
        if  np.isnan(result['rt']):
            diff = np.nan
        else:
            diff = rt - result['rt']

        # save rt diff
        result['rt_diff'] = diff
        
        # update rt valid flag and save
        if np.isnan(diff) or abs(diff) > threshold:
            result["rt_valid"] = False
        else:
            result["rt_valid"] = True
        
        return result
    
    def integrate_peak(self, peak: dict, array: np.ndarray):
        """
        uses trapazoidal integration to get a peak area value
        Params:
            peak                    dict entry for the peak to be integrated
        """
        # get symmetry threshold
        cfg = self.cfg
        end_threshold = cfg.get("endpoint_threshold")

        # start and end idx for this peak in intensity matrix
        start = peak['left_bound']
        end = peak["right_bound"]

        # get correct ion chromatogram
        row = array

        # get this peak's abundance array
        signal =  row[start:end+1]

        # check to see how close the endpoints are
        left = signal[0]
        right = signal[-1]
        max_val = signal[peak['center'] - peak["left_bound"]]
        
        # calculate % difference from each endpoint to max
        left_diff = 100 * (max_val - left) / max_val
        right_diff = 100 * (max_val - right) / max_val

        # calculate the diff between endpoints
        symmetry = left_diff - right_diff
        peak["bound_symmetry"] = symmetry
        if abs(symmetry) > end_threshold:
            peak["symmetry_valid"] = False
        else:
            peak["symmetry_valid"] = True

        # adjust to baseline
        if len(signal) != len(peak['baseline']):
            print(f"Array length: {len(signal)}\nBaseline length: {len(peak['baseline'])}")
            raise ValueError("baseline and signal arrays are of different length")
        
        net = signal - peak['baseline']
        time_points = np.array([self.time_map[start+i] for i in range(len(net))])

        # tarpazoidal integrate the net value
        peak_area = np.trapezoid(y=net, x=time_points)
        peak["area"] = peak_area
    
    def collect_data(self, molecules: list, mzs: list, rts: list):
        """
        Collects all peaks from a given list of matrices that corrospond to molecule/mz/rt gropuing specified
        Params:
            matrices                            list of IntensityMatrix objects to parse
            molecules,mzs,rts                   lists (index matched) of moleucle,mz,rt triplets
        Returns:
            output                              dict of sample_name: peak list values
        """
        samlpe_name = self.sample_name
        logger.info(f"--------------------Processing Sample {samlpe_name}--------------------")
        
        molecule_map = {}
        peaks = []

        for idx,molecule in enumerate(molecules):
            peak = self.closest_peak(mzs[idx],rts[idx])
            if peak is None:
                logger.info(f"Skipped {molecule} peak in {samlpe_name} sample due to no peaks found")
                continue
            if not peak['rt_valid']:
                logger.info(f"Skipped {molecule} peak in {samlpe_name} sample due to RT invalid")
                continue
            peak["molecule"] = molecule

            # generate im slice
            row_i = self.unique_mzs.index(peak['ion'])
            l = peak['left_bound']
            r = peak['right_bound']

            im_slice = self.intensity_matrix[row_i][l:r+1].astype(float)

            peak['peak_array'] = im_slice

            peaks.append(peak)

            molecule_map[molecule] = (peak['ion'], peak['peak_idx'])
            self.molecule_map = molecule_map

        logger.info(f"Molecules Queried: {len(molecules)} | Peaks Found: {len(peaks)} | Pct Found {(100 * (len(peaks)/len(molecules)))}")

        return peaks

    # endregion

    # region                 ---------- Data Visualization ----------

    def width_histogram(self):
        widths = []
        for row in self.peak_list:
            for peak in row:
                width = peak["right_bound"] - peak["left_bound"]
                widths.append(width)
        
        max_width = max(widths)
        min_width = min(widths)

         # bin
        bins = np.arange(
            min_width,
            max_width + 2,
            1
        )
        
        plt.figure(figsize=(8,5))
        plt.hist(widths,bins=bins)
        plt.xlabel("Peak width (scans)")
        plt.ylabel("Count")
        plt.title("Peak Width Distribution")
        plt.tight_layout()
        plt.show()

    def plot_ic(self, mz: int):
        """
        Plots a given m/z ion chromatogram for visualization
        """
        print(f"Noise Factor: {self.noise_factor}")
        row_idx = self.unique_mzs.index(mz)
        row = self.intensity_matrix[row_idx]

        plt.plot(row)
        plt.xlabel("Index")
        plt.ylabel("Abundance")
        plt.title(f"{mz} Ion Chromatogram")
        plt.show()

    # endregion

    # region                 ---------- Data Storage ----------

    def save_sql_im(self, conn, run_name: str):
        """
        saves this intensity matrix object to the sql database
        
        Returns
        -------
        imID to use to query this object later
        """

        return insert_im(conn,
                    self.sample_name,
                    run_name,
                    self.matrix_type,
                    self.noise_factor,
                    self.intensity_matrix.shape[0],
                    self.intensity_matrix.shape[1])

    def save_h5_object(self, proj_name: str, run_name: str):
        """
        Saves intensity matrix object to a .h5 file in the save_dir
        """

        rundir = get_run_dir(proj_name, run_name)
        rundir.mkdir(exist_ok=True,parents=True)
        h5_file = rundir / f"{run_name}.h5"

        with h5py.File(h5_file, 'a') as f:

            # define group
            grp = f.require_group(f"intensity_matrices/{self.sample_name}")

            # store compressed intensity matrix
            grp.create_dataset('intensity_matrix',
                               data=self.intensity_matrix,
                               compression = 'gzip',
                               compression_opts = 4,
                               chunks = True)
            # store compressed bool baseline mask matrix
            grp.create_dataset('baseline_mask',
                               data=self.baseline_mask,
                               compression = 'gzip',
                               compression_opts = 4,
                               chunks = True)
            # store time and ion maps as well as group atts
            grp.create_dataset('time_array', data = np.array(list(self.time_map.values())))
            grp.create_dataset('unique_mzs', data = np.array(self.unique_mzs))

    @staticmethod
    def load_h5_object(sample_name: str, proj_name: str, run_name: str):
        """
        Loads the .h5 object for a given sample

        Params
        ------
        sample_name                     name of the sample to retreive

        Returns
        -------
        im                              rebuilt IntensityMatrix obj
        """
        proj_dir = get_proj_dir(proj_name)
        db_dir = proj_dir / run_name
        h5_file = db_dir / f"{run_name}.h5"
        cfg_path = get_run_cfg_path(proj_name,run_name)
        cfg = ConfigLoader(cfg_path)

        if not db_dir.exists():
            raise FileNotFoundError(f"Database directory not found: {db_dir}")
        if not h5_file.exists():
            raise FileNotFoundError(f"H5 data file not found: {h5_file}")
        
        with h5py.File(h5_file, 'r') as f:
            grp = f[f"intensity_matrices/{sample_name}"]

            # intensity matrix
            intensity_matrix = grp['intensity_matrix'][:]
            baseline_mask = grp['baseline_mask'][:]
            time_array = grp['time_array'][:]
            unique_mzs = list(grp['unique_mzs'][:])

            # reconstruct time map
            time_map = {i:t for i,t in enumerate(time_array)}

        
        im = IntensityMatrix(intensity_matrix=intensity_matrix,
                             unique_mzs=unique_mzs,
                             cfg=cfg,
                             sample_name=sample_name,
                             time_map=time_map,
                             detect_peaks=True)
        im.baseline_mask = baseline_mask

        return im

    # endregion
