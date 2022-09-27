import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import scipy



################### COMPUTE QUALITY PARAMETERS

def compute_quality_parameters(image, n_scanlines=10, visualize_scanlines_onRoiImage=False, 
                                visualize_scanlines_qualityParameters=False, verbose_timing=False): 
    """Compute the barcode quality parameters on the given image containing a barcode.

    The quality parameters are computed one each scanline, and they are the following.
    - Minimum reflectance, i.e. R_min.
    - Symbol Contrast, i.e. SC. For computing it, also the maximum reflectance, i.e. R_max, is taken into account.
    - Minimum Edge Contrast, i.e. EC_min.
    - MODULATION.
    - DEFECT. For computing it, also the maximum Element Reflectance Non-uniformity, i.e. ERN_max, is taken into account.
    For each of these parameters, a numerical value is computed, and a symbolic grade is assigned, between 'A' and 'F'. 
    In addition, a symbolic grade and a numerical value are assigned to the scanline.

    Finally, an overall symbolic grade and an overall numerical value are assigned to the whole barcode.

    Parameters
    ----------
    image : np.array
        Given input image, containing the barcode
    n_scanlines : int, optional
        Number of scanlines, by default 10
    visualize_scanlines_onRoiImage : bool, optional
        Whether to visualize or not the roi image with the scanlines, by default False
    visualize_scanlines_qualityParameters : bool, optional
        Whether to visualize or not the `n_scanlines` scanlines with the computed quality parameters, by default False
    verbose_timing : bool, optional
        Whether to access the timing information about the process, by default False

    Returns
    -------
    overall_quality_parameters_dict : dict
        Dictionary containing the computed quality parameters. The keys are the following.
        - For each scanline 'i', with 'i' in [0,`n_scanlines`-1], the key 'scanline_i' is present in the dictionary.
          `overall_quality_parameters_dict['scanline_i']` is itself a dictionary, containing the quality parameters of that 
          scanline. In particular, the keys are the following.
          * 'R_min' and 'R_min_grade'
          * 'SC' and 'SC_grade'
          * 'EC_min' and 'EC_min_grade'
          * 'MODULATION' and 'MODULATION_grade'
          * 'DEFECT' and 'DEFECT_grade'
          * 'scanline_min_grade' and 'scanline_min_value'
        - 'OVERALL_NUMERICAL_VALUE' and 'OVERALL_SYMBOL_GRADE' which represent, respectively, the overall numerical value and
          the overall symbol grade of the barcode.
        - 'R_min_MEAN' and 'R_min_MEAN_grade' which represent, respectively, the mean 'R_min' value across the scanlines and 
          the corresponding grade.
        - 'SC_MEAN' and 'SC_MEAN_grade' which represent, respectively, the mean 'SC' value across the scanlines and the 
          corresponding grade.
        - 'EC_min_MEAN' and 'EC_min_MEAN_grade' which represent, respectively, the mean 'EC_min' value across the scanlines 
          and the corresponding grade.
        - 'MODULATION_MEAN' and 'MODULATION_MEAN_grade' which represent, respectively, the mean 'MODULATION' value across the
          scanlines and the corresponding grade.
        - 'DEFECT_MEAN' and 'DEFECT_MEAN_grade' which represent, respectively, the mean 'DEFECT' value across the
          scanlines and the corresponding grade.  

    """    
    # Detect and refine the bounding box surrounding the barcode in the given image
    roi_image, bb_points_sorted_rot, bb_width, bb_height = detect_and_refine_boundingBox(image, 
                            use_same_threshold=False, compute_barcode_structure_algorithm=1, verbose_timing=False,
                            outlier_detection_level=0.02, visualization_dict=None)

    start_time = time.time()
    
    # Pixel indices of the scanlines along the height of the roi image
    scanlines_indices = np.linspace(start=0, stop=bb_height, num=n_scanlines+2, dtype=int)[1:-1]    

    # Normalized roi image. Each pixel is a real number between 0 and 100
    roi_image_norm = 100*(roi_image/255)
    
    if visualize_scanlines_onRoiImage:  # Visualize the roi image with the scanlines
        roi_image_tmp = roi_image.copy()
        roi_image_tmp = cv2.cvtColor(roi_image_tmp, cv2.COLOR_GRAY2RGB)
        roi_image_tmp[scanlines_indices,:,:] = np.array([255,0,0])
        plt.figure()
        plt.imshow(roi_image_tmp, 'gray') 
        plt.title('ROI image, with the 10 scanlines')
    
    if visualize_scanlines_qualityParameters:
        fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(16,16), squeeze=True)
    
    # List of the quality parameters dictionaries related to the single scanlines
    quality_parameters_dict_scanline_list = []
    for i, scanline_index in enumerate(scanlines_indices):
        # Current scanline
        scanline = roi_image_norm[scanline_index, :]

        # Quality parameters dictionary related to the current scanline
        quality_parameters_dict_scanline = compute_quality_parameters_scanline(scanline)
        quality_parameters_dict_scanline_list.append(quality_parameters_dict_scanline)

        if visualize_scanlines_qualityParameters:  # Visualize the quality parameters of the currrent scanline
            R_min = quality_parameters_dict_scanline['R_min']
            R_max = quality_parameters_dict_scanline['R_max']
            GB = quality_parameters_dict_scanline['GB']

            ECmin_spaceElement_minIndex = quality_parameters_dict_scanline['modulation_dict']['ECmin_spaceElement_minIndex']
            ECmin_barElement_maxIndex = quality_parameters_dict_scanline['modulation_dict']['ECmin_barElement_maxIndex']
            ECmin_spaceElement_minValue = scanline[ECmin_spaceElement_minIndex]
            ECmin_barElement_maxValue = scanline[ECmin_barElement_maxIndex]

            ERNmax_peak_index = quality_parameters_dict_scanline['defect_dict']['ERNmax_peak_index']
            ERNmax_valley_index = quality_parameters_dict_scanline['defect_dict']['ERNmax_valley_index']
            ERNmax_peak_value = scanline[ERNmax_peak_index]
            ERNmax_valley_value = scanline[ERNmax_valley_index]

            axs[i//2,i%2].plot(scanline)
            axs[i//2,i%2].axhline(R_min, c='g', label='R_min')
            axs[i//2,i%2].axhline(R_max, c='y', label='R_max')
            axs[i//2,i%2].axhline(GB, c='r', label='GB')
            axs[i//2,i%2].plot([ECmin_spaceElement_minIndex, ECmin_barElement_maxIndex], 
                               [ECmin_spaceElement_minValue, ECmin_barElement_maxValue], 
                                c='m', label='EC_min')
            axs[i//2,i%2].plot([ERNmax_peak_index, ERNmax_valley_index], 
                               [ERNmax_peak_value, ERNmax_valley_value], 
                                c='k', label='ERN_max')
            axs[i//2,i%2].set_title(f'Scanline {i}, quality parameters')
            axs[i//2,i%2].legend()

    # Compute the overall quality parameters dictionary, given the list of quality parameters dictionary on the single 
    # scanlines
    overall_quality_parameters_dict = compute_overall_quality_parameters_dict(quality_parameters_dict_scanline_list)

    end_time = time.time()
    if verbose_timing:  # Show timing information
        print('Computing quality parameters time:', end_time-start_time)
        print()

    return overall_quality_parameters_dict



############# COMPUTE QUALITY PARAMETERS ON ONE SCANLINE

def compute_quality_parameters_scanline(scanline):
    """Compute the quality parameters on the given scanline.

    The computed quality parameters are the following.
    - Minimum reflectance, i.e. R_min.
    - Maximum reflectance, i.e. R_max.
    - Symbol Contrast, i.e. SC. 
    - Global Threshold, i.e. GB.
    - Minimum Edge Contrast, i.e. EC_min.
    - MODULATION.
    - DEFECT. For computing it, also the maximum Element Reflectance Non-uniformity, i.e. ERN_max, is taken into account.

    Parameters
    ----------
    scanline : np.array
        Mono-dimensional array, represnting the scanline. Basically, it is a 'row' of the roi image: it contains pixels 
        intensities.

    Returns
    -------
    quality_parameters_dict_scanline : dict
        Dictionary containing the computed quality parameters on the scanline of interest. The keys are the following.
        - 'R_min'
        - 'R_max'
        - 'SC'
        - 'GB'
        - 'modulation_dict', dictionary containing all the information involving MODULATION. The keys are the following.
          * 'EC_min'
          * 'MODULATION'
          * 'ECmin_spaceElement_minIndex' and 'ECmin_barElement_maxIndex', localizing EC_min in the scanline. In particular, 
            'ECmin_spaceElement_minIndex' is the pixel index in the scanline of the minimum space element value, while 
            'ECmin_barElement_maxIndex' is the pixel index in the scanline of the maximum bar element value.
        - 'defect_dict', dictionary containing all the information involving DEFECT. The keys are the following.
          * 'DEFECT'
          * 'ERNmax_peak_index' and 'ERNmax_valley_index', localizing ERN_max in the scanline. In particular, 
            'ERNmax_peak_index' is the pixel index in the scanline of the peak, while 'ERNmax_valley_index' is the pixel 
            index in the scanline of valley.

    """
    # Compute R_min, R_max, SC, GB
    R_min = scanline.min()
    R_max = scanline.max()    
    SC = R_max - R_min    
    GB = R_min + SC/2
    
    # DETECTING THE EDGES
    # Boolean mask (0/1) saying which pixel of the scanline belongs to a bar and which pixel belongs to a space
    mask = (scanline<GB).astype(int)
    mask_1 = mask[1:]  # Mask shifted by one position to the left (deleted the first pixel)
    mask_2 = mask[:-1]  # Mask shifted by one position to the right (deleted the last pixel)
    # Boolean mask (False/True) saying which pixel of the scanline is an edge pixel
    edges_mask = np.abs(mask_1 - mask_2).astype(bool)  
    edges_mask = np.append([False], edges_mask)  # We add also the information for the first pixel, which is for sure not an edge
    # Flag saying if the last pixel of the scanline is an edge or not
    last_pixel_edge = edges_mask[-1]
    # We force the last pixel of the scanline to be not an edge
    edges_mask[-1] = False
    # Pixel indices of the edges in the scanline
    edges_indices = np.indices(mask.shape)[0][edges_mask]
    # Number of elements in the scanline
    n_elements = len(edges_indices)+1
    
    # DETECTING THE PEAKS AND THE VALLEYS
    # Indices of the peaks in the scanline
    peaks_indices = scipy.signal.find_peaks(scanline)[0]
    # Indices of the valleys in the scanline
    valleys_indices = scipy.signal.find_peaks(100-scanline)[0]
    
    # Compute MODULATION
    EC_min, MODULATION, ECmin_spaceElement_minIndex, ECmin_barElement_maxIndex  = compute_modulation(scanline, SC, GB, edges_indices, 
                                                                                          n_elements, last_pixel_edge, 
                                                                                          peaks_indices, valleys_indices)

    # Compute DEFECT
    DEFECT, ERNmax_peak_index, ERNmax_valley_index = compute_defect(scanline, SC, edges_indices, n_elements, last_pixel_edge, 
                                                       peaks_indices, valleys_indices)

    # Build the quality parameters dictionary
    quality_parameters_dict_scanline = {
        'R_min': R_min,
        'R_max': R_max,
        'SC': SC,
        'GB': GB,

        'modulation_dict': {
            'EC_min': EC_min,
            'MODULATION': MODULATION,
            'ECmin_spaceElement_minIndex': ECmin_spaceElement_minIndex,
            'ECmin_barElement_maxIndex': ECmin_barElement_maxIndex
        },

        'defect_dict': {
            'DEFECT': DEFECT,
            'ERNmax_peak_index': ERNmax_peak_index,
            'ERNmax_valley_index': ERNmax_valley_index
        }
    }

    return quality_parameters_dict_scanline


def compute_modulation(scanline, SC, GB, edges_indices, n_elements, last_pixel_edge, peaks_indices, valleys_indices): 
    """Compute MODULATION on the given scanline.

    Parameters
    ----------
    scanline : np.array of float
        Mono-dimensional array, represnting the scanline. Basically, it is a 'row' of the roi image: it contains pixels 
        intensities.
    SC : float
        Symbol Contrast of the scanline
    GB : float
        Global Threshold of the scanline
    edges_indices : np.array of int
        Pixel indices of the edges in the scanline
    n_elements : int
        Number of elements in the scanline
    last_pixel_edge : bool
        Whether the last scanline pixel is an edge or not
    peaks_indices : np.array of int
        Pixel indices of the peaks in the scanline
    valleys_indices : np.array of int
        Pixel indices of the valleys in the scanline

    Returns
    -------
    EC_min : float
        EC_min of the scanline
    MODULATION : float
        MODULATION of the scanline
    ECmin_spaceElement_minIndex : int
        Pixel index in the scanline of the minimum space element value associated to EC_min
    ECmin_barElement_maxIndex : int
        Pixel index in the scanline of the maximum bar element value associated to EC_min

    Raises
    ------
    ValueError
        If the scanline contains an anomalus situation

    """
    # List of EC value, one for each pair of adjacent elements in the scanline
    EC_list = []
    # List of couples (spaceElement_min_index, barElement_max_index), one for each pair of adjacent elements in the scanline.
    # One element is a space, the other a bar.
    # 'spaceElement_min_index' represents the pixel index in the scanline of the space element minimum value associated to EC.
    # 'barElement_max_index' represents the pixel index in the scanline of the bar element minimum value associated to EC.
    peaksValleys_adjacentElements_list = []

    # Iterating across all pairs of adjacent elements
    for i in range(n_elements-1):
        # Current pair of adjacent elements: elements 'i' and 'i'+1.
        # We refer to them as 'element_1' and 'element_2'
        
        # Compute the indices of the first and last pixels of the two elements.
        if i==0:  # 'element_1' is the very first element in the scanline
            element1_first_index = 0
            element1_last_index = edges_indices[0]-1
            element2_first_index = edges_indices[0]
            element2_last_index = edges_indices[1]-1
        else:  # 'element_1' is not the very first element in the scanline 
            element1_first_index = edges_indices[i-1]
            element1_last_index = edges_indices[i]-1
            element2_first_index = edges_indices[i]
            if i!=n_elements-2:  # 'element_2' is not the very last element in the scanline
                element2_last_index = edges_indices[i+1]-1
            else:  # 'element_2' is the very last element in the scanline
                element2_last_index = scanline.shape[0]-1
                if last_pixel_edge:  # Particular case: the last pixel of the scanline is an edge pixel, therefore we don't 
                                     # consider it in the last element
                    element2_last_index -= 1

        # Whether the element 1 is a bar or a space
        element1_bar_flag = scanline[element1_first_index]<GB
        # Whether the element 2 is a bar or a space
        element2_bar_flag = scanline[element2_first_index]<GB
        
        # Anomalus situation, it should not happen
        if (element1_bar_flag and element2_bar_flag) or (not element1_bar_flag and not element2_bar_flag):
            raise ValueError('Discordant adjacent elements')

        # Compute the indices of the first and last pixels of the bar and space elements
        if element1_bar_flag and not element2_bar_flag:
            barElement_first_index, barElement_last_index = element1_first_index, element1_last_index
            spaceElement_first_index, spaceElement_last_index = element2_first_index, element2_last_index
        else:
            barElement_first_index, barElement_last_index = element2_first_index, element2_last_index
            spaceElement_first_index, spaceElement_last_index = element1_first_index, element1_last_index
        
        # Indices of the valleys within the bar element
        valleys_indices_within_barElement = valleys_indices[np.logical_and(valleys_indices>=barElement_first_index,
                                                                           valleys_indices<=barElement_last_index)]
        # No valleys within the bar element: we skip this current pair of adjacent elements
        if valleys_indices_within_barElement.size==0:
                continue
        # Index of the max value in the bar element (more precisely, max value among the valleys in the bar element)
        barElement_max_index = valleys_indices_within_barElement[np.argmax(scanline[valleys_indices_within_barElement])]
        # Max value in the bar element
        barElement_max_value = scanline[barElement_max_index]

        # Indices of the peaks within the space element
        peaks_indices_within_spaceElement = peaks_indices[np.logical_and(peaks_indices>=spaceElement_first_index,peaks_indices<=spaceElement_last_index)]
        # No peaks within the space element: we skip this current pair of adjacent elements
        if peaks_indices_within_spaceElement.size==0:
                continue
        # Index of the min value in the space element (more precisely, min value among the peaks in the bar element)
        spaceElement_min_index = peaks_indices_within_spaceElement[np.argmin(scanline[peaks_indices_within_spaceElement])]
        # Min value in the space element
        spaceElement_min_value = scanline[spaceElement_min_index]#np.min(scanline[indicesOfInterest_within_spaceElement])

        # EC value in the current pair of adjacent elements
        EC_current = spaceElement_min_value-barElement_max_value
        EC_list.append(EC_current)
        peaksValleys_adjacentElements_list.append((spaceElement_min_index, barElement_max_index))
        
    # Compute EC_min
    EC_min = min(EC_list)
    # Compute MODULATION
    MODULATION = EC_min / SC

    # Pixels indices localizing EC_min.
    # Namely, the index of the space element min value and the index of the bar element max value, associated to EC_min.
    ECmin_spaceElement_minIndex, ECmin_barElement_maxIndex = peaksValleys_adjacentElements_list[np.argmin(EC_list)]
    
    return EC_min, MODULATION, ECmin_spaceElement_minIndex, ECmin_barElement_maxIndex 


def compute_defect(scanline, SC, edges_indices, n_elements, last_pixel_edge, peaks_indices, valleys_indices):
    """Compute DEFECT on the given scanline.

    Parameters
    ----------
    scanline : np.array of float
        Mono-dimensional array, represnting the scanline. Basically, it is a 'row' of the roi image: it contains pixels 
        intensities.
    SC : float
        Symbol Contrast of the scanline
    edges_indices : np.array of int
        Pixel indices of the edges in the scanline
    n_elements : int
        Number of elements in the scanline
    last_pixel_edge : bool
        Whether the last scanline pixel is an edge or not
    peaks_indices : np.array of int
        Pixel indices of the peaks in the scanline
    valleys_indices : np.array of int
        Pixel indices of the valleys in the scanline

    Returns
    -------
    DEFECT : float
        DEFECT quality parameter
    ERNmax_peak_index : int
        Pixel index in the scanline of the peak associated to ERN_max
    ERNmax_valley_index : int
        Pixel index in the scanline of the valley associated to ERN_max

    """
    # List of ERN value, one for each element in the scanline
    ERN_list = []
    # List of couples (ERN_peak_index,ERN_valley_index), one for each element in the scanline. 'ERN_peak_index' represents 
    # the pixel index in the scanline of the peak associated to ERN, 'ERN_peak_index' represents the pixel index in the 
    # scanline of the valley associated to ERN.
    peaksValleys_elements_list = []

    # Iterating across all the elements in the scanline
    for i in range(n_elements):
        # Current element 'i': computing its first and last pixels indices
        if i==0:  # First element in the scanline
            element_first_index = 0
            element_last_index = edges_indices[0]-1
        elif i<n_elements-1:  # Intermediate element in the scanline
            element_first_index = edges_indices[i-1]
            element_last_index = edges_indices[i]-1
        else:  # Last element in the scanline
            element_first_index = edges_indices[n_elements-2]
            element_last_index = scanline.shape[0]-1
            if last_pixel_edge:  # Particular case: the last pixel is an edge pixel, therefore we don't consider it in the 
                                 # last element
                element_last_index -= 1

        # Indices of the peaks within the current element
        peaks_indices_within_element = peaks_indices[np.logical_and(peaks_indices>=element_first_index, 
                                                                    peaks_indices<=element_last_index)]
        # Indices of the valleys within the current element
        valleys_indices_within_element = valleys_indices[np.logical_and(valleys_indices>=element_first_index,valleys_indices<=element_last_index)]
        
        # Particular case: either no peaks or no valleys in the current element.
        # We skip this element
        if peaks_indices_within_element.size==0 or valleys_indices_within_element.size==0:
            continue

        # Peaks values within the current element
        peaks_within_element = scanline[peaks_indices_within_element]
        # Valleys values within the current element
        valleys_within_element = scanline[valleys_indices_within_element]

        # ERN value in the current element
        ERN_current = peaks_within_element.max()-valleys_within_element.min()
        ERN_list.append(ERN_current)

        # Peak index associated to the ERN in the current element
        ERNcurrent_peak_index = peaks_indices_within_element[np.argmax(peaks_within_element)]
        # Valley index associated to the ERN in the current element
        ERNcurrent_valley_index = valleys_indices_within_element[np.argmin(valleys_within_element)]
        peaksValleys_elements_list.append((ERNcurrent_peak_index,ERNcurrent_valley_index))
    
    # Compute ERN_max
    ERN_max = max(ERN_list)
    # Compute DEFECT
    DEFECT = ERN_max / SC
    
    # Compute the pixels indices localizing ERN_max.
    # Namely, the index of the peak and the index of the valley associated to ERN_max.
    ERNmax_peak_index, ERNmax_valley_index = peaksValleys_elements_list[np.argmax(ERN_list)]
    
    return DEFECT, ERNmax_peak_index, ERNmax_valley_index



######################### AUXILIARY FUNCTIONS

def compute_overall_quality_parameters_dict(quality_parameters_dict_scanline_list):
    """
    Compute the overall quality parameters dictionary on the whole barcode, given the list of quality parameters dictionaries
    on the single scanlines.

    Parameters
    ----------
    quality_parameters_dict_scanline_list : list of dict
        List of quality parameters dictionaries on the single scanlines.
        Each dictionary `quality_parameters_dict_scanline_list[i]` contains the computed quality parameters on the scanline 
        of interest. The keys are the following.
        - 'R_min'
        - 'R_max'
        - 'SC'
        - 'GB'
        - 'modulation_dict', dictionary containing all the information involving MODULATION. The keys are the following.
          * 'EC_min'
          * 'MODULATION'
          * 'ECmin_spaceElement_minIndex' and 'ECmin_barElement_maxIndex', localizing EC_min in the scanline. In particular, 
            'ECmin_spaceElement_minIndex' is the pixel index in the scanline of the minimum space element value, while 
            'ECmin_barElement_maxIndex' is the pixel index in the scanline of the maximum bar element value.
        - 'defect_dict', dictionary containing all the information involving DEFECT. The keys are the following.
          * 'DEFECT'
          * 'ERNmax_peak_index' and 'ERNmax_valley_index', localizing ERN_max in the scanline. In particular, 
            'ERNmax_peak_index' is the pixel index in the scanline of the peak, while 'ERNmax_valley_index' is the pixel 
            index in the scanline of valley.

    Returns
    -------
    overall_quality_parameters_dict : dict
        Overall quality parameters dictionary. The keys are the following.
        - For each scanline 'i', with 'i' in [0,`n_scanlines`-1], the key 'scanline_i' is present in the dictionary.
          `overall_quality_parameters_dict['scanline_i']` is itself a dictionary, containing the quality parameters of that 
          scanline. In particular, the keys are the following.
          * 'R_min' and 'R_min_grade'
          * 'SC' and 'SC_grade'
          * 'EC_min' and 'EC_min_grade'
          * 'MODULATION' and 'MODULATION_grade'
          * 'DEFECT' and 'DEFECT_grade'
          * 'scanline_min_grade' and 'scanline_min_value'
        - 'OVERALL_NUMERICAL_VALUE' and 'OVERALL_SYMBOL_GRADE' which represent, respectively, the overall numerical value and
          the overall symbol grade of the barcode.
        - 'R_min_MEAN' and 'R_min_MEAN_grade' which represent, respectively, the mean 'R_min' value across the scanlines and 
          the corresponding grade.
        - 'SC_MEAN' and 'SC_MEAN_grade' which represent, respectively, the mean 'SC' value across the scanlines and the 
          corresponding grade.
        - 'EC_min_MEAN' and 'EC_min_MEAN_grade' which represent, respectively, the mean 'EC_min' value across the scanlines 
          and the corresponding grade.
        - 'MODULATION_MEAN' and 'MODULATION_MEAN_grade' which represent, respectively, the mean 'MODULATION' value across the
          scanlines and the corresponding grade.
        - 'DEFECT_MEAN' and 'DEFECT_MEAN_grade' which represent, respectively, the mean 'DEFECT' value across the
          scanlines and the corresponding grade.

    """
    overall_quality_parameters_dict = {}

    n_scanlines = len(quality_parameters_dict_scanline_list)
    for i in range(n_scanlines):
        quality_parameters_dict_currentScanline = quality_parameters_dict_scanline_list[i]
        current_scanline_dict = {
            'R_min': quality_parameters_dict_currentScanline['R_min'],
            'R_min_grade': compute_Rmin_grade(quality_parameters_dict_currentScanline['R_min'], 
                                              quality_parameters_dict_currentScanline['R_max']),
            'SC': quality_parameters_dict_currentScanline['SC'],
            'SC_grade': compute_SC_grade(quality_parameters_dict_currentScanline['SC']),
            'EC_min': quality_parameters_dict_currentScanline['modulation_dict']['EC_min'],
            'EC_min_grade': compute_ECmin_grade(quality_parameters_dict_currentScanline['modulation_dict']['EC_min']),
            'MODULATION': quality_parameters_dict_currentScanline['modulation_dict']['MODULATION'],
            'MODULATION_grade': compute_MODULATION_grade(quality_parameters_dict_currentScanline['modulation_dict']['MODULATION']),
            'DEFECT': quality_parameters_dict_currentScanline['defect_dict']['DEFECT'],
            'DEFECT_grade': compute_DEFECT_grade( quality_parameters_dict_currentScanline['defect_dict']['DEFECT'])
        }

        scanline_min_grade = max([current_scanline_dict['R_min_grade'], current_scanline_dict['SC_grade'], 
                                  current_scanline_dict['EC_min_grade'], current_scanline_dict['MODULATION_grade'], 
                                  current_scanline_dict['DEFECT_grade']])
        scanline_min_value = compute_scanline_min_value(scanline_min_grade)
        current_scanline_dict['scanline_min_grade'] = scanline_min_grade
        current_scanline_dict['scanline_min_value'] = scanline_min_value

        overall_quality_parameters_dict[f'scanline_{i}'] = current_scanline_dict

    OVERALL_NUMERICAL_VALUE = np.mean([overall_quality_parameters_dict[f'scanline_{i}']['scanline_min_value'] 
                                       for i in range(n_scanlines)])
    OVERALL_SYMBOL_GRADE = compute_overall_symbol_grade(OVERALL_NUMERICAL_VALUE)
    overall_quality_parameters_dict['OVERALL_NUMERICAL_VALUE'] = OVERALL_NUMERICAL_VALUE
    overall_quality_parameters_dict['OVERALL_SYMBOL_GRADE'] = OVERALL_SYMBOL_GRADE

    R_min_MEAN = np.mean([quality_parameters_dict_scanline_list[i]['R_min'] for i in range(n_scanlines)])
    R_max_MEAN = np.mean([quality_parameters_dict_scanline_list[i]['R_max'] for i in range(n_scanlines)])
    R_min_MEAN_grade = compute_Rmin_grade(R_min_MEAN, R_max_MEAN)
    overall_quality_parameters_dict['R_min_MEAN'] = R_min_MEAN
    overall_quality_parameters_dict['R_min_MEAN_grade'] = R_min_MEAN_grade

    SC_MEAN = np.mean([quality_parameters_dict_scanline_list[i]['SC'] for i in range(n_scanlines)])
    SC_MEAN_grade = compute_SC_grade(SC_MEAN)
    overall_quality_parameters_dict['SC_MEAN'] = SC_MEAN
    overall_quality_parameters_dict['SC_MEAN_grade'] = SC_MEAN_grade

    EC_min_MEAN = np.mean([quality_parameters_dict_scanline_list[i]['modulation_dict']['EC_min'] for i in range(n_scanlines)])
    EC_min_MEAN_grade = compute_ECmin_grade(EC_min_MEAN)
    MODULATION_MEAN = np.mean([quality_parameters_dict_scanline_list[i]['modulation_dict']['MODULATION'] for i in range(n_scanlines)])
    MODULATION_MEAN_grade = compute_MODULATION_grade(MODULATION_MEAN)
    overall_quality_parameters_dict['EC_min_MEAN'] = EC_min_MEAN
    overall_quality_parameters_dict['EC_min_MEAN_grade'] = EC_min_MEAN_grade
    overall_quality_parameters_dict['MODULATION_MEAN'] = MODULATION_MEAN
    overall_quality_parameters_dict['MODULATION_MEAN_grade'] = MODULATION_MEAN_grade

    DEFECT_MEAN = np.mean([quality_parameters_dict_scanline_list[i]['defect_dict']['DEFECT'] for i in range(n_scanlines)])
    DEFECT_MEAN_grade = compute_DEFECT_grade(DEFECT_MEAN)
    overall_quality_parameters_dict['DEFECT_MEAN'] = DEFECT_MEAN
    overall_quality_parameters_dict['DEFECT_MEAN_grade'] = DEFECT_MEAN_grade
    
    return overall_quality_parameters_dict
    

def compute_Rmin_grade(R_min, R_max):
    """Compute the symbolic grade associated to R_min, given its numerical value.

    The symbolic grade is in {'A','F'}

    Parameters
    ----------
    R_min : float
    R_max : float

    Returns
    -------
    R_min_grade : str

    """
    if R_min<=0.5*R_max:
        R_min_grade = 'A'
    else:
        R_min_grade = 'F' 
    return  R_min_grade

def compute_SC_grade(SC):
    """Compute the symbolic grade associated to SC, given its numerical value.

    The symbolic grade is in {'A','B','C','D','F'}

    Parameters
    ----------
    SC : float

    Returns
    -------
    SC_grade : str

    """
    if SC>=70:
        SC_grade = 'A'
    elif SC>=55:
        SC_grade = 'B'
    elif SC>=40:
        SC_grade = 'C'
    elif SC>=20:
        SC_grade = 'D'
    else:
        SC_grade = 'F'
    return SC_grade 

def compute_ECmin_grade(EC_min):
    """Compute the symbolic grade associated to EC_min, given its numerical value.

    The symbolic grade is in {'A','F'}

    Parameters
    ----------
    EC_min : float

    Returns
    -------
    EC_min_grade : str

    """
    if EC_min>=15:
        EC_min_grade = 'A'
    else:
        EC_min_grade = 'F'
    return EC_min_grade

def compute_MODULATION_grade(MODULATION):
    """Compute the symbolic grade associated to MODULATION, given its numerical value.

    The symbolic grade is in {'A','B','C','D','F'}

    Parameters
    ----------
    MODULATION : float

    Returns
    -------
    MODULATION_grade : str

    """
    if MODULATION>=0.70:
        MODULATION_grade = 'A'
    elif MODULATION>=0.60:
        MODULATION_grade = 'B'
    elif MODULATION>=0.50:
        MODULATION_grade = 'C'
    elif MODULATION>=0.40:
        MODULATION_grade = 'D'
    else:
        MODULATION_grade = 'F'
    return MODULATION_grade 

def compute_DEFECT_grade(DEFECT):
    """Compute the symbolic grade associated to DEFECT, given its numerical value.

    The symbolic grade is in {'A','B','C','D','F'}

    Parameters
    ----------
    DEFECT : float

    Returns
    -------
    DEFECT_grade : str

    """
    if DEFECT<=0.15:
        DEFECT_grade = 'A'
    elif DEFECT<=0.20:
        DEFECT_grade = 'B'
    elif DEFECT<=0.25:
        DEFECT_grade = 'C'
    elif DEFECT<=0.30:
        DEFECT_grade = 'D'
    else:
        DEFECT_grade = 'F'
    return DEFECT_grade

def compute_scanline_min_value(scanline_min_grade):
    """Compute the numerical value associated to the scanline, given its symbolic grade.

    The symbolic grade is in {'A','B','C','D','F'}, The numerical value is an integer in [0..4].

    Parameters
    ----------
    scanline_min_grade : str

    Returns
    -------
    scanline_min_value : int

    """
    if scanline_min_grade=='A':
        scanline_min_value = 4
    elif scanline_min_grade=='B':
        scanline_min_value = 3
    elif scanline_min_grade=='C':
        scanline_min_value = 2
    elif scanline_min_grade=='D':
        scanline_min_value = 1
    else: 
        scanline_min_value = 0
    return scanline_min_value

def compute_overall_symbol_grade(overall_numerical_value):
    """Compute the overall symbolic grade associated to the barcode, given its numerical value.

    The symbolic grade is in {'A','B','C','D','F'}, The numerical value is a real number in [0,4].

    Parameters
    ----------
    overall_numerical_value : float

    Returns
    -------
    overall_symbol_grade : str

    """
    if overall_numerical_value<=4.0 and overall_numerical_value>=3.5:
        overall_symbol_grade = 'A'
    elif overall_numerical_value<3.5 and overall_numerical_value>=2.5:
        overall_symbol_grade = 'B'
    elif overall_numerical_value<2.5 and overall_numerical_value>=1.5:
        overall_symbol_grade = 'C'
    elif overall_numerical_value<1.5 and overall_numerical_value>=0.5:
        overall_symbol_grade = 'D'
    else:
        overall_symbol_grade = 'F'
    return overall_symbol_grade


        
    
