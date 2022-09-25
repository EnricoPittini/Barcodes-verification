import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils
import math
import matplotlib.patches as patches
import time

import scipy


import importlib  

module = importlib.import_module('0) APIs')
detect_and_refine_boundingBox = module.detect_and_refine_boundingBox


def compute_defect(scanline, SC, edges_indices, n_elements, last_pixel_edge, peaks_indices, valleys_indices):
    ERN_list = []
    peaksValleys_elements_list = []
    for i in range(n_elements):
        if i==0:
            element_first_index = 0
            element_last_index = edges_indices[0]-1
        elif i<n_elements-1:
            element_first_index = edges_indices[i-1]
            element_last_index = edges_indices[i]-1
        else:
            element_first_index = edges_indices[n_elements-2]
            element_last_index = scanline.shape[0]-1
            if last_pixel_edge:
                element_last_index -= 1
        peaks_indices_within_element = peaks_indices[np.logical_and(peaks_indices>=element_first_index,peaks_indices<=element_last_index)]
        valleys_indices_within_element = valleys_indices[np.logical_and(valleys_indices>=element_first_index,valleys_indices<=element_last_index)]
        if peaks_indices_within_element.size==0 or valleys_indices_within_element.size==0:
            #ERN_list.append(0)
            #peaksValleys_elements_list.append((None,None))
            continue
        peaks_within_element = scanline[peaks_indices_within_element]
        valleys_within_element = scanline[valleys_indices_within_element]
        ERN_current = peaks_within_element.max()-valleys_within_element.min()
        ERN_list.append(ERN_current)
        ERNcurrent_peak_index = peaks_indices_within_element[np.argmax(peaks_within_element)]
        ERNcurrent_valley_index = valleys_indices_within_element[np.argmin(valleys_within_element)]
        peaksValleys_elements_list.append((ERNcurrent_peak_index,ERNcurrent_valley_index))
    
    ERN_max = max(ERN_list)
    DEFECT = ERN_max / SC
    
    ERNmax_peak_index, ERNmax_valley_index = peaksValleys_elements_list[np.argmax(ERN_list)]
    
    return DEFECT, ERNmax_peak_index, ERNmax_valley_index


def compute_modulation(scanline, SC, GB, edges_indices, n_elements, last_pixel_edge, peaks_indices, valleys_indices):    
    EC_list = []
    peaksValleys_adjacentElements_list = []

    for i in range(n_elements-1):
        #print(i)
        if i==0:
            element1_first_index = 0
            element1_last_index = edges_indices[0]-1
            element2_first_index = edges_indices[0]
            element2_last_index = edges_indices[1]-1
        else:
            element1_first_index = edges_indices[i-1]
            element1_last_index = edges_indices[i]-1
            element2_first_index = edges_indices[i]
            if i!=n_elements-2:
                element2_last_index = edges_indices[i+1]-1
            else:
                element2_last_index = scanline.shape[0]-1
                if last_pixel_edge:
                    element2_last_index -= 1
        element1_bar_flag = scanline[element1_first_index]<GB
        element2_bar_flag = scanline[element2_first_index]<GB
        #print(element1_bar_flag,element2_bar_flag)
        if (element1_bar_flag and element2_bar_flag) or (not element1_bar_flag and not element2_bar_flag):
            raise ValueError('Discordant adjacent elements')
        if element1_bar_flag and not element2_bar_flag:
            barElement_first_index, barElement_last_index = element1_first_index, element1_last_index
            spaceElement_first_index, spaceElement_last_index = element2_first_index, element2_last_index
        else:
            barElement_first_index, barElement_last_index = element2_first_index, element2_last_index
            spaceElement_first_index, spaceElement_last_index = element1_first_index, element1_last_index
        #print(barElement_first_index, barElement_last_index)
        #print(spaceElement_first_index, spaceElement_last_index)

        valleys_indices_within_barElement = valleys_indices[np.logical_and(valleys_indices>=barElement_first_index,valleys_indices<=barElement_last_index)]
        #peaks_indices_within_barElement = peaks_indices[np.logical_and(peaks_indices>=barElement_first_index,peaks_indices<=barElement_last_index)]
        #indicesOfInterest_within_barElement = valleys_indices_within_barElement#np.concatenate([valleys_indices_within_barElement,peaks_indices_within_barElement])
        if valleys_indices_within_barElement.size==0:
                print('WARNING: bar element without vallyes')
                continue
        barElement_max_index = valleys_indices_within_barElement[np.argmax(scanline[valleys_indices_within_barElement])]
        barElement_max_value = scanline[barElement_max_index]#np.max(scanline[indicesOfInterest_within_barElement])
        #print(barElement_max_value)

        #print(spaceElement_first_index,spaceElement_last_index)
        peaks_indices_within_spaceElement = peaks_indices[np.logical_and(peaks_indices>=spaceElement_first_index,peaks_indices<=spaceElement_last_index)]
        #valleys_indices_within_spaceElement = valleys_indices[np.logical_and(valleys_indices>=spaceElement_first_index,valleys_indices<=spaceElement_last_index)]
        #indicesOfInterest_within_spaceElement = peaks_indices_within_spaceElement#np.concatenate([valleys_indices_within_spaceElement,peaks_indices_within_spaceElement])
        if peaks_indices_within_spaceElement.size==0:
                print('WARNING: space element without peaks')
                continue
        spaceElement_min_index = peaks_indices_within_spaceElement[np.argmin(scanline[peaks_indices_within_spaceElement])]
        spaceElement_min_value = scanline[spaceElement_min_index]#np.min(scanline[indicesOfInterest_within_spaceElement])
        #print(spaceElement_min_value)

        #print(spaceElement_min_value-barElement_max_value)

        EC_current = spaceElement_min_value-barElement_max_value
        EC_list.append(EC_current)
        #print((spaceElement_min_index,barElement_max_index))
        peaksValleys_adjacentElements_list.append((spaceElement_min_index, barElement_max_index))
        
    EC_min = min(EC_list)
    MODULATION = EC_min / SC
    ECmin_spaceElement_minIndex, ECmin_barElement_maxIndex = peaksValleys_adjacentElements_list[np.argmin(EC_list)]
    
    return EC_min, MODULATION, ECmin_spaceElement_minIndex, ECmin_barElement_maxIndex 


def compute_quality_parameters_scanline(scanline):
    R_min = scanline.min()
    R_max = scanline.max()    
    SC = R_max - R_min    
    GB = R_min + SC/2
    
    mask = (scanline<GB).astype(int)
    mask_1 = mask[1:]
    mask_2 = mask[:-1]
    edges_mask = np.abs(mask_1 - mask_2).astype(bool)
    edges_mask = np.append([False], edges_mask)
    last_pixel_edge = edges_mask[-1]
    edges_mask[-1] = False
    edges_indices = np.indices(mask.shape)[0][edges_mask]
    n_elements = len(edges_indices)+1
    
    peaks_indices = scipy.signal.find_peaks(scanline)[0]
    valleys_indices = scipy.signal.find_peaks(100-scanline)[0]
    
    EC_min, MODULATION, ECmin_spaceElement_minIndex, ECmin_barElement_maxIndex  = compute_modulation(scanline, SC, GB, edges_indices, 
                                                                                          n_elements, last_pixel_edge, 
                                                                                          peaks_indices, valleys_indices)

    DEFECT, ERNmax_peak_index, ERNmax_valley_index = compute_defect(scanline, SC, edges_indices, n_elements, last_pixel_edge, 
                                                       peaks_indices, valleys_indices)

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


def compute_quality_parameters(image, n_scanlines=10, visualize_scanlines_onRoiImage=False, 
                               visualize_scanlines_qualityParameters=False, verbose_timing=False): 
    
    roi_image, bb_points_sorted_rot, bb_width, bb_height = detect_and_refine_boundingBox(image, 
                            use_same_threshold=False, compute_barcode_structure_algorithm=1, verbose_timing=False,
                            outlier_detection_level=0.02, visualization_dict=None)

    start_time = time.time()
    
    scanlines_indices = np.linspace(start=0, stop=bb_height, num=n_scanlines+2, dtype=int)[1:-1]    
    roi_image_norm = 100*(roi_image/255)
    
    if visualize_scanlines_onRoiImage:
        roi_image_tmp = roi_image.copy()
        roi_image_tmp = cv2.cvtColor(roi_image_tmp, cv2.COLOR_GRAY2RGB)
        roi_image_tmp[scanlines_indices,:,:] = np.array([255,0,0])
        plt.figure()
        plt.imshow(roi_image_tmp, 'gray') 
        plt.title('ROI image, with the 10 scanlines')
    
    if visualize_scanlines_qualityParameters:
        fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(16,16), squeeze=True)
    
    quality_parameters_dict_scanline_list = []
    for i, scanline_index in enumerate(scanlines_indices):
        scanline = roi_image_norm[scanline_index, :]

        quality_parameters_dict_scanline = compute_quality_parameters_scanline(scanline)
        quality_parameters_dict_scanline_list.append(quality_parameters_dict_scanline)

        if visualize_scanlines_qualityParameters:
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

    overall_quality_parameters_dict = compute_overall_quality_parameters_dict(quality_parameters_dict_scanline_list)

    end_time = time.time()
    if verbose_timing:
        print('Computing quality parameters time:', end_time-start_time)
        print()

    return overall_quality_parameters_dict


def compute_Rmin_grade(R_min, R_max):
    if R_min<=0.5*R_max:
        R_min_grade = 'A'
    else:
        R_min_grade = 'F' 
    return  R_min_grade

def compute_SC_grade(SC):
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
    if EC_min>=15:
        EC_min_grade = 'A'
    else:
        EC_min_grade = 'F'
    return EC_min_grade

def compute_MODULATION_grade(MODULATION):
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

def compute_overall_quality_parameters_dict(quality_parameters_dict_scanline_list):
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
    OVERALL_NUMERICAL_GRADE = compute_overall_symbol_grade(OVERALL_NUMERICAL_VALUE)
    overall_quality_parameters_dict['OVERALL_NUMERICAL_VALUE'] = OVERALL_NUMERICAL_VALUE
    overall_quality_parameters_dict['OVERALL_NUMERICAL_GRADE'] = OVERALL_NUMERICAL_GRADE

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
        
    
