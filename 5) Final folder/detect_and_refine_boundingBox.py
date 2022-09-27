import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils
import math
import time


# return detection_results_dict, rotation_results_dict, barcode_structure_dict, refinement_results_dict,



############################ OVERALL FUNCTION 
def detect_and_refine_boundingBox(image, use_same_threshold=False, compute_barcode_structure_algorithm=1, verbose_timing=False,
                            outlier_detection_level=0.02, visualization_dict=None):
    """Detect and refine the bounding box enclosing the barcode in the given image and refine it. 

    Parameters
    ----------
    image : np.array
        Input image
    use_same_threshold : bool, optional
        Whether to use or not the same threshold in the two thresholding operations, by default False.
        The first thresholding operator is performed for detecting the barcode, while the second for computing the barcode 
        structure. The Otsu's algorithm is used.
    compute_barcode_structure_algorithm : int, optional
        Algorithm for computing the barcode structure, by default 1
    verbose_timing : bool, optional
        Whether to print information about the solving time or not, by default False
    outlier_detection_level : float, optional
        Level for pruning the outlier bar, by default 0.02
    visualization_dict : dict, optional
        Dictionary containing the information regarding the desired plots, by default None.
        If 'all', all the plots are made.
        The keys are:  'visualize_original_image_boundingBox', 'visualize_rotated_image_boundingBox',
        'visualize_barcode_structure', 'visualize_refined_roi_withQuantities', 'visualize_refined_roi'.

    """
    if visualization_dict=='all':
        del visualization_dict
        visualization_dict = {}
        visualization_dict['visualize_original_image_boundingBox'] = True
        visualization_dict['visualize_rotated_image_boundingBox'] = True
        visualization_dict['visualize_rotatedImageBoundingBox_after_fixedHorizontalBarsProblem'] = True
        visualization_dict['visualize_barcode_structure'] = True
        visualization_dict['visualize_barcodeStructure_after_fixedWrongBarProblem'] = True
        visualization_dict['visualize_refined_roi_withQuantities'] = True
        visualization_dict['visualize_refined_roi'] = True

    if visualization_dict is None:
        visualization_dict = {}
    if 'visualize_original_image_boundingBox' not in visualization_dict:
        visualization_dict['visualize_original_image_boundingBox'] = False
    if 'visualize_rotated_image_boundingBox' not in visualization_dict:
        visualization_dict['visualize_rotated_image_boundingBox'] = False
    if 'visualize_rotatedImageBoundingBox_after_fixedHorizontalBarsProblem' not in visualization_dict:
        visualization_dict['visualize_rotatedImageBoundingBox_after_fixedHorizontalBarsProblem'] = False
    if 'visualize_barcode_structure' not in visualization_dict:
        visualization_dict['visualize_barcode_structure'] = False
    if 'visualize_barcodeStructure_after_fixedWrongBarProblem' not in visualization_dict:
        visualization_dict['visualize_barcodeStructure_after_fixedWrongBarProblem'] = False
    if 'visualize_refined_roi_withQuantities' not in visualization_dict:
        visualization_dict['visualize_refined_roi_withQuantities'] = False
    if 'visualize_refined_roi' not in visualization_dict:
        visualization_dict['visualize_refined_roi'] = False 

    start_time = time.time()      

    # 1) DETECT ROI
    bb_points_sorted, bb_width, bb_height, threshold = detect_boundingBox(image, 
                                            visualize_bounding_box=visualization_dict['visualize_original_image_boundingBox'])
    detecting_bb_end_time = time.time()

    # 2) ROTATE IMAGE AND BOUNDING BOX
    image_rot, bb_points_sorted_rot = rotate_image_boundingBox(image, bb_points_sorted, bb_width, bb_height, 
                                            visualize_rot_image_bb=visualization_dict['visualize_rotated_image_boundingBox'])
    rotating_image_bb_end_time = time.time()

    # 3) FIX HORIZONTAL BARS CASE
    image_rot, bb_points_sorted_rot, bb_width, bb_height = fix_horizontalBars_case(image_rot, bb_points_sorted_rot, bb_width, 
                bb_height, 
                visualize_fixed_image_bb=visualization_dict['visualize_rotatedImageBoundingBox_after_fixedHorizontalBarsProblem'])
    fixing_horizontalBarsCase_end_time = time.time()

    # ROI image
    gray_rot = cv2.cvtColor(image_rot, cv2.COLOR_BGR2GRAY)  # Gray rotated image
    roi_image = gray_rot[int(bb_points_sorted_rot[0][1]):int(bb_points_sorted_rot[0][1]+bb_height), 
                             int(bb_points_sorted_rot[0][0]):int(bb_points_sorted_rot[0][0]+bb_width)]

    # 4) COMPUTE BARCODE STRUCTURE
    barcode_structure_dict = compute_barcode_structure(roi_image, bb_width, bb_height,
                                                       algorithm=compute_barcode_structure_algorithm, 
                                                       threshold=threshold if use_same_threshold else None,
                                                       verbose=False, visualize_refined_bb=False,
                                                       visualize_barcode_structure=visualization_dict['visualize_barcode_structure'])
    computing_barcode_structure_end_time = time.time()

    # 5) FIND AND FIX WRONG BAR (if any)
    find_and_fix_wrong_bar(roi_image, bb_height, barcode_structure_dict, level=outlier_detection_level, 
                           visualize_fixed_barcode_structure=visualization_dict['visualize_barcodeStructure_after_fixedWrongBarProblem'])
    fixing_wrong_bar_end_time = time.time()
  
    # Compute barcode quantities, and inject them into the barcode structure dict
    barcode_structure_dict['first_bar_x'] =  min(barcode_structure_dict['bars_start']) 
    barcode_structure_dict['last_bar_x'] =  max([s+w for s,w in zip(barcode_structure_dict['bars_start'],
                                                                    barcode_structure_dict['bars_width'])])-1 
    barcode_structure_dict['X'] =  min(barcode_structure_dict['bars_width'])  
    barcode_structure_dict['min_half_height_up'] =  min(barcode_structure_dict['bars_halfHeightUp'])   
    barcode_structure_dict['min_half_height_down'] =  min(barcode_structure_dict['bars_halfHeightDown'])
    
    # 6) REFINE THE BOUNDING BOX AND THE ROI IMAGE
    roi_image_ref, bb_points_sorted_rot_ref, bb_width_ref, bb_height_ref = refine_roi(gray_rot, bb_points_sorted_rot, 
                                                         bb_height, bb_width, barcode_structure_dict, 
                                                         visualize_refined_roi_withQuantities=visualization_dict['visualize_refined_roi_withQuantities'], 
                                                         visualize_refined_roi=visualization_dict['visualize_refined_roi'])
    refine_bb_roi_image_end_time = time.time()

    if verbose_timing:
        print('Computing roi time:', detecting_bb_end_time-start_time)
        print('Rotating image and bounding box time:', rotating_image_bb_end_time-detecting_bb_end_time)
        print('Fixing horizontal bars case time:', fixing_horizontalBarsCase_end_time-rotating_image_bb_end_time)
        print('Computing barcode structure time:', computing_barcode_structure_end_time-fixing_horizontalBarsCase_end_time)
        print('Fixing wrong bar case time:', fixing_wrong_bar_end_time-computing_barcode_structure_end_time)
        print('Refining roi image and bounding box time:', refine_bb_roi_image_end_time-fixing_wrong_bar_end_time)
        print()

    
    return roi_image_ref, bb_points_sorted_rot_ref, bb_width_ref, bb_height_ref


        


