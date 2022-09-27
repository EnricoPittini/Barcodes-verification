import time
import os
import cv2

from detect_boundingBox import detect_boundingBox
from rotate_boundingBox import rotate_boundingBox
from refine_ROIimage import refine_ROIimage
from compute_quality_parameters import compute_quality_parameters
from build_output_file import build_output_file

""" verbose_timing : bool, optional
        Whether to print the timing information about the process or not, by default False"""



def verify_barcode(image_path, use_same_threshold=False, compute_barcode_structure_algorithm=1, n_scanlines=10, 
                   outlier_detection_level=0.02, visualization_dict=None, verbose_timing=False, create_output_file=False,
                   output_file_name=None, output_file_type='excel', output_folder_path='./out'):

    image = cv2.imread(image_path) 
    image_name = '.'.join(os.path.basename(image_path).split('.')[:-1])
    print(image_name)   

    visualization_dict = _populate_visualization_dict(visualization_dict) 

    start_time = time.time()

    bb_points_sorted, bb_width, bb_height, threshold = detect_boundingBox(image, 
                                        visualize_bounding_box=visualization_dict['visualize_originalImage_boundingBox'])
    end_detectBB_time = time.time()

    image_rot, bb_points_sorted_rot, roi_image, angle = rotate_boundingBox(image, bb_points_sorted, bb_width, bb_height, 
                                    fix_horizontalBars_case=True, 
                                    visualize_rotatedImage_boundingBox=visualization_dict['visualize_rotatedImage_boundingBox'])
    end_rotateBB_time = time.time()

    roi_image_ref, bb_points_sorted_rot_ref, barcode_structure_dict = refine_ROIimage(roi_image, image_rot, 
                                    bb_points_sorted_rot, 
                                    compute_barcode_structure_algorithm=compute_barcode_structure_algorithm, 
                                    threshold=threshold if use_same_threshold else None,
                                    fix_wrongBar_case=True, 
                                    outlier_detection_level=outlier_detection_level, 
                                    visualize_barcode_structure=visualization_dict['visualize_barcode_structure'], 
                                    visualize_refinedRoi_withQuantities=visualization_dict['visualize_refinedRoi_withQuantities'], 
                                    visualize_refinedRoi=visualization_dict['visualize_refinedRoi'])
    end_refineROI_time = time.time()

    overall_quality_parameters_dict = compute_quality_parameters(roi_image_ref, n_scanlines=n_scanlines, 
                                    visualize_scanlines_onRoiImage=visualization_dict['visualize_scanlines_onRoiImage'], 
                                    visualize_scanlines_qualityParameters=visualization_dict['visualize_scanlines_qualityParameters'])
    end_computeQualityParameters_time = time.time()

    if verbose_timing:
        print('TIMING INFORMATION')
        print('\tDetect bounding box:', end_detectBB_time-start_time)
        print('\tRotate bounding box:', end_rotateBB_time-end_detectBB_time)
        print('\tRefine ROI image:', end_refineROI_time-end_rotateBB_time)
        print('\tCompute quality parameters:', end_computeQualityParameters_time-end_refineROI_time)
        print()

    detection_dict = {
        'bb_points_sorted': bb_points_sorted,
        'bb_width': bb_width,
        'bb_height': bb_height,
        'threshold': threshold
    }

    rotation_dict = {
        'image_rot': image_rot,
        'bb_points_sorted_rot': bb_points_sorted_rot,
        'roi_image': roi_image,
        'angle': angle
    }

    refinement_dict = {
        'roi_image_ref': roi_image_ref,
        'bb_points_sorted_rot_ref': bb_points_sorted_rot_ref,
        'barcode_structure_dict': barcode_structure_dict
    }

    if create_output_file:
        build_output_file(detection_dict, rotation_dict, refinement_dict, overall_quality_parameters_dict, image_name, n_scanlines=n_scanlines,
                           output_file_name=output_file_name, output_file_type=output_file_type, output_folder_path=output_folder_path)

    return detection_dict, rotation_dict, refinement_dict, overall_quality_parameters_dict



def _populate_visualization_dict(visualization_dict):
    if visualization_dict=='all':
        visualization_dict = {}
        visualization_dict['visualize_originalImage_boundingBox'] = True
        visualization_dict['visualize_rotatedImage_boundingBox'] = True
        visualization_dict['visualize_barcode_structure'] = True
        visualization_dict['visualize_refinedRoi_withQuantities'] = True
        visualization_dict['visualize_refinedRoi'] = True
        visualization_dict['visualize_scanlines_onRoiImage'] = True
        visualization_dict['visualize_scanlines_qualityParameters'] = True

    if visualization_dict is None:
        visualization_dict = {}
    if 'visualize_originalImage_boundingBox' not in visualization_dict:
        visualization_dict['visualize_originalImage_boundingBox'] = False
    if 'visualize_rotatedImage_boundingBox' not in visualization_dict:
        visualization_dict['visualize_rotatedImage_boundingBox'] = False
    if 'visualize_barcode_structure' not in visualization_dict:
        visualization_dict['visualize_barcode_structure'] = False
    if 'visualize_refinedRoi_withQuantities' not in visualization_dict:
        visualization_dict['visualize_refinedRoi_withQuantities'] = False
    if 'visualize_refinedRoi' not in visualization_dict:
        visualization_dict['visualize_refinedRoi'] = False 
    if 'visualize_scanlines_onRoiImage' not in visualization_dict:
        visualization_dict['visualize_scanlines_onRoiImage'] = False 
    if 'visualize_scanlines_qualityParameters' not in visualization_dict:
        visualization_dict['visualize_scanlines_qualityParameters'] = False

    return visualization_dict
