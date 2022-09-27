from detect_boundingBox import detect_boundingBox
from rotate_boundingBox import rotate_boundingBox
from refine_roiImage import refine_roiImage
from compute_quality_parameters import compute_quality_parameters



def verify_barcode(image, use_same_threshold=False, compute_barcode_structure_algorithm=1, n_scanlines=10, 
                   outlier_detection_level=0.02, visualization_dict=None, verbose_timing=False):

    visualization_dict = _populate_visualization_dict(visualization_dict) 

    bb_points_sorted, bb_width, bb_height, threshold = detect_boundingBox(image, 
                                                                          visualize_bounding_box=visualization_dict['visualize_originalImage_boundingBox'])

    image_rot, bb_points_sorted_rot, roi_image, angle = rotate_boundingBox(image, bb_points_sorted, bb_width, bb_height, 
                                                                           fix_horizontalBars_case=True, 
                                                                           visualize_rotatedImage_boundingBox=visualization_dict['visualize_rotatedImage_boundingBox'])

    roi_image_ref, bb_points_sorted_rot_ref, barcode_structure_dict = refine_roiImage(roi_image, image_rot, 
                                                                            bb_points_sorted_rot, 
                                                                            compute_barcode_structure_algorithm=compute_barcode_structure_algorithm, 
                                                                            threshold=threshold if use_same_threshold else None,
                                                                            fix_wrongBar_case=True, 
                                                                            outlier_detection_level=outlier_detection_level, 
                                                                            visualize_barcode_structure=visualization_dict['visualize_barcode_structure'], 
                                                                            visualize_refinedRoi_withQuantities=visualization_dict['visualize_refined_roi_withQuantities'], 
                                                                            visualize_refinedRoi=visualization_dict['visualize_refined_roi'])


    overall_quality_parameters_dict = compute_quality_parameters(roi_image_ref, n_scanlines=n_scanlines, 
                                                                visualize_scanlines_onRoiImage=visualization_dict['visualize_scanlines_onRoiImage'], 
                                                                visualize_scanlines_qualityParameters=visualization_dict['visualize_scanlines_qualityParameters'], 
                                                                verbose_timing=False)



def _populate_visualization_dict(visualization_dict):
    if visualization_dict=='all':
        visualization_dict = {}
        visualization_dict['visualize_originalImage_boundingBox'] = True
        visualization_dict['visualize_rotatedImage_boundingBox'] = True
        visualization_dict['visualize_rotatedImageBoundingBox_after_fixedHorizontalBarsProblem'] = True
        visualization_dict['visualize_barcode_structure'] = True
        visualization_dict['visualize_barcodeStructure_after_fixedWrongBarProblem'] = True
        visualization_dict['visualize_refined_roi_withQuantities'] = True
        visualization_dict['visualize_refined_roi'] = True
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
    if 'visualize_refined_roi_withQuantities' not in visualization_dict:
        visualization_dict['visualize_refined_roi_withQuantities'] = False
    if 'visualize_refined_roi' not in visualization_dict:
        visualization_dict['visualize_refined_roi'] = False 
    if 'visualize_scanlines_onRoiImage' not in visualization_dict:
        visualization_dict['visualize_scanlines_onRoiImage'] = False 
    if 'visualize_scanlines_qualityParameters' not in visualization_dict:
        visualization_dict['visualize_scanlines_qualityParameters'] = False

    return visualization_dict