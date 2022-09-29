import time
import os
import cv2

from detect_boundingBox import detect_boundingBox
from rotate_boundingBox import rotate_boundingBox
from refine_ROIimage import refine_ROIimage
from compute_quality_parameters import compute_quality_parameters

from build_output_file import build_output_file



def verify_barcode(image_path, use_same_threshold=False, compute_barcode_structure_algorithm=1, n_scanlines=10, 
                   outlier_detection_level=0.02, visualization_dict=None, verbose_timing=False, create_output_file=False,
                   output_file_name=None, output_file_type='excel 1', output_folder_path='./out'):
    """Verify the printing quality of the barcode contained in the given input image.

    This process consists in four subsequent operations.
    1) DETECT THE BOUNDING BOX
       The bounding box surrounding the barcode in the input image is detected.
       See the `detect_boundingBox` function.
    2) ROTATE THE BOUNDING BOX
       The image and the bounding box are rotated such that the barcode bars are now perfectly vertical.
       From this operation, the ROI image is computed, which is the sub-image containing the barcode, with the bars perfectly
       vertical. Basically, the ROI image is the rotated image cropped around the rotated barcode.
       Remark: the ROI image is gray-scale.
       See the `rotate_boundingBox` function.
    3) REFINE THE ROI IMAGE
       The ROI image is refined, according to a certain standard.
       - Along the width, the refined ROI image is such that there are exactly 10*X pixels before the first barcode bar and
         after the last barcode bar, where X is the minimum width of a bar.
       - Along the height, the ROI image is refined in order to perfectly fit the bar with smallest height. Basically, 
         the height of the refined ROI image is equal to the minimum height of a barcode bar. 
       In order to perform this refinement, the precise and complete structure of the barcode is computed: every dimension 
       about each bar is computed.
       See the `refine_ROIimage` function.
    4) COMPUTE THE QUALITY PARAMETERS
       Finally, the quality parameters of the barcode are computed, on the refined ROI image.
       For computing the quality parameters, `n_scanlines` equally spaced horizontal lines are considered in the given ROI image.
       The quality parameters are computed one each scanline, and they are the following.
       - Minimum reflectance, i.e. R_min.
       - Symbol Contrast, i.e. SC. For computing it, also the maximum reflectance, i.e. R_max, is taken into account.
       - Minimum Edge Contrast, i.e. EC_min.
       - MODULATION.
       - DEFECT. For computing it, also the maximum Element Reflectance Non-uniformity, i.e. ERN_max, is taken into account.
       For each of these parameters, a numerical value is computed, and a symbolic grade is assigned, between 'A' and 'F'. 
       In addition, a symbolic grade and a numerical value are assigned to the whole scanline.
       Finally, an overall symbolic grade and an overall numerical value are assigned to the whole barcode.
       See the `compute_quality_parameters` function.

    For more information, see the report of this project.

    For each of these four steps, a dictionary containing the information regarding that step is returned in output.

    Optionally, an output file can be created, containing the information computed from this process.

    Parameters
    ----------
    image_path : str
        Path to the input image
    use_same_threshold : bool, optional
        Whether to use or not the same threshold in the two thresholding operations, by default False.
        The first thresholding operator is performed for detecting the barcode, while the second for computing the barcode 
        structure. 
        If False, the Otsu's algorithm is used for computing the threshold in both the thresholding operations.
        If True, the Otsu's algorithm is used only in the first thresholding operation, and then the same threshold is used
        for the second operation.
    compute_barcode_structure_algorithm : int, optional
         Algorithm for computing the barcode structure, by default 1
    n_scanlines : int, optional
        Number of scanlines used for computing the quality parameters, by default 10
    outlier_detection_level : float, optional
        Level for detecting a possible outlier bar (i.e. wrong bar) in the computed barcode structure, by default 0.02.
        The bigger, the easier is that a bar is considered as an outlier, i.e. wrong.
    visualization_dict : _type_, optional
        Dictionary containing the information regarding the desired plots, by default None.
        If 'all', all the plots are made.
        The possible keys are the following, containing boolean values.
        - visualize_original_image_boundingBox : whether to visualize or not the original input image with the detected 
                                                 bounding box surrounding the barcode. (Operation 1)
        - visualize_rotated_image_boundingBox : whether to visualize or not the rotated input image with the rotated bounding
                                                 box. (Operation 2)
        - visualize_barcode_structure : whether to visualize the barcode structure or not.  (Operation 3)
        - visualize_refinedRoi_withQuantities : whether to visualize or not the refined ROI image with the computed quantities.
                                                (Operation 3)
        - visualize_refinedRoi : whether to visualize or not the refined ROI image. (Operation 3)
        - visualize_scanlines_onRoiImage : whether to visualize or not the ROI image with the scanlines. (Operation 4)
        - visualize_scanlines_qualityParameters : whether to visualize or not the scan reflectance profile of each scanline,
                                                  with also the computed quality parameters. (Operation 4)
    verbose_timing : bool, optional
        Whether to print or not the information regarding the time of each one of the four operations, by default False
    create_output_file : bool, optional
        Whether to create an output file containing the computed information or not, by default False
    output_file_name : _type_, optional
        Name of the optional output file, by default None.
        If None, the output file name is 'output <NAME OF THE IMAGE>'
    output_file_type : str, optional
        Type of the optional output file, by default 'excel 1'.
        Three possibilites:
        - 'excel 1': excel file containing the information recquested in the description of the project
        - 'excel 2': richier excel file, containing almost all the computed quantities.
        - 'json': exhaustive json file containing all the computed information. WARNING: the creation of the file can take
                  a lot of time.
    output_folder_path : str, optional
        Path of the folder for the optional output file, by default './out'

    Returns
    -------
    detection_dict : dict
        Dictionary containing the information and results obtained from the first operation, i.e. from the detection of the 
        barcode. The keys are the following.
        - bb_points_sorted : np.array
            Array 4x2, containing the coordinates of the four bounding box verteces. 
            The points are ordered in the following way: up-left, up-right, bottom-left, bottom-right. This is our standard 
            ordering.
        - bb_width : int
            Width of the bounding box.
        - bb_height : int
            Height of the bounding box.
        - threshold : float
            Threshold used in the thresholding operation for detecting the bounding box.
            It has been computed using the Otsu's algorithm.
    rotation_dict : dict
        Dictionary containing the information and results obtained from the second operation, i.e. from the rotation of the 
        barcode. The keys are the following.
        - image_rot : np.array
            Rotated input image.
        - bb_points_sorted_rot : int
            The rotated bounding box. More precisely, array 4x2 containing the coordinates of the 4 verteces of the rotated 
            bounding box. These four verteces are ordered according to our standard ordering.
        - roi_image : int
            Rotated input image cropped around the rotated bounding box. Basically, sub-image containing only the barcode (i.e. 
            the ROI), whose bars are perfectly vertical.
            It is important to point out that this ROI image is in gray-scale, not colored.
        - angle : float
            Orientation of the original bounding box with in the original image.
            Basically, angle of the original bounding box with respect to the original horixontal axis.
    refinement_dict : dict
        Dictionary containing the information and results obtained from the third operation, i.e. from the refinement of the 
        ROI image. The keys are the following.
        'roi_image_ref': np.array
            ROI image after the refinement. Refinement along the width and height, according to the X dimension and to the 
            minimum height of a bar.
            Basically, it is the rotated image cropped around the rotated and refined bounding box.
        'bb_points_sorted_rot_ref': np.array
            Bounding box surrounding the barcode, rotated and refined. Array 4x2, containing the coordinates of the four 
            bounding box points. 
            These four verteces are ordered according to our standard ordering.
        'barcode_structure_dict': dict
            Dictionary containing the information about the barcode structure. The keys are the following.
            - X : minimum width of a bar.
            - min_half_height_up : minimum half height of a bar from the middle of the ROI image upward.
            - min_half_height_down : minimum half height of a bar from the middle of the ROI image downward.
            - height : height of the barcode. It is equal to min_half_height_up+min_half_height_down+1.
            - first_bar_x : horixontal coordinate of the first pixel of the first bar (i.e. left-most bar).
            - last_bar_x : horixontal coordinate of the last pixel of the last bar (i.e. right-most bar).
            - bars_start : list contaning, for each bar, the horixontal coordinate of the first pixel of that bar.
            - bars_width : list contaning, for each bar, the width of that bar.
            - bars_height : list contaning, for each bar, the height of that bar.
            - bars_halfHeightUp :  list contaning, for each bar, the half height from the middle of the ROI image upward of that 
                                bar.
            - bars_halfHeightDown :  list contaning, for each bar, the half height from the middle of the ROI image downward of that 
                                bar.
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
          * 'scanline_grade' and 'scanline_value' which represent, respectively, the symbolic grade and the numerical value 
            of the scanline.
          * 'n_edges', representing the number of edges in the scanline. 
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

    image = cv2.imread(image_path) 
    image_name = '.'.join(os.path.basename(image_path).split('.')[:-1])

    # Populate the visualization dict, handling also the particular cases
    visualization_dict = _populate_visualization_dict(visualization_dict) 

    start_time = time.time()

    # 1) DETECT THE BOUNDING BOX
    bb_points_sorted, bb_width, bb_height, threshold = detect_boundingBox(image, 
                                        visualize_bounding_box=visualization_dict['visualize_originalImage_boundingBox'])
    end_detectBB_time = time.time()

    # 2) ROTATE THE BOUNDING BOX
    image_rot, bb_points_sorted_rot, roi_image, angle = rotate_boundingBox(image, bb_points_sorted, bb_width, bb_height, 
                                    fix_horizontalBars_case=True, 
                                    visualize_rotatedImage_boundingBox=visualization_dict['visualize_rotatedImage_boundingBox'])
    end_rotateBB_time = time.time()

    # 3) REFINE THE ROI IMAGE
    # And compute the barcode structure
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

    # 4) COMPUTE THE QUALITY PARAMETERS
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

    # Create and return tthe dictionaries containing the information regarding each of the four operations

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

    if create_output_file:  # Create output file
        build_output_file(detection_dict, rotation_dict, refinement_dict, overall_quality_parameters_dict, image_name, n_scanlines=n_scanlines,
                           output_file_name=output_file_name, output_file_type=output_file_type, output_folder_path=output_folder_path)

    return detection_dict, rotation_dict, refinement_dict, overall_quality_parameters_dict



def _populate_visualization_dict(visualization_dict):
    """Populate the visualization dict, handling also the particular cases"""
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
