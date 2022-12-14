import os
import json
import pandas as pd
import numpy as np
import copy



def build_output_file(detection_dict, rotation_dict, refinement_dict, overall_quality_parameters_dict, image_name, n_scanlines=10,
        output_file_name=None, output_file_type='excel 1', output_folder_path='./out'):
    """Build the output file containing the information computed from the process of the verification of the barcode, using the
    `verify_barcode` function.

    Parameters
    ----------
    detection_dict : dict
        Dictionary containing the information and results obtained from the first operation, i.e. from the detection of the 
        barcode. 
        See the `verify_barcode` and `detect_boundingBox` functions.
    rotation_dict : dict
        Dictionary containing the information and results obtained from the second operation, i.e. from the rotation of the 
        barcode. 
        See the `verify_barcode` and `rotate_boundingBox` functions.
    refinement_dict : dict
        Dictionary containing the information and results obtained from the third operation, i.e. from the refinement of the 
        ROI image. 
        See the `verify_barcode` and `refine_boundingBox` functions.
    overall_quality_parameters_dict : dict
        Dictionary containing the computed quality parameters. 
        See the `verify_barcode` and `compute_quality_parameters` functions.
    image_name : str
        Name of the input image
    n_scanlines : int, optional
        Number of scanlines used for computing the quality parameters, by default 10
    output_file_name : _type_, optional
        Name of the optional output file, by default None.
        If None, the output file name is 'output <NAME OF THE IMAGE>'
    output_file_type : str, optional
        Type of the optional output file, by default 'excel 1'.
        Three possibilites:
        - 'excel 1': excel file containing the information recquested in the description of the project
        - 'excel 2': richier excel file, containing all the computed quantities, except for the images(i.e. rotated input 
                     image, ROI image and refined ROI image.)
        - 'json': exhaustive json file containing all the computed information, except for the images (i.e. rotated input 
                  image, ROI image and refined ROI image.)
    output_folder_path : str, optional
        Path of the folder for the optional output file, by default './out'

    Raises
    ------
    ValueError
        If a wrong `output_file_type` is given

    Notes
    -----
    Notes about the structure of the output file.

    Regarding the json file, it has the exact same structure of the dictionaries `detection_dict`, `rotation_dict`, 
    `refinement_dict`, `overall_quality_parameters_dict`. The only difference is that it does not contain the images (i.e. 
    rotated input image, ROI image and refined ROI image.)

    Regarding the excel output files, the information is structured into different sheets.

    Output file type 'excel 1'. The sheets are the following.
    - Global quantities
        It contains the following information.
        * Name of the image
        * Bounding box coordinates
        * Centre of the bounding box
        * Angle of the rotation
        * X dimension (i.e. minimum width of a barcode bar)
        * Height of the barcode (i.e. minimum height of a barcode bar)
        * Overall symbolic grade of the barcode
    - Bars/spaces widths
        For each bar and space, its width is reported, in units by X dimension.
        It is a list, where the first element refers to the first bar, and the last element to the last bar.
        Basically, sequence of bars/spaces from left to right.
    - Scanlines quality parameters
        For each scanline, its quality parameters are reported. Namely:
        * R_min and R_min_grade
        * SC and SC_grade
        * EC_min and EC_min_grade
        * MODULATION and MODULATION_grade
        * DEFECT and DEFECT_grade
        * Symbolic grade and numerical value

    Output file type 'excel 2'. The sheets are the following.
    - General quantities
        It contains the following information.
        * Name of the image
        * Bounding box coordinates
        * Angle of the rotation
        * Rotated bounding box coordinates        
        * Rotated and refined bounding box coordinates        
    - Barcode global structure
        It contains the following information. It is important to remark that these quantities are computed with respect to
        the ROI image (not refined): this is the reference system.
        * X dimension (i.e. minimum width of a barcode bar)
        * Height of the barcode (i.e. minimum height of a barcode bar)
        * Horizontal coordinate of the first pixel of the first barcode bar
        * Horizontal coordinate of the last pixel of the last barcode bar
        * Minimum half height of a bar from the middle of the ROI image upward.
        * Minimum half height of a bar from the middle of the ROI image downward.
    - Bars local structure
        For each barcode bar, from the leftmost to the rightmost, it contains the following information. It is important to 
        remark that these quantities are computed with respect to the ROI image (not refined): this is the reference system.
        * Horixontal coordinate of the first pixel of that bar.
        * Width of that bar.
        * Height of that bar.
        * Half height from the middle of the ROI image upward of that bar.
        * Half height from the middle of the ROI image downward of that bar.
    - Global quality parameters
        It contains the following information.
        * OVERALL_NUMERICAL_VALUE and OVERALL_SYMBOL_GRADE
        * R_min_MEAN and R_min_MEAN_grade (mean computed over all the scanlines)
        * SC_MEAN and SC_MEAN_grade (mean computed over all the scanlines)
        * EC_min_MEAN and EC_min_MEAN_grade (mean computed over all the scanlines)
        * MODULATION_MEAN and MODULATION_MEAN_grade (mean computed over all the scanlines)
        * DEFECT_MEAN and DEFECT_MEAN_grade (mean computed over all the scanlines)
    - Scanlines quality parameters
        For each scanline, its quality parameters are reported. Namely:
        * R_min and R_min_grade
        * SC and SC_grade
        * EC_min and EC_min_grade
        * MODULATION and MODULATION_grade
        * DEFECT and DEFECT_grade
        * Symbolic grade and numerical value

    """

    if output_file_name is None:
        output_file_name = 'output ' + image_name
    output_file_name += '.xlsx' if 'excel' in output_file_type else '.json'
    output_path = os.path.join(output_folder_path, output_file_name)
    
    if output_file_type=='excel 1':
        global_quantities_df = pd.DataFrame({
            'image_name' : image_name,
            'bb_points_sorted': [detection_dict['bb_points_sorted']],
            'bb_centre' : [_compute_boundingBox_centre(detection_dict['bb_points_sorted'])],
            'angle' : [rotation_dict['angle']],
            'X' : [refinement_dict['barcode_structure_dict']['X']],
            'height' : [refinement_dict['barcode_structure_dict']['height']],
            'OVERALL_SYMBOL_GRADE' : [overall_quality_parameters_dict['OVERALL_SYMBOL_GRADE']]
        })
        
        barcode_structure_df = pd.DataFrame(_compute_barsSpaces_widths(refinement_dict['barcode_structure_dict']))
        barcode_structure_df.index.name = 'bars_spaces'

        scanlines_qualityParameters_df = pd.DataFrame([overall_quality_parameters_dict[f'scanline_{i}'] 
                                                      for i in range(n_scanlines)])
        scanlines_qualityParameters_df.index.name = 'scanlines'

        with pd.ExcelWriter(output_path) as writer:
            global_quantities_df.to_excel(writer, sheet_name='Global quantities')
            barcode_structure_df.to_excel(writer, sheet_name='Bars_spaces widths')
            scanlines_qualityParameters_df.to_excel(writer, sheet_name='Scanlines quality parameters')

    elif output_file_type=='excel 2':
        general_quantities_df = pd.DataFrame({
            'image_name' : image_name,
            'bb_points_sorted': [detection_dict['bb_points_sorted']],
            'angle' : [rotation_dict['angle']],
            'bb_points_sorted_rot': [rotation_dict['bb_points_sorted_rot']],
            'bb_points_sorted_rot_ref': [refinement_dict['bb_points_sorted_rot_ref']]
        })
        
        barcode_global_structure_df = pd.DataFrame({
            'X' : [refinement_dict['barcode_structure_dict']['X']],
            'height': [refinement_dict['barcode_structure_dict']['height']],
            'first_bar_x': [refinement_dict['barcode_structure_dict']['first_bar_x']],
            'last_bar_x': [refinement_dict['barcode_structure_dict']['last_bar_x']],
            'min_half_height_up': [refinement_dict['barcode_structure_dict']['min_half_height_up']],
            'min_half_height_down': [refinement_dict['barcode_structure_dict']['min_half_height_down']]
        })

        bars_local_structure_df = pd.DataFrame(_compute_bars_local_structure_dict(refinement_dict['barcode_structure_dict']))
        bars_local_structure_df.index.name = 'bars'

        global_qualityParameters_df = pd.DataFrame({
            'OVERALL_NUMERICAL_VALUE' : [overall_quality_parameters_dict['OVERALL_NUMERICAL_VALUE']],
            'OVERALL_SYMBOL_GRADE': [overall_quality_parameters_dict['OVERALL_SYMBOL_GRADE']],
            'R_min_MEAN': [overall_quality_parameters_dict['R_min_MEAN']],
            'R_min_MEAN_grade': [overall_quality_parameters_dict['R_min_MEAN_grade']],
            'SC_MEAN': [overall_quality_parameters_dict['SC_MEAN']],
            'SC_MEAN_grade': [overall_quality_parameters_dict['SC_MEAN_grade']],
            'EC_min_MEAN': [overall_quality_parameters_dict['EC_min_MEAN']],
            'EC_min_MEAN_grade': [overall_quality_parameters_dict['EC_min_MEAN_grade']],
            'MODULATION_MEAN': [overall_quality_parameters_dict['MODULATION_MEAN']],
            'MODULATION_MEAN_grade': [overall_quality_parameters_dict['MODULATION_MEAN_grade']],
            'DEFECT_MEAN': [overall_quality_parameters_dict['DEFECT_MEAN']],
            'DEFECT_MEAN_grade': [overall_quality_parameters_dict['DEFECT_MEAN_grade']],
        })

        scanlines_qualityParameters_df = pd.DataFrame([overall_quality_parameters_dict[f'scanline_{i}'] 
                                                      for i in range(n_scanlines)])
        scanlines_qualityParameters_df.index.name = 'scanlines'

        with pd.ExcelWriter(output_path) as writer:
            general_quantities_df.to_excel(writer, sheet_name='General quantities')
            barcode_global_structure_df.to_excel(writer, sheet_name='Barcode global structure')
            bars_local_structure_df.to_excel(writer, sheet_name='Bars local structure')
            global_qualityParameters_df.to_excel(writer, sheet_name='Global quality parameters')
            scanlines_qualityParameters_df.to_excel(writer, sheet_name='Scanlines quality parameters')

    elif output_file_type=='json':
        output_dict = {
            'image_name': image_name,
            'detection_dict': detection_dict,
            'rotation_dict': rotation_dict,
            'refinement_dict': refinement_dict,
            'overall_quality_parameters_dict': overall_quality_parameters_dict
        } 
        del output_dict['rotation_dict']['image_rot']
        del output_dict['rotation_dict']['roi_image']
        del output_dict['refinement_dict']['roi_image_ref']
        output_dict = _transform_dict_for_json(output_dict)
        with open(output_path, 'w') as out_file:
            json.dump(output_dict, out_file, indent=6)

    else:
        raise ValueError(f'Unsopported output file type {output_file_type}')




def _compute_boundingBox_centre(bb_points_sorted):
    return np.mean(bb_points_sorted, axis=0)

def _compute_barsSpaces_widths(barcode_structure_dict):
    X = barcode_structure_dict['X']

    barsSpaces_widths = []
    current_space_start = None
    for bar_start, bar_width in zip(barcode_structure_dict['bars_start'], barcode_structure_dict['bars_width']):
        barsSpaces_widths.append(bar_width/X)
        if current_space_start is not None:
            current_space_width = bar_start-current_space_start
            barsSpaces_widths.append(current_space_width/X)
        current_space_start = bar_start + bar_width

    spacesBars_flags = [True if i%2==0 else False for i in range(len(barsSpaces_widths))]

    return {
        'bars_spaces_widths': barsSpaces_widths,
        'bars_spaces_flags': spacesBars_flags
    }

def _transform_dict_for_json(d):
    d_res = copy.deepcopy(d)
    for key in d:
        if type(d[key])==np.ndarray:
            #print(f'ARRAY {key}')
            d_res[key] = d[key].tolist()
        elif type(d[key])==dict:
            #print(f'DICT {key}')
            d_res[key] = _transform_dict_for_json(d[key])
    return d_res

def _compute_bars_local_structure_dict(barcode_structure_dict):
    bars_local_structure_dict = copy.deepcopy(barcode_structure_dict)
    del bars_local_structure_dict['X']
    del bars_local_structure_dict['height']
    del bars_local_structure_dict['first_bar_x']
    del bars_local_structure_dict['last_bar_x']
    del bars_local_structure_dict['min_half_height_up']
    del bars_local_structure_dict['min_half_height_down']
    return bars_local_structure_dict
