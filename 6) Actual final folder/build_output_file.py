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
    detection_dict : _type_
        _description_
    rotation_dict : _type_
        _description_
    refinement_dict : _type_
        _description_
    overall_quality_parameters_dict : _type_
        _description_
    image_name : _type_
        _description_
    n_scanlines : int, optional
        _description_, by default 10
    output_file_name : _type_, optional
        _description_, by default None
    output_file_type : str, optional
        _description_, by default 'excel 1'
    output_folder_path : str, optional
        _description_, by default './out'

    Raises
    ------
    ValueError
        _description_
    """

    #output_folder_path = os.path.normpath(output_folder_path)

    if output_file_name is None:
        output_file_name = 'output ' + image_name
    output_file_name += '.xlsx' if 'excel' in output_file_type else '.json'
    output_path = os.path.join(output_folder_path, output_file_name)

    #print(output_path)
    
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
        
        #barcode_global_structure_df = pd.DataFrame(_compute_barsSpaces_widths(refinement_dict['barcode_structure_dict']))
        #barcode_global_structure_df.index.name = 'bars_spaces'
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
