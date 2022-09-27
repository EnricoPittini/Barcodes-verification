
import os
import json
import pandas as pd
import numpy as np
import copy


def _compute_boundingBox_centre(bb_points_sorted):
    return np.mean(bb_points_sorted, axis=0)

def _compute_barsSpaces_widths(barcode_structure_dict):
    X = barcode_structure_dict['X']

    barsSpaces_widths = []
    current_space_start = None
    for bar_start, bar_width in zip(barcode_structure_dict['bars_start'], barcode_structure_dict['bars_width']):
        barsSpaces_widths.append(bar_width)
        if current_space_start is not None:
            current_space_width = bar_start-current_space_start
            barsSpaces_widths.append(current_space_width)
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
            d_res['key'] = list(d[key])
        elif type(d[key])==dict:
            d_res['key'] = _transform_dict_for_json(d[key])
    return d_res



def build_output_file(detection_dict, rotation_dict, refinement_dict, overall_quality_parameters_dict, image_name, n_scanlines=10,
        output_file_name=None, output_file_type='excel', output_folder_path='./out'):

    #output_folder_path = os.path.normpath(output_folder_path)

    if output_file_name is None:
        output_file_name = 'output ' + image_name
    output_file_name += '.xlsx' if output_file_type=='excel' else '.json'
    output_path = os.path.join(output_folder_path, output_file_name)

    print(output_path)
    
    if output_file_type=='excel':
        global_quantities_df = pd.DataFrame({
            'image_name' : image_name,
            'bb_points_sorted': [detection_dict['bb_points_sorted']],
            'bb_centre' : [_compute_boundingBox_centre(detection_dict['bb_points_sorted'])],
            'angle' : [rotation_dict['angle']],
            'X' : [refinement_dict['barcode_structure_dict']['X']],
            'height' : [refinement_dict['barcode_structure_dict']['height']],
            'OVERALL_SYMBOL_GRADE' : [overall_quality_parameters_dict['OVERALL_SYMBOL_GRADE']]
        })
        
        """barcode_structure_df = pd.DataFrame({
            'bars_start': refinement_dict['barcode_structure_dict']['bars_start'],
            'bars_width': refinement_dict['barcode_structure_dict']['bars_width'],
            'bars_halfHeightUp': refinement_dict['barcode_structure_dict']['bars_halfHeightUp'],
            'bars_halfHeightDown': refinement_dict['barcode_structure_dict']['bars_halfHeightDown']
        })"""
        barcode_structure_df = pd.DataFrame(_compute_barsSpaces_widths(refinement_dict['barcode_structure_dict']))
        barcode_structure_df.index.name = 'bars_spaces'

        scanlines_qualityParameters_df = pd.DataFrame([overall_quality_parameters_dict[f'scanline_{i}'] 
                                                      for i in range(n_scanlines)])
        scanlines_qualityParameters_df.index.name = 'scanlines'

        with pd.ExcelWriter(output_path) as writer:
            global_quantities_df.to_excel(writer, sheet_name='Global quantities')
            barcode_structure_df.to_excel(writer, sheet_name='Bars_spaces widths')
            scanlines_qualityParameters_df.to_excel(writer, sheet_name='Scanline quality parameters')

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
            json.dump(output_dict, out_file, indent=4)

    else:
        raise ValueError(f'Unsopported output file type {output_file_type}')
