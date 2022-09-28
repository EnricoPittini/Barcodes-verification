import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import time


def refine_ROIimage(roi_image, image_rot, bb_points_sorted_rot, compute_barcode_structure_algorithm=1, threshold=None,
                       fix_wrongBar_case=True, outlier_detection_level=0.02, visualize_barcode_structure=False, 
                       visualize_refinedRoi_withQuantities=False, visualize_refinedRoi=False):
    """Refine the ROI image containing the barcode.

    The ROI image has been obtained by cropping the input image around the bounding box containing the barcode, after they 
    have been both rotated in order to have the barcode bars perfectly vertical.

    This refinement of the ROI image is done both along the width and along the height.
    - Along the width, the refined ROI image is such that there are exactly 10*X pixels before the first barcode bar and
      after the last barcode bar, where X is the minimum width of a bar.
    - Along the height, the ROI image is refined in order to perfectly fit the barcode bar with smallest height. Basically, 
      the height of the refined ROI image is equal to the minimum height of a barcode bar.

    In order to perform this refinement, the precise and complete structure of the barcode is computed. Namely, the 
    following quantities are computed. It is very important to underline that these quantities are computed with respect to
    the current ROI image, and not the refined one: the reference system is the current ROI image.
    - X : minimum width of a bar.
    - min_half_height_up : minimum half height of a bar from the middle of the ROI image upward.
    - min_half_height_down : minimum half height of a bar from the middle of the ROI image downward.
    - height : height of the barcode. It is equal to min_half_height_up+min_half_height_down+1.
    - first_bar_x : horixontal coordinate of the first pixel of the first bar (i.e. left-most bar).
    - last_bar_x : horixontal coordinate of the last pixel of the last bar (i.e. right-most bar).
    - Finally, the following quantities are computed for each barcode bar.
        * Horixontal coordinate of the first pixel.
        * Width.
        * Height.
        * Half height from the middle of the ROI image upward.
        * Half height from the middle of the ROI image downward.
    So, both global quantities (i.e. related to the whole barcode) and local quantities (i.e. related to the single bars) are 
    computed: exhaustive barcode structure.

    Finally, it is worth observing that, when computing the barcode structure, it can happen that a wrong bar is detected. 
    Basically, something which is not a barcode bar is detected as a bar: obviously, this negatively impact the computed 
    barcode structure.
    This situation can be fixed, by pruning from the barcode structure bars which are outliers with respect to the other bars.
    For minimizing the possibility to wrongly prune an actual true bar, at most one bar is pruned from the barcode structure.
    At most one wrong bar.
    In addition, a very cautious approach has been followed: see the `_fix_wrongBar_case` function.

    Parameters
    ----------
    roi_image : np.array
        ROI image containing only the barcode, whose bars are perfectly vertical.
        Basically, input image cropped around the bounding box surrounding the barcode, after they have been both rotated in 
        order to have the barcode bars perfectly vertical.
        It is important to point out that this ROI image is in gray-scale, not colored.
    image_rot : np.array
        Input image containing the barcode, rotated in order to have the bounding box perfectly aligned with the image axes
    bb_points_sorted_rot : _type_
        Bounding box surrounding the barcode, rotated in order to be perfectly aligned with the image axes.
        Array 4x2, containing the coordinates of the four bounding box points. 
        The points are ordered in the following way: up-left, up-right, bottom-left, bottom-right.
    compute_barcode_structure_algorithm : int, optional
        Algorithm to use for computing the barcode structure, by default 1.
        Choices among 1,2,3,4.
    threshold : float, optional
        Threshold to use for thresholding the ROI image, by default None.
        If None is given, the Otsu's algorithm is used for finding the threshold.
        The thresholding operation is used for computing the barcode structure.
    fix_wrongBar_case : bool, optional
        Whether to fix the wrong bar case or not, by default True.
        If True, the possible wrong bar case is handleled: the wrong bar is pruned from the barcode structure.
        If False, the possible wrong bar case is not handleled: the barcode structure is not touched.
    outlier_detection_level : float, optional
        Level for detecting the outlier bar, by default 0.02.
        The bigger, the easier is that a bar is considered as an outlier, i.e. wrong.
    visualize_barcode_structure : bool, optional
        Whether to visualize the barcode structure or not, by default False
    visualize_refinedRoi_withQuantities : bool, optional
        Whether to visualize or not the refined ROI image with the computed quantities, by default False
    visualize_refinedRoi : bool, optional
        Whether to visualize or not the refined ROI image, by default False

    Returns
    -------
    roi_image_ref : np.array
        ROI image after the refinement.
        Basically, input image cropped around the bounding box, after they have been both rotated in order to have the 
        barcode bars perfectly vertical and after that the bounding box has been refined.
        It is important to point out that this ROI image is in gray-scale, not colored.
    bb_points_sorted_rot_ref : np.array
        Bounding box surrounding the barcode, rotated in order to be perfectly aligned with the image axes and refined along
        the width and height, according to the X dimension and to the minimum height of a bar.
        Array 4x2, containing the coordinates of the four bounding box points. 
        The points are ordered in the following way: up-left, up-right, bottom-left, bottom-right.
    barcode_structure_dict : dict
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

    """

    # Compute the structure of the barcode. Actually, compute only the "local" structure, namely quantities related to the 
    # individual bars of the barcode.
    barcode_localStructure_dict = _compute_barcode_structure(roi_image, threshold=threshold, 
                                                            algorithm=compute_barcode_structure_algorithm)

    # Fix the possible wrong bar case, by pruning the wrong bar from the barcode structure
    if fix_wrongBar_case:
        _fix_wrong_bar(barcode_localStructure_dict, level=outlier_detection_level) 

    # Compute the global barcode quantities, and create the final barcode structure dict
    barcode_structure_dict = barcode_localStructure_dict.copy()
    barcode_structure_dict['first_bar_x'] =  min(barcode_structure_dict['bars_start']) 
    barcode_structure_dict['last_bar_x'] =  max([s+w for s,w in zip(barcode_structure_dict['bars_start'],
                                                                    barcode_structure_dict['bars_width'])])-1 
    barcode_structure_dict['X'] =  min(barcode_structure_dict['bars_width'])  
    barcode_structure_dict['min_half_height_up'] =  min(barcode_structure_dict['bars_halfHeightUp'])   
    barcode_structure_dict['min_half_height_down'] =  min(barcode_structure_dict['bars_halfHeightDown'])
    barcode_structure_dict['height'] = barcode_structure_dict['min_half_height_up'] + barcode_structure_dict['min_half_height_down'] + 1

    if visualize_barcode_structure:  # Visualize the barcode structure
        plot_barcode_Structure(roi_image, barcode_structure_dict)

    # Refine the ROI image, and the bounding box coordinates
    roi_image_ref, bb_points_sorted_rot_ref = _refine_roi(roi_image, image_rot, bb_points_sorted_rot, barcode_structure_dict, 
                                                         visualize_refinedRoi_withQuantities=visualize_refinedRoi_withQuantities, 
                                                         visualize_refinedRoi=visualize_refinedRoi)

    return roi_image_ref, bb_points_sorted_rot_ref, barcode_structure_dict




def plot_barcode_Structure(roi_image, barcode_structure_dict):
    """Plot the barcode structure

    Parameters
    ----------
    roi_image : np.array
        ROI image containing the barcode (not refined yet)
    barcode_structure_dict : dict
        Dictionary containing the information about the barcode structure. 
        Actually, it is enough to contain only the "local" structure, namely quantities related to the individual bars of the
        barcode. Namely, keys 'bars_start', 'bars_width', 'bars_halfHeightUp', 'bars_halfHeightDown'.

    """
    bb_height = roi_image.shape[0]
    half_height = math.ceil(bb_height/2)

    bars_start, bars_width, bars_halfHeightUp, bars_halfHeightDown = (barcode_structure_dict['bars_start'], 
                                                                     barcode_structure_dict['bars_width'],
                                                                     barcode_structure_dict['bars_halfHeightUp'], 
                                                                     barcode_structure_dict['bars_halfHeightDown'])

    plt.figure(figsize=(10,10))
    roi_image_show = roi_image.copy()
    roi_image_show = cv2.cvtColor(roi_image_show, cv2.COLOR_GRAY2RGB ) 
    n_bars = len(bars_start)
    for b in range(n_bars):
        roi_image_show[[half_height-bars_halfHeightUp[b]-1,half_height+bars_halfHeightDown[b]-1],
                        bars_start[b]:bars_start[b]+bars_width[b],:] = np.array([255,0,0])
        roi_image_show[half_height-bars_halfHeightUp[b]-1:half_height+bars_halfHeightDown[b]-1+1,
                       [bars_start[b],bars_start[b]+bars_width[b]-1],:] = np.array([255,0,0])

    plt.imshow(roi_image_show)
    plt.title('Exhaustive barcode structure')
    plt.show() 



def _compute_barcode_structure(roi_image, threshold=None, algorithm=1):
    """Compute the complete barcode structure.

    Actually, it computes only the "local" structure, namely quantities related to the individual bars of the barcode. 
    Namely, it computes:
    - the starting pixel of each bar;
    - the width of each bar;
    - the half height up of each bar;
    - the half height down of each bar.

    Parameters
    ----------
    roi_image : np.array
        The original image cropped around the ROI (i.e. the barcode)
    threshold : int, optional
        Threshold to use for thresholding the ROI image, by default None.
        If None is given the Otsu's algorithm is used for finding the threshold.
    algorithm : int, optional
        Algorithm to use for computing the barcode structure, by default 1.
        Choices among 1,2,3,4.

    Returns
    -------
    barcode_localStructure_dict : dict
        Dictionary containing the barcode local structure. The keys are the following.
        - bars_start : list contaning, for each bar, the horixontal coordinate of the first pixel of that bar.
        - bars_width : list contaning, for each bar, the width of that bar.
        - bars_height : list contaning, for each bar, the height of that bar.
        - bars_halfHeightUp :  list contaning, for each bar, the half height from the middle of the ROI image upward of that 
                               bar.
        - bars_halfHeightDown :  list contaning, for each bar, the half height from the middle of the ROI image downward of that 
                               bar.

    Raises
    ------
    ValueError
        If a wrong algorithm index is given

    """

    # Theìreshold the ROI image, either by the given threshold or by using Otsu's
    if threshold is None:
        threshold ,ROI_thresh = cv2.threshold(roi_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        threshold, ROI_thresh = cv2.threshold(roi_image, threshold,255,cv2.THRESH_BINARY)

    # Fix the algorithm chosen by the user
    if algorithm==1:
        algorithm_function = _algorithm1
    elif algorithm==2:
        algorithm_function = _algorithm2
    elif algorithm==3:
        algorithm_function = _algorithm3
    elif algorithm==4:
        algorithm_function = _algorithm4
    else:
        raise ValueError(f'Invalid algorithm {algorithm}')

    # Apply the algorith for computing the barcode local structure
    bars_start, bars_width, bars_halfHeightUp, bars_halfHeightDown = algorithm_function(ROI_thresh)  

    # Create the dictionary
    barcode_localStructure_dict = {}
    barcode_localStructure_dict['bars_start'] =  bars_start 
    barcode_localStructure_dict['bars_width'] =  bars_width   
    barcode_localStructure_dict['bars_height'] = [bars_halfHeightUp[i]+bars_halfHeightDown[i]+1 for i in range(len(bars_halfHeightUp))] 
    barcode_localStructure_dict['bars_halfHeightUp'] =  bars_halfHeightUp   
    barcode_localStructure_dict['bars_halfHeightDown'] =  bars_halfHeightDown  
    
    return barcode_localStructure_dict



def _find_outliers(v, level=0.02):
    """Find the most outlier value of the given vector. 

    The approach is the following.
    - First of all, the outliers are detected. If no outlier is present, return None.
    - Then, the most outlier value is returned.

    Parameters
    ----------
    v : np.array
        Monodimensional vector
    level : float, optional
        Level for deciding if a value is an outlier or not, by default 0.02.
        The bigger, the easier is that a value of `v` is considered as an outlier.

    Returns
    -------
    float or None
        Element of `v` which is the most outlier value. If no outlier value is present, None is returned.

    """
    q1, q3 = tuple(np.quantile(v, [level,1-level]))
    IQR = q3-q1
    outliers_mask = np.logical_or(v>q3+IQR, v<q1-IQR)
    if v[outliers_mask].size==0:
        return None
    return np.argmax([v[i]-q3-IQR if (outliers_mask[i] and v[i]-q3-IQR>0) else abs(v[i]-q1+IQR) if outliers_mask[i] else 0  for i in range(len(v))])


def _fix_wrong_bar(barcode_localStructure_dict, level=0.02):
    """Prune a possible wrongly-detected bar from the current barcode structure.

    A wrongly-detected bar is an entity which has been detected has a barcode bar but it is not a bar.
    At most one bar is pruned, for minimizing the possibility of wrongly pruning actual true bars.

    A detected bar is recognized as wrong if it is an outlier with respect to the other bars.
    For minimizing the possibility of wrongly pruning an actual true bar, the following cautious algorithm has been followed.
    - Detect the outliers bars with respect to the bars height. 
      If no outlier bar with, end the algorithm (no bar is pruned).
      Otherwise, take into account the most outlier bar.
    - Do the same thing with respect to the bars area.
    - If the outlier bar with respect to the height and the outlier bar with respect to the area are different, end the 
      algorithm (no bar is pruned).
      Otherwise, the wrong bar is exactly that bar which is the most outlier bar both for the height and for the area.
    
    Important remark: if a wrongly-detected bar is recognized, it is removed from the barcode structure dictionary given in 
    input.

    Parameters
    ----------
    barcode_localStructure_dict : dict
        Dictionary containing the barcode local structure. The keys are the following.
        - bars_start : list contaning, for each bar, the horixontal coordinate of the first pixel of that bar.
        - bars_width : list contaning, for each bar, the width of that bar.
        - bars_height : list contaning, for each bar, the height of that bar.
        - bars_halfHeightUp :  list contaning, for each bar, the half height from the middle of the ROI image upward of that 
                               bar.
        - bars_halfHeightDown :  list contaning, for each bar, the half height from the middle of the ROI image downward of that 
                               bar.
    level : float, optional
        Level for detecting the outlier bar, by default 0.02.
        The bigger, the easier is that a bar is considered as an outlier, i.e. wrong.

    """
    bars_start, bars_width, bars_halfHeightUp, bars_halfHeightDown = (barcode_localStructure_dict['bars_start'], 
                                                                     barcode_localStructure_dict['bars_width'],
                                                                     barcode_localStructure_dict['bars_halfHeightUp'],
                                                                     barcode_localStructure_dict['bars_halfHeightDown'] ) 

    # Number of barcode bars
    n_bars = len(bars_start)
    # Height of each barcode bar
    bars_height = np.array([bars_halfHeightUp[i]+bars_halfHeightDown[i]+1 for i in range(n_bars)])
    # Area of each barcode bar
    bars_area = np.array([bars_height[i]+bars_width[i] for i in range(n_bars)])

    # Bar which is the most outlier bar with respect to the height
    wrong_bars_height_index = _find_outliers(bars_height, level=level)    
    
    # Bar which is the most outlier bar with respect to the area
    wrong_bars_area_index = _find_outliers(bars_area, level=level)
    
    # Either we don't have an height outlier bar or we don't have an area outlier bar or the height outlier bar is different
    # from the area outlier bar
    if wrong_bars_area_index is None or wrong_bars_height_index is None or wrong_bars_area_index!=wrong_bars_height_index:
        return None

    # The height outlier bar is the same of the area outlier bar: this is our outlier bar.
    # Namely, this is the wrongly detected bar.
    wrong_bar_index = wrong_bars_area_index

    # Delete the wrong bar from the given input dictionary
    if wrong_bar_index is not None:  
        del bars_start[wrong_bar_index]
        del bars_width[wrong_bar_index]
        del bars_halfHeightUp[wrong_bar_index]
        del bars_halfHeightDown[wrong_bar_index]
                      


def _refine_roi(roi_image, image_rot, bb_points_sorted_rot, barcode_structure_dict, visualize_refinedRoi_withQuantities=False,
               visualize_refinedRoi=False): 
    """Refine the ROI image containing the barcode, and the bounding box coordinates

    This refinement of the ROI image is done both along the width and along the height.
    - Along the width, the refined ROI image is such that there are exactly 10*X pixels before the first barcode bar and
      after the last barcode bar, where X is the minimum width of a bar.
    - Along the height, the ROI image is refined in order to perfectly fit the barcode bar with smallest height. Basically, 
      the height of the refined ROI image is equal to the minimum height of a barcode bar.

    Parameters
    ----------
    roi_image : np.array
        ROI image containing only the barcode, whose bars are perfectly vertical.
        Basically, input image cropped around the bounding box surrounding the barcode, after they have been both rotated in 
        order to have the barcode bars perfectly vertical.
        It is important to point out that this ROI image is in gray-scale, not colored.
    image_rot : np.array
        Input image containing the barcode, rotated in order to have the bounding box perfectly aligned with the image axes
    bb_points_sorted_rot : _type_
        Bounding box surrounding the barcode, rotated in order to be perfectly aligned with the image axes.
        Array 4x2, containing the coordinates of the four bounding box points. 
        The points are ordered in the following way: up-left, up-right, bottom-left, bottom-right.
    barcode_structure_dict : dict
        Dictionary containing the information about the barcode structure. In particular, we are interested in the global
        quantities, contained in the following keys.
        - X : minimum width of a bar.
        - min_half_height_up : minimum half height of a bar from the middle of the ROI image upward.
        - min_half_height_down : minimum half height of a bar from the middle of the ROI image downward.
        - height : height of the barcode. It is equal to min_half_height_up+min_half_height_down+1.
        - first_bar_x : horixontal coordinate of the first pixel of the first bar (i.e. left-most bar).
        - last_bar_x : horixontal coordinate of the last pixel of the last bar (i.e. right-most bar).
    visualize_refinedRoi_withQuantities : bool, optional
        Whether to visualize or not the refined ROI image with the computed quantities, by default False
    visualize_refinedRoi : bool, optional
        Whether to visualize or not the refined ROI image, by default False

    Returns
    ----------
    roi_image_ref : np.array
        ROI image after the refinement.
        Basically, input image cropped around the bounding box, after they have been both rotated in order to have the 
        barcode bars perfectly vertical and after that the bounding box has been refined.
        It is important to point out that this ROI image is in gray-scale, not colored.
    bb_points_sorted_rot_ref : np.array
        Bounding box surrounding the barcode, rotated in order to be perfectly aligned with the image axes and refined along
        the width and height, according to the X dimension and to the minimum height of a bar.
        Array 4x2, containing the coordinates of the four bounding box points. 
        The points are ordered in the following way: up-left, up-right, bottom-left, bottom-right.

    """
    # Gray-scale input image, rotated in order to have the barcode bars perfectly vertical
    gray_rot = cv2.cvtColor(image_rot, cv2.COLOR_BGR2GRAY)

    # Dimensions of the ROI image, i.e. dimensions of the bounding box surrounding the barcode
    bb_height, bb_width = roi_image.shape
                  
    # Global quantities about the barcode structure. They are used for the refinement process.
    first_bar_x, last_bar_x, X, min_half_height_up, min_half_height_down = (barcode_structure_dict['first_bar_x'], 
                                                                     barcode_structure_dict['last_bar_x'],
                                                                     barcode_structure_dict['X'],
                                                                     barcode_structure_dict['min_half_height_up'],
                                                                     barcode_structure_dict['min_half_height_down'] )

    half_height = math.ceil(bb_height/2)

    # Refinement of the bounding box.
    # Refinement only along the width, for visualization purposes.
    bb_points_sorted_rot_ref = bb_points_sorted_rot.copy()
    bb_points_sorted_rot_ref[[0,2],0] = bb_points_sorted_rot[[0,2],0] - (10*X-first_bar_x) 
    bb_points_sorted_rot_ref[[1,3],0] = bb_points_sorted_rot[[1,3],0] + (10*X-(bb_width-last_bar_x-1))

    if visualize_refinedRoi_withQuantities:  # Visualize the refined ROI with the computed (global) quantities
        roi_image_ref = gray_rot[int(bb_points_sorted_rot_ref[0][1]):int(bb_points_sorted_rot_ref[2][1])+1, 
                                    int(bb_points_sorted_rot_ref[0][0]):int(bb_points_sorted_rot_ref[1][0])+1].copy()
        bb_width_ref  = roi_image_ref.shape[1]
        plt.figure()
        plt.imshow(roi_image_ref, 'gray')
        plt.axvline(10*X, c='orange', label='10*X')
        plt.axvline(bb_width_ref-10*X-1, c='red', label='-10*X')
        plt.axhline(half_height-min_half_height_up-1, c='green', label='Min up height')
        plt.axhline(half_height+min_half_height_down-1, c='blue', label='Min down height')
        plt.title('Refined ROI, with the computed quantities')
        plt.legend()

    # Conclude the refinement of the bounding box and of the roi image. Refinement also along the height.
    bb_points_sorted_rot_ref[[0,1],1] = bb_points_sorted_rot[[0,1],1] + half_height - 1 - min_half_height_up 
    bb_points_sorted_rot_ref[[2,3],1] = bb_points_sorted_rot[[0,1],1] + half_height - 1 + min_half_height_down 
    roi_image_ref = gray_rot[int(bb_points_sorted_rot_ref[0][1]):int(bb_points_sorted_rot_ref[2][1])+1, 
                                int(bb_points_sorted_rot_ref[0][0]):int(bb_points_sorted_rot_ref[1][0])+1].copy()

    if visualize_refinedRoi:  # Visualize the refined ROI 
        plt.figure()
        plt.imshow(roi_image_ref, 'gray')
        plt.title('Refined ROI')

    return roi_image_ref, bb_points_sorted_rot_ref




##################### ALGORITHMS FOR COMPUTING THE BARCODE STRUCTURE
""" 
The following are the four different algorithms for computing the barcode structure: `_algorithm1`, `_algorithm2`, 
`_algorithm3`, `_algorithm4`.
They all have the same interface.

INPUTS
------
ROI_thresh : np.array
    Thresholded ROI imagy

OUTPUTS
------
- bars_start : list contaning, for each bar, the horixontal coordinate of the first pixel of that bar.
- bars_width : list contaning, for each bar, the width of that bar.
- bars_halfHeightUp :  list contaning, for each bar, the half height from the middle of the ROI image upward of that 
                       bar.
- bars_halfHeightDown :  list contaning, for each bar, the half height from the middle of the ROI image downward of that 
                         bar.

"""

def _algorithm1(ROI_thresh):
    bb_height, bb_width = ROI_thresh.shape

    # INIZIALIZATION
    half_height = math.ceil(bb_height/2)
    half_height_index = half_height-1

    bars_start = []
    bars_width = []
    bars_halfHeightUp = []
    bars_halfHeightDown = []

    i = 0  # Index for iterating over the pixels

    # CYCLE
    # We scan each pixel along the horizontal line in the exact middle of the ROI image
    while i<bb_width:

        # White pixel: we go to the next pixel
        if ROI_thresh[half_height_index, i]==255:
            i += 1
            continue

        # Black pixel
        # 'i' is the first pixel in this current barcode bar

        # Width of this current bar
        X_curr = 1    
        # Index representing the last pixel in this current bar
        i_end = i+1

        # We go right, till finding a white pixel.
        # In this way, we compute the width of this current bar.
        while ROI_thresh[half_height_index, i_end]==0:
            X_curr += 1
            i_end += 1

        # Now we search upward and downward along the vertical line 'i_med'.
        i_med = int((i+i_end)/2)
        # Index for goind upward.
        j_up = half_height_index-1
        # Index for goind downward.
        j_down = half_height_index+1
        # Half upward height of this current bar
        half_height_up_curr = 0
        # Half downard height of this current bar
        half_height_down_curr = 0

        # Cycle, in which we go upward and downard at the same time, for computing `half_height_up_curr` and 
        # `half_height_down_curr`
        up_reached = j_up<0 or (ROI_thresh[j_up, i_med]==255 and  ROI_thresh[j_up, i_med-1]==255 and ROI_thresh[j_up, i_med+1]==255)
        down_reached = j_down<0 or (ROI_thresh[j_down, i_med]==255 and  ROI_thresh[j_down, i_med-1]==255 and ROI_thresh[j_down, i_med+1]==255)
        while not up_reached or not down_reached:
            if not up_reached:
                j_up -= 1
                half_height_up_curr += 1
            if not down_reached:
                j_down += 1
                half_height_down_curr += 1
            up_reached = j_up<0 or (ROI_thresh[j_up, i_med]==255 and  ROI_thresh[j_up, i_med-1]==255 and ROI_thresh[j_up, i_med+1]==255)
            down_reached = j_down>=bb_height or (ROI_thresh[j_down, i_med]==255 and  ROI_thresh[j_down, i_med-1]==255 and ROI_thresh[j_down, i_med+1]==255)

        bars_start.append(i)
        bars_width.append(X_curr)
        bars_halfHeightUp.append(half_height_up_curr)
        bars_halfHeightDown.append(half_height_down_curr)

        # We update `i`: we pass to the white pixel right after the current bar
        i = i_end
    
    return bars_start, bars_width, bars_halfHeightUp, bars_halfHeightDown


def _algorithm2(ROI_thresh):
    bb_height, bb_width = ROI_thresh.shape

    half_height = math.ceil(bb_height/2)
    half_height_index = half_height-1

    # INIZIALIZATION
    i = 0  # Index for iterating over the pixels

    bars_start = []
    bars_width = []
    bars_halfHeightUp = []
    bars_halfHeightDown = []

    # CYCLE
    # We scan each pixel along the horizontal line in the exact middle of the ROI image
    while i<bb_width:

        # White pixel: we go to the next pixel
        if ROI_thresh[half_height_index, i]==255:
            i += 1
            continue

        # Black pixel
        # 'i' is the first pixel in this current barcode bar

        # Width of this current bar
        X_curr = 1    
        # Index representing the last pixel in this current bar. Actually, `i_end` is the pixel after the last pixel (i.e. 
        # first white pixel)
        i_end = i+1

        # We go right, till finding a white pixel.
        # In this way, we compute the width of this current bar.
        while ROI_thresh[half_height_index, i_end]==0:
            X_curr += 1
            i_end += 1

        # Now we search upward and downward
        # Index for goind upward.
        j_up = half_height_index-1
        # Index for goind downward.
        j_down = half_height_index+1
        # Half upward height of this current bar
        half_height_up_curr = 0
        # Half downard height of this current bar
        half_height_down_curr = 0

        # Flag saying whether the max up height has been reached or not
        up_reached = j_up<0 or ((ROI_thresh[j_up, i:i_end]==0).sum()/X_curr)<0.33         
        # Flag saying whether the max down height has been reached or not
        down_reached = j_down>bb_height-1 or ((ROI_thresh[j_down, i:i_end]==0).sum()/X_curr)<0.33

        # Cycle, in which we go upward and downard at the same time, for computing `half_height_up_curr` and 
        # `half_height_down_curr`
        while not up_reached or not down_reached:

            if not up_reached:
                half_height_up_curr += 1
            if not down_reached:
                half_height_down_curr += 1

            # We separate the increasing of `X_curr` left, right, up, down: 4 possibilities

            # Left increasing of `X_curr` on the vertical level `j_up`. "Left increasing" means before the index `i`.
            X_inc_left_up = 0
            # Right increasing of `X_curr` on the vertical level `j_up`. "Right increasing" means after the index `i_end`.
            X_inc_right_up = 0
            # Left increasing of `X_curr` on the vertical level `j_down`. "Left increasing" means before the index `i`.
            X_inc_left_down = 0
            # Right increasing of `X_curr` on the vertical level `j_down`. "Right increasing" means after the index `i_end`.
            X_inc_right_down = 0

            if not up_reached:  # Vertical level `j_up`
                while i-1>=0 and ROI_thresh[j_up, i-1]==0:  # Left expansion of `X_curr`
                    #print(8)
                    X_inc_left_up += 1
                    i -= 1 
                while i_end<bb_width and ROI_thresh[j_up, i_end]==0:  # Right expansion of `X_curr`
                    X_inc_right_up += 1
                    i_end +=1 
            if not down_reached:  # Vertical level `j_down`
                while i-1>=0 and ROI_thresh[j_down, i-1]==0:  # Left expansion of `X_curr`
                    X_inc_left_down += 1
                    i -= 1 
                while i_end<bb_width and ROI_thresh[j_down, i_end]==0:  # Right expansion of `X_curr`
                    X_inc_right_down += 1
                    i_end +=1 

            # Update `X_curr`, adding the maximum, both left and right.
            X_curr += max([X_inc_left_up,X_inc_left_down]) + max([X_inc_right_up,X_inc_right_down])

            j_up -= 1
            # Understand if we have reached the up-top of this current bar.
            # We have reached the up-top if the number of black pixels in this level `j_up` is less than the 0.33% of `X_curr`.
            up_reached = up_reached or j_up<0 or ((ROI_thresh[j_up, i:i_end]==0).sum()/X_curr)<0.33              

            j_down += 1
            # Understand if we have reached the down-top of this current bar.
            # We have reached the down-top if the number of black pixels in this level `j_down` is less than the 0.33% of `X_curr`.
            down_reached = down_reached or j_down>bb_height-1 or ((ROI_thresh[j_down, i:i_end]==0).sum()/X_curr)<0.33


        # Now we have computed the actual `X_curr`

        # Update the lists, inserting the values for this current bar
        bars_start.append(i)
        bars_width.append(X_curr)
        bars_halfHeightUp.append(half_height_up_curr)
        bars_halfHeightDown.append(half_height_down_curr)

        # We update `i`: we pass to the white pixel right after the current bar
        i = i_end
    
    return bars_start, bars_width, bars_halfHeightUp, bars_halfHeightDown


def _algorithm3(ROI_thresh):
    bb_height, bb_width = ROI_thresh.shape

    half_height = math.ceil(bb_height/2)
    half_height_index = half_height-1

    # INIZIALIZATION
    i = 0  # Index for iterating over the pixels

    bars_start = []
    bars_width = []
    bars_halfHeightUp = []
    bars_halfHeightDown = []

    # CYCLE
    # We scan each pixel along the horizontal line in the exact middle of the ROI image
    while i<bb_width:
        
         #print(i)

        # White pixel: we go to the next pixel
        if ROI_thresh[half_height_index, i]==255:
            i += 1
            continue

        # Black pixel
        # 'i' is the first pixel in this current barcode bar

        # Width of this current bar
        X_curr = 1    
        # Index representing the last pixel in this current bar. Actually, `i_end` is the pixel after the last pixel (i.e. 
        # first white pixel)
        i_end = i+1

        # We go right, till finding a white pixel.
        # In this way, we compute the width of this current bar.
        while ROI_thresh[half_height_index, i_end]==0:
            X_curr += 1
            i_end += 1
            
        # Index representing the next pixel in which we must go after the analysis of this current bar. Basically, `i_next`
        # represents the first white pixel after this bar along the middle horizontal line (i.e. `half_height_index`)
        i_next = i_end

        # Now we search upward and downward
        # Index for goind upward.
        j_up = half_height_index-1
        # Index for goind downward.
        j_down = half_height_index+1
        # Half upward height of this current bar
        half_height_up_curr = 0
        # Half downard height of this current bar
        half_height_down_curr = 0

        # Number of vertical levels explored so far
        l = 1
        # Index in the middle  between `i` and `i_end`
        i_med = math.ceil((i+i_end-1)/2)

        # Width of this current bar, to the left with respect to `i_med`
        X_curr_left = i_med-i
        # Width of this current bar, to the right with respect to `i_med`
        X_curr_right = i_end-i_med-1

        # Flag saying whether the max up height has been reached or not.
        # We have reached the up-top if the number of black pixels in this level `j_up` is less than the 0.33% of `X_curr`.
        up_reached = j_up<0 or ((ROI_thresh[j_up, i:i_end]==0).sum()/X_curr)<0.3
        # Flag saying whether the max down height has been reached or not.
        # We have reached the down-top if the number of black pixels in this level `j_down` is less than the 0.33% of `X_curr`.
        down_reached = j_down>bb_height-1 or ((ROI_thresh[j_down, i:i_end]==0).sum()/X_curr)<0.3

        # Cycle, in which we go upward and downard at the same time, for computing `half_height_up_curr` and 
        # `half_height_down_curr`
        while not up_reached or not down_reached:

            if not up_reached:
                half_height_up_curr += 1
            if not down_reached:
                half_height_down_curr += 1

            if not up_reached:  # Vertical level `j_up`
                # Left width `X_curr_left` on this current up level
                X_curr_left_up = 0
                # Right width `X_curr_right` on this current up level
                X_curr_right_up = 0
                # Index for going left and right, starting from `i_med`
                ii = 1
                # Flag saying if the left-most pixel has been reached (white pixel or border of the image)
                left_reached = i_med-ii<0 or ROI_thresh[j_up, i_med-ii]==255
                # Flag saying if the right-most pixel has been reached (white pixel or border of the image)
                right_reached = i_med+ii>bb_width-1 or ROI_thresh[j_up, i_med+ii]==255
                while not left_reached or not right_reached:  
                    if not left_reached:
                        X_curr_left_up += 1
                    if not right_reached:
                        X_curr_right_up += 1
                    ii += 1 
                    left_reached = i_med-ii<0 or ROI_thresh[j_up, i_med-ii]==255
                    right_reached = i_med+ii>bb_width-1 or ROI_thresh[j_up, i_med+ii]==255
                # Update `X_curr_left` using the mean, injecting this new value `X_curr_left_up`
                X_curr_left = (l*X_curr_left + X_curr_left_up)/(l+1)
                # Update `X_curr_right` using the mean, injecting this new value `X_curr_right_up`
                X_curr_right = (l*X_curr_right + X_curr_right_up)/(l+1)
                l = l+1  # Update the number of seen levels

            if not down_reached:  # Vertical level `j_down`
                # Left width `X_curr_left` on this current down level
                X_curr_left_down = 0
                # Right width `X_curr_right` on this current down level
                X_curr_right_down = 0
                # Index for going left and right, starting from `i_med`
                ii = 1
                # Flag saying if the left-most pixel has been reached (white pixel or border of the image)
                left_reached = i_med-ii<0 or ROI_thresh[j_down, i_med-ii]==255
                # Flag saying if the right-most pixel has been reached (white pixel or border of the image)
                right_reached = i_med+ii>bb_width-1 or ROI_thresh[j_down, i_med+ii]==255
                while not left_reached or not right_reached:  
                    if not left_reached:
                        X_curr_left_down += 1
                    if not right_reached:
                        X_curr_right_down += 1
                    ii += 1 
                    left_reached = i_med-ii<0 or ROI_thresh[j_down, i_med-ii]==255
                    right_reached = i_med+ii>bb_width-1 or ROI_thresh[j_down, i_med+ii]==255
                # Update `X_curr_left` using the mean, injecting this new value `X_curr_left_down`
                X_curr_left = (l*X_curr_left + X_curr_left_down)/(l+1)
                # Update `X_curr_right` using the mean, injecting this new value `X_curr_right_down`
                X_curr_right = (l*X_curr_right + X_curr_right_down)/(l+1)
                l = l+1  # Update the number of seen levels

            # Update the starting pixel of this current bar
            i = math.ceil(i_med-X_curr_left)
            # Update the ending pixel of this current bar (actually, first white pixel after the bar)
            i_end = int(i_med+X_curr_right)+1
            # Update `X_curr`
            X_curr = int(X_curr_left+X_curr_right)+1

            j_up -= 1
            # Understand if we have reached the up-top of this current bar.
            # We have reached the up-top if the number of black pixels in this level `j_up` is less than the 0.33% of `X_curr`.
            up_reached = up_reached or j_up<0 or ((ROI_thresh[j_up, i:i_end]==0).sum()/X_curr)<0.33              

            j_down += 1
            # Understand if we have reached the down-top of this current bar.
            # We have reached the down-top if the number of black pixels in this level `j_down` is less than the 0.33% of `X_curr`.
            down_reached = down_reached or j_down>bb_height-1 or ((ROI_thresh[j_down, i:i_end]==0).sum()/X_curr)<0.33     


        # Now we have computed the actual `X_curr`  

        # Update the lists, by adding the quantities computed on this current bar
        bars_start.append(i)
        bars_width.append(X_curr)
        bars_halfHeightUp.append(half_height_up_curr)
        bars_halfHeightDown.append(half_height_down_curr)

        # We update `i`: we pass to the white pixel right after the current bar (along the middle horizontal line `half_height_index`)
        i = i_next
    
    return bars_start, bars_width, bars_halfHeightUp, bars_halfHeightDown


def _algorithm4(ROI_thresh):
    bb_height, bb_width = ROI_thresh.shape

    half_height = math.ceil(bb_height/2)
    half_height_index = half_height-1
    half_width = math.ceil(bb_width/2)

    # INIZIALIZATION
    i = 0  # Index for iterating over the pixels

    bars_start = []
    bars_width = []
    bars_halfHeightUp = []
    bars_halfHeightDown = []

    # Table, where the indices correspond to all possible widths to the left of `i_mid` for each bar. All possible 
    # `X_curr_left`.
    # This table counts the occourances of each `X_curr_left`.
    X_curr_left_table = np.zeros((half_width-1))
    # Table, where the indices correspond to all possible widths to the left of `i_mid` for each bar. All possible `X_curr_right`.
    # This table counts the occourances of each `X_curr_right`.
    X_curr_right_table = np.zeros((half_width-1))

    # CYCLE
    # We scan each pixel along the horizontal line in the exact middle of the ROI image
    while i<bb_width:

        # White pixel: we go to the next pixel
        if ROI_thresh[half_height_index, i]==255:
            i += 1
            continue

        # Black pixel
        # 'i' is the first pixel in this current barcode bar

        # Width of this current bar
        X_curr = 1    
        # Index representing the last pixel in this current bar. Actually, `i_end` is the pixel after the last pixel (i.e. 
        # first white pixel)
        i_end = i+1

        # We go right, till finding a white pixel.
        # In this way, we compute the width of this current bar.
        while ROI_thresh[half_height_index, i_end]==0:
            X_curr += 1
            i_end += 1
            
        # Index representing the next pixel in which we must go after the analysis of this current bar. Basically, `i_next`
        # represents the first white pixel after this bar along the middle horizontal line (i.e. `half_height_index`)
        i_next = i_end

        # Now we search upward and downward
        # Index for goind upward.
        j_up = half_height_index-1
        # Index for goind downward.
        j_down = half_height_index+1
        # Half upward height of this current bar
        half_height_up_curr = 0
        # Half downard height of this current bar
        half_height_down_curr = 0

        # Index in the middle between `i` and `i_end`
        i_med = int((i+i_end-1)/2)

        # Initialize the table `X_curr_left_table` for this current bar
        X_curr_left_table *= 0  # Azzerate
        X_curr_left_table[i_med-i] += 1  # Place a 1 on the just found left width
        # Initialize the table `X_curr_left_table` for this current bar
        X_curr_right_table *= 0  # Azzerate
        X_curr_right_table[i_end-i_med-1] += 1  # Place a 1 on the just found right width

        # Flag saying whether the max up height has been reached or not.
        # We have reached the up-top if the number of black pixels in this level `j_up` is less than the 0.33% of `X_curr`.
        up_reached = j_up<0 or ((ROI_thresh[j_up, i:i_end]==0).sum()/X_curr)<0.3
        # Flag saying whether the max down height has been reached or not.
        # We have reached the down-top if the number of black pixels in this level `j_down` is less than the 0.33% of `X_curr`.
        down_reached = j_down>bb_height-1 or ((ROI_thresh[j_down, i:i_end]==0).sum()/X_curr)<0.3
        #print(7)

        # Cycle, in which we go upward and downard at the same time, for computing `half_height_up_curr` and 
        # `half_height_down_curr`
        while not up_reached or not down_reached:

            if not up_reached:
                half_height_up_curr += 1
            if not down_reached:
                half_height_down_curr += 1

            if not up_reached:  # Vertical level `j_up`
                # Left width `X_curr_left` on this current up level
                X_curr_left_up = 0
                # Right width `X_curr_right` on this current up level
                X_curr_right_up = 0
                # Index for going left and right, starting from `i_med`
                ii = 1
                # Flag saying if the left-most pixel has been reached (white pixel or border of the image)
                left_reached = i_med-ii<0 or ROI_thresh[j_up, i_med-ii]==255
                # Flag saying if the right-most pixel has been reached (white pixel or border of the image)
                right_reached = i_med+ii<0 or ROI_thresh[j_up, i_med+ii]==255
                while not left_reached or not right_reached:  
                    if not left_reached:
                        X_curr_left_up += 1
                    if not right_reached:
                        X_curr_right_up += 1
                    ii += 1 
                    left_reached = i_med-ii<0 or ROI_thresh[j_up, i_med-ii]==255
                    right_reached = i_med+ii<0 or ROI_thresh[j_up, i_med+ii]==255
                # Update `X_curr_left` using the mean, injecting this new value `X_curr_left_up`
                X_curr_left_table[X_curr_left_up] += 1
                # Update `X_curr_right` using the mean, injecting this new value `X_curr_right_up`
                X_curr_right_table[X_curr_right_up] += 1

            if not down_reached:  # Vertical level `j_down`
                # Left width `X_curr_left` on this current down level
                X_curr_left_down = 0
                # Right width `X_curr_right` on this current down level
                X_curr_right_down = 0
                # Index for going left and right, starting from `i_med`
                ii = 1
                # Flag saying if the left-most pixel has been reached (white pixel or border of the image)
                left_reached = i_med-ii<0 or ROI_thresh[j_down, i_med-ii]==255
                # Flag saying if the right-most pixel has been reached (white pixel or border of the image)
                right_reached = i_med+ii<0 or ROI_thresh[j_down, i_med+ii]==255
                while not left_reached or not right_reached:  
                    if not left_reached:
                        X_curr_left_down += 1
                    if not right_reached:
                        X_curr_right_down += 1
                    ii += 1 
                    left_reached = i_med-ii<0 or ROI_thresh[j_down, i_med-ii]==255
                    right_reached = i_med+ii<0 or ROI_thresh[j_down, i_med+ii]==255
                # Update `X_curr_left` using the mean, injecting this new value `X_curr_left_down`
                X_curr_left_table[X_curr_left_up] += 1
                # Update `X_curr_right` using the mean, injecting this new value `X_curr_right_down`
                X_curr_right_table[X_curr_right_up] += 1

            X_curr_left = np.argmax(X_curr_left_table)   
            X_curr_right = np.argmax(X_curr_right_table)  

            # Update the starting pixel of this current bar
            i = i_med-X_curr_left
            # Update the ending pixel of this current bar (actually, first white pixel after the bar)
            i_end = i_med+X_curr_right+1
            # Update `X_curr`
            X_curr = X_curr_left+X_curr_right+1

            j_up -= 1
            # Understand if we have reached the up-top of this current bar.
            # We have reached the up-top if the number of black pixels in this level `j_up` is less than the 0.33% of `X_curr`.
            up_reached = up_reached or j_up<0 or ((ROI_thresh[j_up, i:i_end]==0).sum()/X_curr)<0.33              

            j_down += 1
            # Understand if we have reached the down-top of this current bar.
            # We have reached the down-top if the number of black pixels in this level `j_down` is less than the 0.33% of `X_curr`.
            down_reached = down_reached or j_down>bb_height-1 or ((ROI_thresh[j_down, i:i_end]==0).sum()/X_curr)<0.33     


        # Now we have computed the actual `X_curr`  

        # Update the lists, by adding the quantities computed on this current bar
        bars_start.append(i)
        bars_width.append(X_curr)
        bars_halfHeightUp.append(half_height_up_curr)
        bars_halfHeightDown.append(half_height_down_curr)

        # We update `i`: we pass to the white pixel right after the current bar  (along the middle horizontal line `half_height_index`)
        i = i_next
    
    return bars_start, bars_width, bars_halfHeightUp, bars_halfHeightDown



