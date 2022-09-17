import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils
import math
import matplotlib.patches as patches
import time



def compute_barcode_structure(roi_image, bb_width, bb_height, algorithm=1, verbose=False, visualize_refined_bb=False,
                              visualize_barcode_structure=False):
    """Compute the complete barcode structure.

    Namely, it computes:
    - the starting pixel of each bar;
    - the width of each bar;
    - the half height up of each bar;
    - the half height down of each bar.

    Parameters
    ----------
    roi_image : np.array
        The original image cropped around the ROI (i.e. the barcode)
    bb_width : int
        Width of the bounding box
    bb_height : int
        Height of the bounding box
    algorithm : int, optional
        Algorithm to use for computing the barcode structure, by default 1.
        Choices among 1,2,3,4.
    verbose : bool, optional
        Whether to print the solving time or not, by default False
    visualize_refined_bb : bool, optional
        Whether to visualize the refined ROI image or not, by default False
    visualize_barcode_structure : bool, optional
        Whether to visualize the barcode structure or not, by default False

    Returns
    -------
    bars_start : list of int
        List containing the starting pixel of each bar
    bars_width : list of int
        List containing the width of each bar
    bars_halfHeightUp : list of int
        List containing the half heigth up of each bar
    bars_halfHeightDown : list of int
        List containing the half height down of each bar

    Raises
    ------
    ValueError
        If a wrong algorithm index is given

    """
    _ ,ROI_thresh = cv2.threshold(roi_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    start_time = time.time()

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

    bars_start, bars_width, bars_halfHeightUp, bars_halfHeightDown = algorithm_function(ROI_thresh, bb_width, bb_height)
       
    first_bar_x = min(bars_start)
    last_bar_x = max([s+w for s,w in zip(bars_start,bars_width)])-1
    X = min(bars_width)
    min_half_height_up = min(bars_halfHeightUp)
    min_half_height_down = min(bars_halfHeightDown)
    
    half_height = int(bb_height/2)
    
    end_time = time.time()

    if verbose:
        print('Time:', end_time-start_time)    
  
    if visualize_refined_bb:      
        plt.figure(figsize=(3, 3))
        plt.imshow(roi_image, 'gray')
        plt.axhline(half_height-min_half_height_up-1, c='green', label='Min up height')
        plt.axhline(half_height+min_half_height_down-1, c='blue', label='Min down height')
        plt.axvline(first_bar_x, c='red', label='first_bar_x')
        plt.axvline(last_bar_x, c='orange', label='last_bar_x')
        plt.title('Refined ROI, with the computed quantities')
        plt.legend()

    if visualize_barcode_structure:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(roi_image, 'gray')
        n_bars = len(bars_start)
        for b in range(n_bars):
            # Create a Rectangle patch
            rect = patches.Rectangle((bars_start[b]-0.5, half_height-bars_halfHeightUp[b]-1-0.5), bars_width[b], 
                                     bars_halfHeightUp[b]+bars_halfHeightDown[b]+1, linewidth=1, edgecolor='r', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
        plt.show()
        ax.set_title('Exaustive barcode structure')
    
    
    return bars_start, bars_width, bars_halfHeightUp, bars_halfHeightDown



###### ALGORITHM 1
def _algorithm1(ROI_thresh, bb_width, bb_height):
    half_height = int(bb_height/2)
    half_height_index = half_height-1

    # INIZIALIZATION
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
        # Index representing the last pixel in this current bar. Actually, `i_end` is the pixel after the last pixel (i.e. 
        # first white pixel)
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

        # Flag saying whether the max up height has been reached or not: the three consecutive pixels i_med-1, i_med, I-med+1
        # must be all white (on the level j_up)
        up_reached = j_up<0 or (ROI_thresh[j_up, i_med]==255 and  ROI_thresh[j_up, i_med-1]==255 and ROI_thresh[j_up, i_med+1]==255)
        # Flag saying whether the max down height has been reached or not: the three consecutive pixels i_med-1, i_med, I-med+1
        # must be all white (on the level j_down)
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

        # Update the lists, inserting the values for this current bar
        bars_start.append(i)
        bars_width.append(X_curr)
        bars_halfHeightUp.append(half_height_up_curr)
        bars_halfHeightDown.append(half_height_down_curr)

        # We update `i`: we pass to the white pixel right after the current bar
        i = i_end
    
    return bars_start, bars_width, bars_halfHeightUp, bars_halfHeightDown


###### ALGORITHM 2
def _algorithm2(ROI_thresh, bb_width, bb_height):
    half_height = int(bb_height/2)
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


##### ALGORITHM 3
def _algorithm3(ROI_thresh, bb_width, bb_height):
    half_height = int(bb_height/2)
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


##### ALGORITHM 4
def _algorithm4(ROI_thresh, bb_width, bb_height):
    half_height = int(bb_height/2)
    half_height_index = half_height-1
    half_width = int(bb_width/2)

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