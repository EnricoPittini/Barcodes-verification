import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils
import math


def sort_bb_points(bb_points):
    """Function which sorts the bb points according to our standard ordering, namely upper-left -> upper-right -> lower-left 
    -> lower-right."""

    min_width = bb_points[:,0].min()
    min_height = bb_points[:,1].min()
    max_width = bb_points[:,0].max()
    max_height = bb_points[:,1].max()
    def normalize(value, axis=0):
        if axis==0:  # Horizontal dimension
            return min_width if (value-min_width<max_width-value) \
                            else max_width
        elif axis==1:  # Vertical dimension
            return min_height if (value-min_height<max_height-value) \
                            else max_height
    bb_points_sorted = np.array(sorted([tuple(v) for v in bb_points], key=lambda t: (normalize(t[1], axis=1),
                                                                                                normalize(t[0], axis=0))))

    return bb_points_sorted


def sort_bb_points_for_visualization(bb_points_sorted):
    """Function which sorts the bb points differently, for making the bb compliant with the visualization API"""
    bb_rot = bb_points_sorted.copy()
    bb_rot[2, :] = bb_points_sorted[3, :]
    bb_rot[3, :] = bb_points_sorted[2, :]
    return bb_rot.astype(int)


def fix_horizontalBars_case(image_rot, bb_points_sorted_rot, bb_width, bb_height, visualize_fixed_image_bb=False):
    """Fix the horizontal barcode bars problem, if present. 
    If this problem is present, the input input is rotated in order to have the bars perfectly vertical, instead of perfectly
    horizontal. The bounding box is rotated accordingly too.

    It is important to point out that the input image (and the input bounding box) is the one obtained from the first rotation,
    the one which made the bounding box perfectly aligned with the image axes. 

    Parameters
    ----------
    image_rot : np.array
        Input image, rotated in order to have the bounding box perfectly aligned with the image axes
    bb_points_sorted_rot : np.array
        Array containing the coordinates of the 4 verteces of the bounding box. It is a rotated bounding box, such that 
        it is perfectly aligned with the image axes.
    width : int
        Width of the given bounding box.
    height : int
        Height of the given bounding box.
    visualize_fixed_image_bb : bool, optional
        Whether to visualize or not the fixed input image with the fixed bounding box, by default True

    Returns
    -------
    image_rot_rot : np.array
        Input image, after the possible fixing of the horizontal bars problem. 
        If this problem is not present, then this is equal to `image_rot`.
    bb_points_sorted_rot_rot : np.array
        Coordinates of the bounding box points, after the possible fixing of the horizontal bars problem. 
        If this problem is not present, then this is equal to `bb_points_sorted_rot`.
    bb_width : int
        Width of the bounding box `bb_points_sorted_rot_rot`, after the possible fixing of the horizontal bars problem. 
        If this problem is not present, then this is equal to `width`.
    bb_height : int
        Height of the bounding box `bb_points_sorted_rot_rot`, after the possible fixing of the horizontal bars problem. 
        If this problem is not present, then this is equal to `heigth`.

    """
    gray_rot = cv2.cvtColor(image_rot, cv2.COLOR_BGR2GRAY)
    roi_image = gray_rot[int(bb_points_sorted_rot[0][1]):int(bb_points_sorted_rot[0][1]+bb_height), 
                         int(bb_points_sorted_rot[0][0]):int(bb_points_sorted_rot[0][0]+bb_width)]

    # Understand if the barcode is rotated (i.e. horizontal bars) or not.
    # For doing so, we compute the horizontal and vertical gradient images: if the sum on the latter is bigger than the sum 
    # on the former, this means that the barcode is rotated.
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    gradX = cv2.Sobel(roi_image, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(roi_image, ddepth=ddepth, dx=0, dy=1, ksize=-1)
    barcode_rotated = cv2.convertScaleAbs(gradY).sum()>cv2.convertScaleAbs(gradX).sum()

    if not barcode_rotated:
        return image_rot, bb_points_sorted_rot, bb_width, bb_height

    else:
        height, width = gray_rot.shape
        # Source points for computing the homography. These are the four verteces of the current image
        # N.B. : with "current image" we mean the image obtained after the rotation performed before (rotation for making the 
        # bounding box perfectly aligned)
        coordinates_source = np.array([[0, 0],
                                    [width-1, 0],
                                    [0, height-1],
                                    [width-1, height-1]], dtype='float32')

        # Dimensions of the destination image (i.e. rotated image)
        destination_height, destination_width = width, height

        # Corresponding destination points, for computing the homography. These are the corresponding four verteces of the rotated 
        # image.
        coordinates_destination = np.array([[destination_width-1, 0],
                                                [destination_width-1, destination_height-1],
                                                [0, 0],                                        
                                                [0, destination_height-1]], dtype='float32')

        # Computing the trasformation, i.e. homography/warping. It's a rotation
        H = cv2.getPerspectiveTransform(coordinates_source, coordinates_destination)

        # Applying the trasformation: we rotate the current image
        #gray_rot_rot = cv2.warpPerspective(gray_rot, H, (destination_width, destination_height))

        image_rot_rot = cv2.warpPerspective(image_rot, H, (destination_width, destination_height))

        bb_points_sorted_rot_rot = cv2.perspectiveTransform(bb_points_sorted_rot.reshape(-1,1,2),H)
        bb_points_sorted_rot_rot = bb_points_sorted_rot_rot[:,0,:]

        bb_width, bb_height = bb_height, bb_width

        bb_points_sorted_rot_rot = sort_bb_points(bb_points_sorted_rot_rot)

        if visualize_fixed_image_bb:
            image_rot_rot_bb = image_rot_rot.copy()
            cv2.drawContours(image_rot_rot_bb, [sort_bb_points_for_visualization(bb_points_sorted_rot_rot)], -1, (0, 255, 0), 3)
            plt.figure()
            plt.imshow(image_rot_rot_bb, 'gray')
            plt.title('Fixed horizontal bars problem')

        return image_rot_rot, bb_points_sorted_rot_rot, bb_width, bb_height

        
