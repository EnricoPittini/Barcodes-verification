import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils
import math

from utils import sort_bb_points, sort_bb_points_for_visualization



def rotate_boundingBox(image, bb_points_sorted, bb_width, bb_height, fix_horizontalBars_case=True, visualize_rotatedImage_boundingBox=False):
    """Rotate the given image and bounding box surrounding the barcode, such that the barcode bars become perfectly vertical.

    Basically, a rotation is performed such that the bounding box becomes perfectly aligned with the image axes.

    If the barcode is rotated, i.e. the bars are horixontal, the bars are not perfectly vertical, but perfectly horizontal.
    In this case, another rotation must be performed, in order to make the barcode bars perfectly vertical.

    Parameters
    ----------
    image : np.array
        Input image, containing a barcode
    bb_points_sorted : np.array
        Bounding box surrounding the barcode.
        Array 4x2, containing the coordinates of the four bounding box points. 
        The points are ordered in the following way: up-left, up-right, bottom-left, bottom-right.
    bb_width : int
        Width of the detected bounding box
    bb_height : int
        Height of the detected bounding box
    fix_horizontalBars_case : bool, optional
        Whether to fix the horizontal bars case or not, by default True.
        If True, the possible horizontal bars case is handleled: another rotation is performed, for making the barcode bars 
        perfectly vertical.
        If False, the possible horizontal bars case is not handleled: the barcode bars remain perfectly horizontal.
    visualize_rotatedImage_boundingBox : bool, optional
        Whether to visualize or not the rotated input image with the rotated bounding box, by default False

    Returns
    -------
    image_rot : np.array
        Rotated input image
    bb_points_sorted_rot : np.array
        The rotated bounding box. More precisely, array 4x2 containing the coordinates of the 4 verteces of the rotated 
        bounding box. These four verteces are ordered according to our standard ordering, namely upper-left -> upper-right -> 
        lower-left -> lower-right.
    roi_image : np.array
        Rotated input image cropped around the rotated bounding box. Basically, sub-image containing only the barcode (i.e. 
        the ROI), perfectly aligned with respect to the image axes.
        It is important to point out that this ROI image is in gray-scale, not colored.
    angle : float
        Orientation of the original bounding box with in the original image.
        Basically, angle of the original bounding box with respect to the original horixontal axis.

    """

    # Rotate the image and the bounding box, such that the bounding box becomes perfectly aligned with the image axes
    image_rot, bb_points_sorted_rot, angle = _rotate_image_boundingBox(image, bb_points_sorted, bb_width, bb_height)

    # Fix the horizontal bars case.
    # In this way, we are sure that the barcode bars are perfectly vertical.
    if fix_horizontalBars_case:
        image_rot, bb_points_sorted_rot, bb_width, bb_height = _fix_horizontalBars_case(image_rot, bb_points_sorted_rot, 
                                                                         bb_width, bb_height, visualize_fixed_image_bb=False)

    if visualize_rotatedImage_boundingBox:  # Visualize the rotated image with the rotated bounding box
        image_rot_bb = image_rot.copy()
        cv2.drawContours(image_rot_bb, [sort_bb_points_for_visualization(bb_points_sorted_rot)], -1, (0, 255, 0), 3)
        plt.figure()
        plt.imshow(image_rot_bb, 'gray')
        plt.title('Rotated image, with the rotated bounding box')

    # Gray-scale rotated image
    gray_rot = cv2.cvtColor(image_rot, cv2.COLOR_BGR2GRAY)  
    # Crop the rotated image around the rotated bounding box: ROI image
    roi_image = gray_rot[int(bb_points_sorted_rot[0][1]):int(bb_points_sorted_rot[0][1]+bb_height), 
                             int(bb_points_sorted_rot[0][0]):int(bb_points_sorted_rot[0][0]+bb_width)]

    return image_rot, bb_points_sorted_rot, roi_image, angle




def _rotate_image(image, angle, center):
  """Function which rotates the given image by the given angle with respect to the given centre"""
  # Rotation matrix
  rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
  # Rotated image
  image_rot = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return image_rot, rot_mat



def _rotate_image_boundingBox(image, bb_points_sorted, bb_width, bb_height):
    """Rotate the given image and the bounding box containing the barcode, such that the bounding box becomes perfectly 
    aligned with the image axes.

    Parameters
    ----------
    image : np.array
        Input image, containing the barcode
    bb_points_sorted : np.array
        Bounding box surrounding the barcode.
        Array 4x2, containing the coordinates of the four bounding box points. 
        The points are ordered in the following way: up-left, up-right, bottom-left, bottom-right.
    bb_width : int
        Width of the bounding box.
    bb_height : int
        Height of the bounding box.

    Returns
    -------
    image_rot : np.array
        Rotated input image
    bb_points_sorted_rot : np.array
        The rotated bounding box. More precisely, array containing the 4 verteces of the rotated bounding box

    """
    
    # First two bounding box verteces
    point1 = bb_points_sorted[0, :]
    point2 = bb_points_sorted[1, :]
    #point3 = bb_points_sorted[2, :]
    #point4 = bb_points_sorted[3, :]
    
    # Angle between the line connecting point1-point2 and the horixontal axis
    angle = math.degrees(math.atan((point2[1]-point1[1])/(point2[0]-point1[0])))
    #angle2 = 90-abs(math.degrees(math.atan((point1[1]-point3[1])/(point1[0]-point3[0]))))
    #angle = (angle1+angle2)/2
    
    if abs(angle)<10**(-4):  # The angle is 0: the bounding box is already perfectly aligned with the image axes.
                             # No rotation is perfomed.
        image_rot, bb_points_sorted_rot = image, bb_points_sorted
    
    else:  # The angle is not 0: a rotation of the image must be perfomed.

        # Bounding box rotated
        bb_points_sorted_rot = np.array([point1,
                              [point1[0]+bb_width-1,point1[1]],
                              [point1[0],point1[1]+bb_height-1],
                              [point1[0]+bb_width-1,point1[1]+bb_height-1]], dtype='float32') 
        
        # Rotate the image, by angle `angle` and with respect to the centre `point1`
        image_rot, rot_mat = _rotate_image(image, angle=angle, center=point1)
    
    return image_rot, bb_points_sorted_rot, angle



def _fix_horizontalBars_case(image_rot, bb_points_sorted_rot, bb_width, bb_height):
    """Fix the horizontal barcode bars problem, if present. 
    If this problem is present, the input image is rotated in order to have the bars perfectly vertical, instead of perfectly
    horizontal. The bounding box surrounding the barcode is rotated accordingly too.

    It is important to point out that the input image (and the input bounding box) is the one obtained from the first rotation,
    the one which made the bounding box perfectly aligned with the image axes. 

    Parameters
    ----------
    image_rot : np.array
        Input image containing the barcode, rotated in order to have the bounding box perfectly aligned with the image axes
    bb_points_sorted_rot : np.array
        Bounding box surrounding the barcode, rotated in order to be perfectly aligned with the image axes.
        Array 4x2, containing the coordinates of the four bounding box points. 
        The points are ordered in the following way: up-left, up-right, bottom-left, bottom-right.
    width : int
        Width of the given bounding box.
    height : int
        Height of the given bounding box.

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

    # Gray-scale input image, rotated in order to have the barcode perfectly aligned w.r.t. the image axes
    gray_rot = cv2.cvtColor(image_rot, cv2.COLOR_BGR2GRAY)
    # Crop the rotated image around the rotated bounding box: ROI image.
    # It is used for understanding if the barcode is rotated (i.e. horizontal bars) or not.
    roi_image = gray_rot[int(bb_points_sorted_rot[0][1]):int(bb_points_sorted_rot[0][1]+bb_height), 
                         int(bb_points_sorted_rot[0][0]):int(bb_points_sorted_rot[0][0]+bb_width)]

    # Understand if the barcode is rotated (i.e. horizontal bars) or not.
    # For doing so, we compute the horizontal and vertical gradient images: if the sum on the latter is bigger than the sum 
    # on the former, this means that the barcode is rotated (bigger changes along the horizontal axis rather than the 
    # vertical axis)
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    gradX = cv2.Sobel(roi_image, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(roi_image, ddepth=ddepth, dx=0, dy=1, ksize=-1)
    barcode_rotated = cv2.convertScaleAbs(gradY).sum()>cv2.convertScaleAbs(gradX).sum()

    if not barcode_rotated:
        # The barcode is not rotated, i.e. the bars are perfectly vertical.
        # We don't do anything
        return image_rot, bb_points_sorted_rot, bb_width, bb_height 

    else:
        # The barcode is rotated, i.e. the bars are perfectly horizontal.
        # We must perform a rotation, implemented through a warping/homography.

        # Dimensions of the input image
        height, width = gray_rot.shape

        # Source points for computing the homography. These are the four verteces of the current image.
        # N.B. : with "current image" we mean the image obtained after the rotation performed before (rotation for making the 
        # bounding box perfectly aligned)
        coordinates_source = np.array([[0, 0],
                                    [width-1, 0],
                                    [0, height-1],
                                    [width-1, height-1]], dtype='float32')

        # Dimensions of the destination image (i.e. rotated image). The width becomes the height and the height becomes the 
        # width.
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
        image_rot_rot = cv2.warpPerspective(image_rot, H, (destination_width, destination_height))

        # Applying the trasformation: we rotated the bounding box verteces
        bb_points_sorted_rot_rot = cv2.perspectiveTransform(bb_points_sorted_rot.reshape(-1,1,2),H)
        bb_points_sorted_rot_rot = bb_points_sorted_rot_rot[:,0,:]
        # Sort the verteces according to our standard ordering, namely upper-left -> upper-right -> lower-left -> lower-right
        bb_points_sorted_rot_rot = sort_bb_points(bb_points_sorted_rot_rot)  

        # Width and height of the new rotated image (the dimensions are swapped)
        bb_width, bb_height = bb_height, bb_width

        return image_rot_rot, bb_points_sorted_rot_rot, bb_width, bb_height