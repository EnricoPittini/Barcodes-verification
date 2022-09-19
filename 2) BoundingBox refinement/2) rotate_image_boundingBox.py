import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils
import math


def rotate_image(image, angle, center):
  """Function which rotates the given image, by the given angle, along the given centre"""
  # Rotation matrix
  rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
  # Rotated image
  image_rot = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return image_rot, rot_mat


def sort_bb_points_for_visualization(bb_points_sorted):
    """Function which sorts the bb points differently, for making the bb compliant with the visualization API"""
    bb_rot = bb_points_sorted.copy()
    bb_rot[2, :] = bb_points_sorted[3, :]
    bb_rot[3, :] = bb_points_sorted[2, :]
    return bb_rot.astype(int)


def rotate_image_boundingBox(image, bb_points_sorted, bb_width, bb_height, visualize_rot_image_bb=False):
    """Rotate the given image and the given bounding box, such that the bounding box becomes perfectly aligned with the 
    image axes.

    Parameters
    ----------
    image : np.array
        Input image
    bb_points_sorted : np.array
        Array 4x2, containing the coordinates of the four bounding box points. 
        The points are ordered in the following way: up-left, up-right, bottom-left, bottom-right.
    bb_width : int
        Width of the bounding box.
    bb_height : _type_
        Height of the bounding box.
    visualize_rot_image_bb : bool, optional
        Whether to visualize or not the rotated input image with the rotated bounding box, by default False

    Returns
    -------
    image_rot : np.array
        Rotated input image
    bb_points_sorted_rot : np.array
        The rotated bounding box. More precisely, array containing the 4 verteces of the rotated bounding box
    """
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    point1 = bb_points_sorted[0, :]
    point2 = bb_points_sorted[1, :]
    point3 = bb_points_sorted[2, :]
    point4 = bb_points_sorted[3, :]
    
    angle1 = math.degrees(math.atan((point2[1]-point1[1])/(point2[0]-point1[0])))
    #angle2 = 90-abs(math.degrees(math.atan((point1[1]-point3[1])/(point1[0]-point3[0]))))
    angle = angle1#(angle1+angle2)/2
    
    if abs(angle)<10**(-4):  # The angle is 0: the bounding box is already perfectly aligned. No rotation is perfomed.
        #gray_rot, image_rot, bb_points_sorted_rot = gray, image, bb_points_sorted
        image_rot, bb_points_sorted_rot = image, bb_points_sorted
    
    else:  # The angle is not 0: a rotation of the image must be perfomed.
        bb_points_sorted_rot = np.array([point1,
                              [point1[0]+bb_width-1,point1[1]],
                              [point1[0],point1[1]+bb_height-1],
                              [point1[0]+bb_width-1,point1[1]+bb_height-1]], dtype='float32') 
        
        image_rot, rot_mat = rotate_image(image, angle=angle, center=point1)

    if visualize_rot_image_bb:
        image_rot_bb = image_rot.copy()
        cv2.drawContours(image_rot_bb, [sort_bb_points_for_visualization(bb_points_sorted_rot)], -1, (0, 255, 0), 3)
        plt.figure()
        plt.imshow(image_rot_bb, 'gray')
        plt.title('Rotated image, with the rotated bounding box')
    
    return image_rot, bb_points_sorted_rot