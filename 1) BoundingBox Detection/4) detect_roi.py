# Version 4
# https://pyimagesearch.com/2014/11/24/detecting-barcodes-images-python-opencv/

# import the necessary packages
import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils


def detect_roi(image, visualize_bounding_box=False, visualize_final_image=False):
    """Detect the ROI surrounding the barcode in the given image.

    It returns the bounding box coordinates. In addition, it warps the bounding box, in order to rotate it and making the 
    bars perfectly vertical.

    This works both if the barcode is non-rotated (i.e. the bars are vertical) and if the barcode is rotated (i.e. the bars 
    are horizontal). 

    Parameters
    ----------
    image : np.array
        Input image.
    visualize_bounding_box : bool, optional
        Whether to visualize or not the input image with the detected bounding box, by default False
    visualize_final_image : bool, optional
        Whether to visualize or not the warped bounding box, by default False

    Returns
    -------
    bb_sorted : np.array
        Array 4x2, containing the coordinates of the four bounding box points. 
        The points are ordered in the following way: up-left, up-right, bottom-left, bottom-right.
    
    bb_image_warped : np.array
        Image representing the warped bounding box: the barcode bars are perfectly vertical.

    rotated : bool
        Whether the barcode was non-rotated (i.e. the bars were vertical) or rotated (i.e. the bars were horizontal).

    """

    # Convert the image to grey scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute both the horizontal and vertical derivative, using the Sobel filter
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
    # Subtract the y-gradient from the x-gradient: we get the final gradient image
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # Blur the gradient image    
    blurred = cv2.blur(gradient, (9, 9))

    # Threshold the gradient image
    _ ,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Closing: fill the bounding box
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Opening: remove small things outside
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    
    # Dilate: slight enlarge the box
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    dilated = cv2.dilate(opened, kernel, iterations=3)
    
    # Find the bounding box (OpenCV API)
    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
    # Compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
    # This is our bounding box: 4 points, each of them 2 coordinates
    box = np.int0(box)  

    if visualize_bounding_box:
        # Draw a bounding box around the detected barcode and display the image
        image_bb = image.copy()
        cv2.drawContours(image_bb, [box], -1, (0, 255, 0), 3)
        plt.figure()
        plt.imshow(image_bb, 'gray')
        plt.title('Bounding box')
    
    # Sorting the points of the bounding box, such that they follow the ordering: up-left, up-right, bottom-left, bottom-right.
    bb_sorted = box.astype('float32')
    min_width = bb_sorted[:,0].min()
    min_height = bb_sorted[:,1].min()
    max_width = bb_sorted[:,0].max()
    max_height = bb_sorted[:,1].max()
    def normalize(value, axis=0):
        if axis==0:  # Horizontal dimension
            return min_width if (value-min_width<max_width-value) \
                             else max_width
        elif axis==1:  # Vertical dimension
            return min_height if (value-min_height<max_height-value) \
                              else max_height
    bb_sorted = np.array(sorted([tuple(v) for v in bb_sorted], key=lambda t: (normalize(t[1], axis=1),
                                                                                                normalize(t[0], axis=0))))

    # The 4 bounding box points are the coordinates of the source image, which will be used for computing the warping/homography.
    # Warping/homography from the source image,i.e. bounding box, to the destination image, i.e. enlarged bounding box.
    coordinates_source = bb_sorted

    # Compute the width and height of the bounding box.
    # Why? Because we will use these quantities for defining the dimensions of the destination image of the warping/homography.
    def dist(point1, point2):
        return np.sqrt(np.sum((point1-point2)**2))
    destination_height = int(max([dist(coordinates_source[0],coordinates_source[2]),
                             dist(coordinates_source[1],coordinates_source[3])]))
    destination_width = int(max([dist(coordinates_source[0],coordinates_source[1]),
                             dist(coordinates_source[2],coordinates_source[3])]))

    # Define the destination image points corresponding to the source image points.
    # These are simply the 4 angles of the full destination image, which has width `destination_width` and height `destination_height`
    coordinates_destination = np.array([[0, 0],
                                        [destination_width-1, 0],
                                        [0, destination_height-1],
                                        [destination_width-1, destination_height-1]], dtype='float32')

    # Computing the trasformation, i.e. homography. Warping.
    H = cv2.getPerspectiveTransform(coordinates_source, coordinates_destination)

    # Applying the trasformation
    bb_image_warped = cv2.warpPerspective(gray, H, (destination_width, destination_height))

    # Understand if the barcode is rotated (i.e. horizontal bars) or not.
    # For doing so, we compute the horizontal and vertical gradient images: if the sum on the latter is bigger than the sum 
    # on the former, this means that the barcode is rotated.
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    gradX = cv2.Sobel(bb_image_warped, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(bb_image_warped, ddepth=ddepth, dx=0, dy=1, ksize=-1)
    rotated = cv2.convertScaleAbs(gradY).sum()>cv2.convertScaleAbs(gradX).sum()

    # If the barcode is rotate, we must rotate the warped image accordingly.
    # We perform the rotation by applying a warping/homography.
    if rotated:
        destination_height_rot, destination_width_rot = destination_width, destination_height
        coordinates_destination_rot = np.array([[destination_width_rot-1, 0],
                                                [destination_width_rot-1, destination_height_rot-1],
                                                [0, 0],                                        
                                                [0, destination_height_rot-1]], dtype='float32')

        # Computing the trasformation, i.e. homography/warping. It's a rotation
        H = cv2.getPerspectiveTransform(coordinates_destination, coordinates_destination_rot)

        # Applying the trasformation: we rotate the warped image
        bb_image_warped = cv2.warpPerspective(bb_image_warped, H, (destination_width_rot, destination_height_rot))

    if visualize_final_image:
        plt.figure()
        plt.imshow(bb_image_warped, 'gray')
        plt.title('Warped bounding box')

    return bb_sorted, bb_image_warped, rotated