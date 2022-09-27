import numpy as np


############################ UTILS
def sort_bb_points(bb_points):
    """Function which sorts the bounding box points according to our standard ordering, namely upper-left -> upper-right -> 
    lower-left -> lower-right."""

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
    """Function which sorts the bounding box points differently from our standard ordering, for making the bb compliant with
    the visualization API"""
    
    bb_rot = bb_points_sorted.copy()
    bb_rot[2, :] = bb_points_sorted[3, :]
    bb_rot[3, :] = bb_points_sorted[2, :]
    return bb_rot.astype(int)