import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils
import math
import matplotlib.patches as patches
import time

def find_outliers(v, level=0.02):
    q1, q3 = tuple(np.quantile(v, [level,1-level]))
    IQR = q3-q1
    outliers_mask = np.logical_or(v>q3+IQR, v<q1-IQR)
    """print(outliers_mask)
    print(v[outliers_mask])"""
    if v[outliers_mask].size==0:
        return None
    return np.argmax([v[i]-q3-IQR if (outliers_mask[i] and v[i]-q3-IQR>0) else abs(v[i]-q1+IQR) if outliers_mask[i] else 0  for i in range(len(v))])


def find_wrong_bar(bars_start, bars_width, bars_halfHeightUp, bars_halfHeightDown, level=0.02):
    n_bars = len(bars_start)
    bars_height = np.array([bars_halfHeightUp[i]+bars_halfHeightDown[i]+1 for i in range(n_bars)])
    #print(bars_height)
    wrong_bars_height_index = find_outliers(bars_height, level=0.02)
    
    bars_area = np.array([bars_height[i]+bars_width[i] for i in range(n_bars)])
    wrong_bars_area_index = find_outliers(bars_area, level=0.02)
    
    if wrong_bars_area_index is None or wrong_bars_height_index is None or wrong_bars_area_index!=wrong_bars_height_index:
        return None
    
    return wrong_bars_area_index