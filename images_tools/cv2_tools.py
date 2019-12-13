"""
Module for cv2 utility functions and maintaining version compatibility
between 3.x and 4.x
"""
import cv2
import os, random


def findContours(*args, **kwargs):
    """
    Wraps cv2.findContours to maintain compatiblity between versions
    3 and 4

    Returns:
        contours, hierarchy
    """
    if cv2.__version__.startswith('4'):
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    elif cv2.__version__.startswith('3'):
        _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    else:
        raise AssertionError(
            'cv2 must be either version 3 or 4 to call this method')

    return contours, hierarchy

def rename(srcfilename, dstfilename, saveFlag=1):
    srcmat = cv2.imread(srcfilename)
    if saveFlag:
        cv2.imwrite(dstfilename, srcmat)

