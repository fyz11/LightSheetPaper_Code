# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 15:31:11 2014

@author: felix

Description:
    Evaluates the dense optical flow between two consecutive image frames for the first image
    
Usage:

    
"""

def Eval_dense_optic_flow(prev, present, params):
    
    """
    Input:
        prev: previous frame, m x n image
        present: current frame, m x n image
        params: a dict object to pass all algorithm parameters.
            fields are the same as that in opencv docs:
            recommended starting values:

            # Flow Options:
            # default for Brox.
            params['alpha'] = 0.012
            ratio = 0.75
            minWidth = 20
            nOuterFPIterations = 7
            nInnerFPIterations = 1
            nSORIterations = 30
            colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
            
        
    Output:
        flow: finds the displacement field between frames, prev and present such that
                prev(y,x) = next(y+flow(y,x)[1], x+flow(y,x)[0])
        where (x,y) is the cartesian coordinates of the image.
    """
    
    import numpy as np 
    import warnings
    import cv2
    import pyflow

    # Check version of opencv installed, if not 3.0.0 then issue alert.
#    if '3.0.0' in cv2.__version__ or '3.1.0' in cv2.__version__:
        # Make the image pixels into floats.
    prev = prev.astype(np.float)
    present = present.astype(np.float)
    
    prev = np.ascontiguousarray(prev)
    present = np.ascontiguousarray(present)
#    if prev.max() > 1.1: # assume 255.
#        prev = prev/255.
#    if present.max() > 1.1:
#        present = present/255.
        
    alpha = params['alpha']
    ratio = params['ratio']
    minWidth = params['minWidth']
    nOuterFPIterations = params['nOuterFPIterations']
    nInnerFPIterations = params['nInnerFPIterations']
    nSORIterations = params['nSORIterations']
    colType = params['colType']

    u, v, im2W = pyflow.coarse2fine_flow(
    prev, present, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
    nSORIterations, colType)

    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    
    return flow
