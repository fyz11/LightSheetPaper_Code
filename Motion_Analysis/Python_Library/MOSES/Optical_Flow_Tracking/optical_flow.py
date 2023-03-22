# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 15:31:11 2014

@author: felix
@email: felixzhou1@gmail.com

This script and all the code within is licensed under the Ludwig License.

see README.md for details on usage.
    
"""

def farnebackflow(prev, present, params):
    r""" Computes the optical flow using Farnebacks Method

    Parameters
    ----------
    prev : numpy array
        previous frame, m x n image
    present :  numpy array
        current frame, m x n image
    params : Python dict
        a dict object to pass all algorithm parameters. Fields are the same as that in the opencv documentation, https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html. Our recommended starting values:
                
            * params['pyr_scale'] = 0.5
            * params['levels'] = 3
            * params['winsize'] = 15
            * params['iterations'] = 3
            * params['poly_n'] = 5
            * params['poly_sigma'] = 1.2
            * params['flags'] = 0
        
    Returns
    -------
    flow : finds the displacement field between frames, prev and present such that :math:`\mathrm{prev}(y,x) = \mathrm{next}(y+\mathrm{flow}(y,x)[1], x+\mathrm{flow}(y,x)[0])` where (x,y) is the cartesian coordinates of the image.
    """
    
    import numpy as np 
    import warnings
    import cv2
    # Check version of opencv installed, if not 3.0.0 then issue alert.
#    if '3.0.0' in cv2.__version__ or '3.1.0' in cv2.__version__:
        # Make the image pixels into floats.
    prev = prev.astype(np.float)
    present = present.astype(np.float)

    if '3.' in cv2.__version__:
        flow = cv2.calcOpticalFlowFarneback(prev, present, None, params['pyr_scale'], params['levels'], params['winsize'], params['iterations'], params['poly_n'], params['poly_sigma'], params['flags']) 
    elif '4.' in cv2.__version__:
        flow = cv2.calcOpticalFlowFarneback(prev, present, None, params['pyr_scale'], params['levels'], params['winsize'], params['iterations'], params['poly_n'], params['poly_sigma'], params['flags']) 
    else:
        flow = cv2.calcOpticalFlowFarneback(prev, present, pyr_scale=params['pyr_scale'], levels=params['levels'], winsize=params['winsize'], iterations=params['iterations'], poly_n=params['poly_n'], poly_sigma=params['poly_sigma'], flags=params['flags']) 
    
    return flow


def BroxFlow(prev, present):

    import pyflow
    import numpy as np 
    # what are the best params here ?
    alpha = 0.012
    ratio = 0.75
    minWidth = 5
    nOuterFPIterations = 5
    nInnerFPIterations = 1
    nSORIterations = 20
    colType = 1  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    
    u, v, _ = pyflow.coarse2fine_flow(
        prev[...,None].astype(np.float)/255., 
        present[...,None].astype(np.float)/255., 
        alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
    
    return np.dstack([u,v])

def TVL1Flow(prev, present):
    import cv2
    dtvl1=cv2.createOptFlow_DualTVL1()
    flow_DTVL1=dtvl1.calc(prev,present,None)
    return flow_DTVL1

def DeepFlow(prev, present):

    import cv2
    deepflow = cv2.optflow.createOptFlow_DeepFlow()
    flow_deepflow = deepflow.calc(prev,present,None)
    return flow_deepflow



