#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 00:08:05 2017

@author: felix
"""

# import the necessary packages
import numpy as np
#import cv2


def image_stats(image):
    import cv2
    # compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    # return the color statistics
    return (lMean, lStd, aMean, aStd, bMean, bStd)

 
def color_transfer_stats(source, target, eps=1e-8):
    """
    Fast colour transfer using colour statistics.
        eps = a small regularization number to ensure numerical stability 
    """
    # convert the images from the RGB to L*ab* color space, being
    # sure to utilizing the floating point data type (note: OpenCV
    # expects floats to be 32-bit, so use that instead of 64-bit)
    import cv2
    
    if target.max() > 1.0000000001:
        # ensure range in [0,1.], converts to float.  
        source = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype("float32")
        target = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype("float32")
    else:
        source = cv2.cvtColor(np.uint8(255*source), cv2.COLOR_RGB2LAB).astype("float32")
        target = cv2.cvtColor(np.uint8(255*target), cv2.COLOR_RGB2LAB).astype("float32")
    
    # compute color statistics for the source and target images
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

    # subtract the means from the target image
    (l, a, b) = cv2.split(target)
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar

    # scale by the standard deviations
    l = (lStdTar )/ (lStdSrc + eps) * l
    a = (aStdTar )/ (aStdSrc + eps) * a
    b = (bStdTar )/ (bStdSrc + eps) * b


    # add in the source mean
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc
    
    # clip the pixel intensities to [0, 255] if they fall outside
    # this range
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    # merge the channels together and convert back to the RGB color
    # space, being sure to utilize the 8-bit unsigned integer data
    # type
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2RGB)
    
    # return the color transferred image
    return transfer
    

#def read_rgb(imgfile):
#    
#    import cv2
#    
#    return cv2.imread(imgfile)[:,:,::-1]

def read_rgb(imgfile):
    
    from PIL import Image
    
    img = np.array(Image.open(imgfile))
    
    if len(img.shape) < 3:
        img = np.dstack([img,img,img])
    if img.shape[-1] > 3:
        img = img[:,:,:3]
    
    return img
    
    

def match_color(source_img, target_img, mode='pca', eps=1e-8, source_mask=None, target_mask=None, ret_matrix=False):
    '''
    Matches the colour distribution of a target image to that of a source image
    using a linear transform based on matrix decomposition. 
        This effectively matches the mean and convariance of the distributions. 
    
    Images are expected to be of form (w,h,c) and can be either float in [0,1] or uint8 [0,255]
    Modes are chol, pca or sym for different choices of basis. The effect is slightly different in each case.
    
    Optionally image masks can be used to selectively bias the color transformation. 
    '''
    
    if target_img.max() > 1.0000000001:
        # ensure range in [0,1.], converts to float.  
        source_img = source_img/255.
        target_img = target_img/255
    else:
        # ensure range in [0,255.]
        source_img = source_img.astype(np.float32); 
        target_img = target_img.astype(np.float32); 
    
  
    # 1. Compute the eigenvectors of the source color distribution (possibly masked)
    if source_mask is not None:
        mu_s = np.hstack([np.mean(source_img[:,:,0][source_mask==1]), np.mean(source_img[:,:,1][source_mask==1]), np.mean(source_img[:,:,2][source_mask==1])])
    else:   
        mu_s = np.hstack([np.mean(source_img[:,:,0]), np.mean(source_img[:,:,1]), np.mean(source_img[:,:,2])])
    s = source_img - mu_s # demean
    s = s.transpose(2,0,1).reshape(3,-1) # convert to (r,g,b), 3 x n_pixels
    
    if source_mask is not None:
        s = s[:, source_mask.ravel()==1] 
    Cs = s.dot(s.T) / s.shape[1] + eps * np.eye(s.shape[0]) # 3x3 covariance matrix. 
    
    # 2. Computes the eigenvectors of the target color distribution (possibly masked)    
    if target_mask is not None:
        mu_t = np.hstack([np.mean(target_img[:,:,0][target_mask==1]), np.mean(target_img[:,:,1][target_mask==1]), np.mean(target_img[:,:,2][target_mask==1])])
    else:   
        mu_t = np.hstack([np.mean(target_img[:,:,0]), np.mean(target_img[:,:,1]), np.mean(target_img[:,:,2])])
    t = target_img - mu_t
    t = t.transpose(2,0,1).reshape(3,-1)   
    
    if target_mask is not None:
        temp = t.copy()
        t = t[:, target_mask.ravel()==1] 
    Ct = t.dot(t.T) / t.shape[1] + eps * np.eye(t.shape[0]) # 3x3 covariance matrix.  

    """
    Color match the mean and covariance of the source image. 
    """    
    if mode == 'chol':
        chol_t = np.linalg.cholesky(Ct)
        chol_s = np.linalg.cholesky(Cs)
        mix_matrix = chol_s.dot(np.linalg.inv(chol_t))
        ts = mix_matrix.dot(t)
    if mode == 'pca':
        eva_t, eve_t = np.linalg.eigh(Ct)
        Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
        eva_s, eve_s = np.linalg.eigh(Cs)
        Qs = eve_s.dot(np.sqrt(np.diag(eva_s))).dot(eve_s.T)
        mix_matrix = Qs.dot(np.linalg.inv(Qt))
        ts = mix_matrix.dot(t)
    if mode == 'sym':
        eva_t, eve_t = np.linalg.eigh(Ct)
        Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
        Qt_Cs_Qt = Qt.dot(Cs).dot(Qt)
        eva_QtCsQt, eve_QtCsQt = np.linalg.eigh(Qt_Cs_Qt)
        QtCsQt = eve_QtCsQt.dot(np.sqrt(np.diag(eva_QtCsQt))).dot(eve_QtCsQt.T)
        mix_matrix = np.linalg.inv(Qt).dot(QtCsQt).dot(np.linalg.inv(Qt))
        ts = mix_matrix.dot(t)
        
    # recover the image shape. 
    if target_mask is not None:
        matched_img_flatten = np.zeros_like(temp)
        matched_img_flatten[:,target_mask.ravel()==1] = ts.copy()
    else:
        matched_img_flatten = ts.copy()
        
    matched_img = matched_img_flatten.reshape(*target_img.transpose(2,0,1).shape).transpose(1,2,0)
    matched_img += mu_s
    
    if target_mask is not None:
        rgb_mask = np.dstack([target_mask, target_mask, target_mask])
        matched_img[rgb_mask==0] = target_img[rgb_mask==0]
    
    # clip limits. 
    matched_img[matched_img>1] = 1
    matched_img[matched_img<0] = 0
    
    if ret_matrix:
        return matched_img, mix_matrix    
    else:
        return matched_img
    
    
def recolor_w_matrix(source_img, target_img, mix_matrix, source_mask=None, target_mask=None):
    '''
    Matches the colour distribution of a target image to that of a source image
    using a linear transform based on matrix decomposition. 
        This effectively matches the mean and convariance of the distributions. 
    
    Images are expected to be of form (w,h,c) and can be either float in [0,1] or uint8 [0,255]
    Modes are chol, pca or sym for different choices of basis. The effect is slightly different in each case.
    
    Optionally image masks can be used to selectively bias the color transformation. 
    '''
    
    if target_img.max() > 1.0000000001:
        # ensure range in [0,1.], converts to float.  
        source_img = source_img/255.
        target_img = target_img/255
    else:
        # ensure range in [0,255.]
        source_img = source_img.astype(np.float32); 
        target_img = target_img.astype(np.float32); 
    
  
    # 1. Compute the eigenvectors of the source color distribution (possibly masked)
    if source_mask is not None:
        mu_s = np.hstack([np.mean(source_img[:,:,0][source_mask==1]), np.mean(source_img[:,:,1][source_mask==1]), np.mean(source_img[:,:,2][source_mask==1])])
    else:   
        mu_s = np.hstack([np.mean(source_img[:,:,0]), np.mean(source_img[:,:,1]), np.mean(source_img[:,:,2])])
    s = source_img - mu_s # demean
    s = s.transpose(2,0,1).reshape(3,-1) # convert to (r,g,b), 3 x n_pixels
    
#    if source_mask is not None:
#        s = s[:, source_mask.ravel()==1] 
#    Cs = s.dot(s.T) / s.shape[1] + eps * np.eye(s.shape[0]) # 3x3 covariance matrix. 
    
    # 2. Computes the eigenvectors of the target color distribution (possibly masked)    
    if target_mask is not None:
        mu_t = np.hstack([np.mean(target_img[:,:,0][target_mask==1]), np.mean(target_img[:,:,1][target_mask==1]), np.mean(target_img[:,:,2][target_mask==1])])
    else:   
        mu_t = np.hstack([np.mean(target_img[:,:,0]), np.mean(target_img[:,:,1]), np.mean(target_img[:,:,2])])
    t = target_img - mu_t
    t = t.transpose(2,0,1).reshape(3,-1)   
    
    temp = t.copy()
    
    # recolor the zero-meaned vector. 
    ts = mix_matrix.dot(t)
        
    
    # handles masking operations
    # recover the image shape. 
    if target_mask is not None:
        matched_img_flatten = np.zeros_like(temp)
        matched_img_flatten[:,target_mask.ravel()==1] = ts.copy()
    else:
        matched_img_flatten = ts.copy()
        
    matched_img = matched_img_flatten.reshape(*target_img.transpose(2,0,1).shape).transpose(1,2,0)
    matched_img += mu_s
    
    if target_mask is not None:
        rgb_mask = np.dstack([target_mask, target_mask, target_mask])
        matched_img[rgb_mask==0] = target_img[rgb_mask==0]
    
    # clip limits. 
    matched_img[matched_img>1] = 1
    matched_img[matched_img<0] = 0
    
    return matched_img



def calc_rgb_histogram(rgb, bins=256, I_range=(0,255)):
    '''
    computes the RGB histogram 
    returns the bins and a tuple of the histograms. 
    '''
    r_hist, bins_ = np.histogram(rgb[:,:,0].ravel(), range=I_range, bins=bins)
    g_hist, bins_ = np.histogram(rgb[:,:,1].ravel(), range=I_range, bins=bins)
    b_hist, bins_ = np.histogram(rgb[:,:,2].ravel(), range=I_range, bins=bins)
     
    return bins_, (r_hist,g_hist, b_hist)
    
    

    
    
    
    

 