#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:47:42 2019

@author: felix
"""

if __name__=="__main__":
    
    
    import sys
    import pandas as pd 
    import json
    
    # parse the cell annotations from VIA
    annotfilepath = '/media/felix/Elements/Shankar LightSheet/Data/Cell_Segmentation/Training_Annotations/Holly/annotations/L871_Emb2_270_attempt_2_HH.csv'
    imagefolder = '/media/felix/Elements/Shankar LightSheet/Data/Cell_Segmentation/Training_Annotations/Holly/annotations/annotation_L871_Emb2'

#    labelpath = '' # read the classes from here. 
    annot = pd.read_csv(annotfilepath, sep=',')
    
    
# =============================================================================
#   Assign the annotations to individual files. 
# =============================================================================
    
    
    