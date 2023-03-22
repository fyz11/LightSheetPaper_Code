# -*- coding: utf-8 -*-
"""
This is a first attempt at a CNN. 
Started by Holly Hathrell on 20 July 2016.
Last updated on: 06/02/2018 
GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python basic_CNN.py
"""

def extract_gray_video_tif(invideo):

    """
    output: imgs: n_imgs x n_rows x n_cols x n_channels stack.
    """    
    from PIL import Image
    import numpy as np     
    img = Image.open(invideo)    
    imgs = []
    read = True
    frame = 0    
    while read:
        try:
            img.seek(frame) # select this as the image
            imgs.append(np.array(img)[None,:])
            frame += 1
        except EOFError:
            # Not enough frames in img
            break        
    return np.concatenate(imgs, axis=0)


def fetch_train_files(all_file_list, slices, key='.mat'):
    
    import os 
    
    slice_nums = []
    want_files = []
    
    for f in all_file_list:
        base = (f.split('/')[-1]).split(key)[0]
        base_splits = base.split('_')
        for b in base_splits:
            if 'slice' in b:
                slice_num = int(b.split('-')[1])
                if slice_num in slices:
                    want_files.append(f)
                slice_nums.append(slice_num)
                break
            
    slice_nums = np.hstack(slice_nums)
    
    return want_files, slice_nums
    
    
def detect_all_files(infolder, key='.mat'):
    
    import os 
    
    all_files = os.listdir(infolder)
    
    files = []
    
    for f in all_files:
        if key in f:
            files.append(os.path.join(infolder, f))
            
    return np.hstack(files)


def random_crops(img, delta_y, delta_x, N = 100, final_size=(256,256)):
    
    from skimage.transform import resize
    
    # Note: image_data_format is 'channel_last'
#    assert img.shape[2] == 3
    nrows, ncols = img.shape[:2] 
    min_size=np.min([nrows, ncols])
    height, width = delta_y, delta_x
    
    if len(delta_y) > 1: 
        dy = np.random.randint(np.max([height[0], 100]), np.min([height[1], int(.9*min_size)]), N)
        dx = np.random.randint(np.max([width[0], 100]), np.min([width[1], int(.9*min_size)]), N)
#        print(dx)
    else:
        dy = height[0]
        dx = width[0]
#    print(height, width)
#    print(dx)
#    print(dy)
    crops = []
    
    if len(delta_y) > 1:
        for ii in range(len(dx)):
            # this is to fit within the image bounds.
            x = np.random.randint(0, img.shape[1] - dx[ii] + 1)
            y = np.random.randint(0, img.shape[0] - dy[ii] + 1)
            crop = img[y:(y+dy[ii]), x:(x+dx[ii]), :]
            crops.append( resize(crop, output_shape=final_size, preserve_range=True))
    else:
        for ii in range(N):
            # this is to fit within the image bounds.
            x = np.random.randint(0, img.shape[1] - dx + 1)
            y = np.random.randint(0, img.shape[0] - dy + 1)
            crop = img[y:(y+dy), x:(x+dx), :]
            crops.append( resize(crop, output_shape=final_size, preserve_range=True))
    return np.array(crops)


def masked_mse(mask_value):
    import keras.backend as K 
    def f(y_true, y_pred):
        mask_true = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        masked_squared_error = K.square(mask_true * (y_true - y_pred))
        # in case mask_true is 0 everywhere, the error would be nan, therefore divide by at least 1
        # this doesn't change anything as where sum(mask_true)==0, sum(masked_squared_error)==0 as well
        masked_mse = K.sum(masked_squared_error, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1)
        return masked_mse
    return f

#import cv2 # Installed using 'conda install -c https://conda.binstar.org/menpo opencv3'
import os    

#os.environ['THEANO_FLAGS'] = "device=gpu1"  
  
#import theano
#theano.config.floatX = 'float32'
import numpy as np
import keras
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, BatchNormalization, Activation, Add
from keras.layers.merge import Concatenate
from keras.layers.pooling import AveragePooling2D
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tifffile import imsave as tifimsave
from skimage.io import imsave, imread
from skimage import img_as_ubyte
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_otsu

import scipy.io as spio
from my_generator_test import image_generator
print(keras.__version__)
seed = 1000
np.random.seed(seed)    # robust runs that should have the same result for the same dataset

import glob 
#==============================================================================
# Set the folder locations and make the directory
#date = '2017_06_02/Test1/'  # Month the day to order chronologially in folders
nb_epochs=100


"""
load the image and annotation files. (use several different folders.)
"""
annotfolders = [ '/media/felix/Elements/Shankar LightSheet/Data/Cell_Segmentation/Training_Annotations/Holly/annotations/annotation_L491_Emb2_masks-2', 
                '/media/felix/Elements/Shankar LightSheet/Data/Cell_Segmentation/Training_Annotations/Holly/annotations/annotation_L455_Emb3_masks-2',
                '/media/felix/Elements/Shankar LightSheet/Data/Cell_Segmentation/Training_Annotations/Holly/annotations/annotation_L505_Emb2_masks-2',
                '/media/felix/Elements/Shankar LightSheet/Data/Cell_Segmentation/Training_Annotations/Holly/annotations/annotation_L864_Emb2_masks-2',
                '/media/felix/Elements/Shankar LightSheet/Data/Cell_Segmentation/Training_Annotations/Holly/annotations/annotation_L871_Emb1_masks-2',
                '/media/felix/Elements/Shankar LightSheet/Data/Cell_Segmentation/Training_Annotations/Holly/annotations/annotation_L455_Emb2_masks-2'] 

# parse the time points and the image file from annot files
imgs_train = []
masks_train = []
##annotfolder = '/media/felix/Elements/Shankar LightSheet/Data/Cell_Segmentation/Training_Annotations/Holly/annotations/annotation_L491_Emb2_masks'
#annotfolder = '/media/felix/Elements/Shankar LightSheet/Data/Cell_Segmentation/Training_Annotations/Holly/annotations/annotation_L491_Emb2_masks-2'
imgfolder = '/media/felix/Elements/Shankar LightSheet/Data/Cell_Segmentation/Training_Annotations/Holly/annotations/tiffs'
#annotfiles = glob.glob(os.path.join(annotfolder, '*.tif'))

for annotfolder in annotfolders[:]:
    annotfiles = glob.glob(os.path.join(annotfolder, '*.tif'))
    for a in annotfiles[:]:
        imgfname = '_'.join(os.path.split(a)[-1].replace('_bleachcorr', '').split('_')[:-1]) + '.tif'
        
        if 'L491' in imgfname: 
            tp = int(os.path.split(a)[-1].split('_')[-1].split('.tif')[0])
        else:
            tp = int(os.path.split(a)[-1].split('_')[-1].split('.tif')[0]) - 1
        
        vidfile = os.path.join(imgfolder, imgfname)
        vid = extract_gray_video_tif(vidfile)
        vidslice = vid[tp-1] # am starting to think not the tp.... 
        
        imgs_train.append(vidslice)
        masks_train.append(imread(a).transpose(1,2,0))
        
# filter for only masks that contain divisions!
val_images = np.ones(len(masks_train)) > 0 
#val_images = np.hstack([(np.sum(mask[...,0]>0)>0) for mask in masks_train])

# select only the valid images. 
imgs_train = [imgs_train[i] for i in range(len(imgs_train)) if val_images[i]==True]
masks_train = [masks_train[i] for i in range(len(masks_train)) if val_images[i]==True]
   
"""
load a second set for validation of loss. 
"""
# take another dataset as the val dataset.
annotfolder = '/media/felix/Elements/Shankar LightSheet/Data/Cell_Segmentation/Training_Annotations/Holly/annotations/annotation_L871_Emb2_masks-2/'
imgfolder = '/media/felix/Elements/Shankar LightSheet/Data/Cell_Segmentation/Training_Annotations/Holly/annotations/tiffs'
annotfiles = glob.glob(os.path.join(annotfolder, '*.tif'))

# parse the time points and the image file from annot files
imgs_test = []
masks_test = []

for a in annotfiles[:]:
    imgfname = '_'.join(os.path.split(a)[-1].replace('_bleachcorr', '').split('_')[:-1]) + '.tif'
    
    if 'L491' in imgfname: 
        tp = int(os.path.split(a)[-1].split('_')[-1].split('.tif')[0])
    else:
        tp = int(os.path.split(a)[-1].split('_')[-1].split('.tif')[0]) - 1 # decrease the index by 1. 
    
    vidfile = os.path.join(imgfolder, imgfname)
    vid = extract_gray_video_tif(vidfile)
    vidslice = vid[tp]
    
    imgs_test.append(vidslice)
    masks_test.append(imread(a).transpose(1,2,0))

# filter for only masks that contain divisions!
val_images = np.ones(len(masks_test)) > 0 
#val_images = np.hstack([(np.sum(mask[...,0]>0)>0) for mask in masks_train])

# select only the valid images. 
imgs_test = [imgs_test[i] for i in range(len(imgs_test)) if val_images[i]==True]
masks_test = [masks_test[i] for i in range(len(masks_test)) if val_images[i]==True]

# =============================================================================
#   Create a training dataset by building random crops. 
# =============================================================================
# implement random crops of the data for training -> none for testing? 
"""
Prebuild the patches for segmentation 
"""
train_data = np.concatenate([random_crops(np.dstack([imgs_train[i], masks_train[i]]), delta_y=[128, 560], delta_x=[128, 560], 
                                          N = 100, final_size=(128,128)) for i in range(len(imgs_train))], axis=0)
    
#train_data = np.array([t for t in train_data if np.sum(t[:,:,1])>0.1 ]) -> this did  the filter. 
    
# split the masks
test_data = np.concatenate([random_crops(np.dstack([imgs_test[i], masks_test[i]]), delta_y=[128, 560], delta_x=[128, 560], 
                                          N = 100, final_size=(128,128)) for i in range(len(imgs_test))], axis=0)
#test_data = np.array([t for t in test_data if np.sum(t[:,:,1])>0.1 ])-> this was a filter. 
    
# save this.
#import scipy.io as spio
#
#spio.savemat('train_attempt1_unet.mat', {'X':train_data,
#                                         'Y':test_data})
#    
train_x = train_data[...,0]/255. # np.array([rescale_intensity(t/255.) for t in train_data[...,0]])
#train_y = np.array([rescale_intensity(t*1.) for t in train_data[...,1]]) # just divisions for now
train_y = np.array([rescale_intensity(t*1.)[...,-1] for t in train_data[:]]) # is there something wrong here? 
#train_y = np.array([(t>threshold_otsu(t) )*1 for t in train_y[:]])
#test_x = test_data[...,0] / 255.

test_x = test_data[...,0]/255.# np.array([rescale_intensity(t/255.) for t in test_data[...,0]])
#test_y = np.array([rescale_intensity(t*1.) for t in test_data[...,1]]) # just divisions for now
test_y = np.array([rescale_intensity(t*1.)[...,-1] for t in test_data[:]])
#test_y = np.array([(t>threshold_otsu(t) )*1 for t in test_y[:]])


# save the data.
import scipy.io as spio 

print('saving')
savefile = '/media/felix/Elements/Shankar LightSheet/Data/Cell_Segmentation/Training_Annotations/Holly/annotations/training_seg-128x128.mat'
spio.savemat(savefile, {'train_x': np.uint8(255*train_x), 
                        'train_y': np.uint8(255*train_y), 
                        'test_x': np.uint8(255*test_x), 
                        'test_y': np.uint8(255*test_y)})
    
    
    



