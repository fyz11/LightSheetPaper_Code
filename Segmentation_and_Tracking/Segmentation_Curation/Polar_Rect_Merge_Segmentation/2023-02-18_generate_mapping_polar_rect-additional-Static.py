# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 23:18:07 2020

@author: felix
"""

def map_between_coords_ij(coord_map_1, coord_map_2, valid_dist_thresh=5):
    
    # first is the ref map. 
    from sklearn.neighbors import NearestNeighbors
    
    nbrs_model = NearestNeighbors(n_neighbors=1, algorithm='auto')
    nbrs_model.fit(coord_map_1.reshape(-1,3))
        
    neighbours_rect_polar_id = nbrs_model.kneighbors(coord_map_2.reshape(-1,3))
    val_dist_img = neighbours_rect_polar_id[1].reshape(coord_map_2.shape[:2])
    val_dist_img_binary = neighbours_rect_polar_id[0].reshape(coord_map_2.shape[:2]) <= valid_dist_thresh
    val_dist_ijs = np.unravel_index(val_dist_img, coord_map_1.shape[:2] )

    return val_dist_ijs, val_dist_img_binary
    


def mkdir(directory):
    """ check if directory exists and create it through Python if it does not yet.

    Parameters
    ----------
    directory : str
        the directory path (absolute or relative) you wish to create (or check that it exists)

    Returns
    -------
        void function, no return
    """
    import os 
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    return []
    
    
def find_embryo_folders(infolder, key):
    
    import os 
    
    folders = os.listdir(infolder)
    folders_found = []
    
    for ff in folders:
        if key in ff:
            folders_found.append(os.path.join(infolder,ff))
    
    if len(folders_found)>0:
        folders_found = np.hstack(folders_found)
        
    return folders_found
    

if __name__=="__main__":
    
    import numpy as np 
    import pylab as plt 
    import scipy.io as spio
    import glob 
    import os 
    import pandas as pd 
    from skimage.io import imread
    from skimage.filters import gaussian
    
    
#    unwrapparamsfolder = '/media/felix/Srinivas4/LifeAct/L401_1View/unwrapped/unwrap_params_geodesic-rotmatrix'
#    unwrapparamsfolder = '/media/felix/Srinivas4/LifeAct/L455_Emb2_TP1-90/unwrapped/unwrap_params_geodesic-rotmatrix'
#    unwrapparamsfolder = '/media/felix/Srinivas4/LifeAct/L455_Emb3_TP1-111/unwrapped/unwrap_params_geodesic-rotmatrix'
#    unwrapparamsfolder = '/media/felix/Srinivas4/LifeAct/L491_Emb2_TP1-100/unwrapped/unwrap_params_geodesic-rotmatrix'
#    unwrapparamsfolder = '/media/felix/Srinivas4/LifeAct/L505_Emb2_TP1-110/unwrapped/unwrap_params_geodesic-rotmatrix'
#    unwrapparamsfolder = '/media/felix/Srinivas4/LifeAct/L864_Emb1_TP1-115/unwrapped/unwrap_params_geodesic-rotmatrix'
#    unwrapparamsfolder = '/media/felix/Srinivas4/LifeAct/L871_Emb1_TP1-66/unwrapped/unwrap_params_geodesic-rotmatrix'
#    unwrapparamsfolder = '/media/felix/Srinivas4/LifeAct/L871_Emb2_TP1-66/unwrapped/unwrap_params_geodesic-rotmatrix'
    # unwrapparamsfolder = '/media/felix/Srinivas4/LifeAct/L917_TP1-120/unwrapped/unwrap_params_geodesic-rotmatrix'
    
    # rootfolder = 'F:\\Shankar-2\\Control_Embryos_Lefty'
    
    """
    Additional static 
    """
    # rootfolder = r'F:\Work\Research\Shankar\Additional_Fixed_Datasets\Addtional_Confocal_Data_for_Fixed_Paper\A_PI DAPI and Actin only'
    rootfolder = r'F:\Work\Research\Shankar\Additional_Fixed_Datasets\Addtional_Confocal_Data_for_Fixed_Paper\B_E5_5 YAP staining'
    # rootfolder = r'F:\Work\Research\Shankar\Additional_Fixed_Datasets\Addtional_Confocal_Data_for_Fixed_Paper\C_Pre_Induction YAP Staining'
    # rootfolder = r'F:\Work\Research\Shankar\Additional_Fixed_Datasets\Addtional_Confocal_Data_for_Fixed_Paper\C_Pre_Induction YAP Staining'
    
    allfolders = find_embryo_folders(rootfolder, key='_reorient_gui')
    allfolders = np.sort(allfolders)
    
    
    for folder in allfolders:
        
        unwrapparamsfolder = os.path.join(folder, 'unwrapped')
        # unwrapparamsfolder = 'F:\Shankar-2\Control_Embryos_Nodal\N1_L477_Emb5_isotropic_reorient_gui\unwrapped'
        
        unwrapmatfiles = glob.glob(os.path.join(unwrapparamsfolder, '*.mat'))
        print(unwrapmatfiles)
        
        
        # saveparamsfolder = '/media/felix/My Passport/Shankar-2/polar_rect_transformations/LifeAct'
        # savefolder = os.path.join(saveparamsfolder, unwrapparamsfolder.split('/')[5])
        savefolder = os.path.join(folder, 'unwrapped', 'polar_to_rect_tfs')
        mkdir(savefolder)
        print(savefolder)
        
        
        for ii in range(len(unwrapmatfiles))[:]:
            
            unwrapmatfile = unwrapmatfiles[ii]
            rect_params = spio.loadmat(unwrapmatfile)['ref_map_rect_xyz'][::-1] # flip the y-axis.
            polar_params = spio.loadmat(unwrapmatfile)['ref_map_polar_xyz']
            
    #        rect_params = gaussian(rect_params, sigma=3, preserve_range=True)
    #        polar_params = gaussian(polar_params, sigma=1, preserve_range=True)
            rect_to_polar_mapping, rect_to_polar_mapping_valid = map_between_coords_ij(polar_params, 
                                                                                        rect_params, valid_dist_thresh=5)
            
            polar_to_rect_mapping, polar_to_rect_mapping_valid = map_between_coords_ij(rect_params, 
                                                                                        polar_params, valid_dist_thresh=5)
    #        savematfile = os.path.join(savefolder, 'polar_rect_mapping_' + os.path.split(unwrapmatfile)[-1])
    #        spio.savemat(savematfile, {'rect_to_polar_mapping': rect_to_polar_mapping,
    #                                   'polar_to_rect_mapping': polar_to_rect_mapping})
            
            savecsvfile_rect_polar_y = os.path.join(savefolder, 'rect_polar_mapping_y_' + os.path.split(unwrapmatfile)[-1].replace('.mat', '.csv'))
            pd.DataFrame(rect_to_polar_mapping[0], index=None, columns=None).to_csv(savecsvfile_rect_polar_y, index=None)
            savecsvfile_rect_polar_x = os.path.join(savefolder, 'rect_polar_mapping_x_' + os.path.split(unwrapmatfile)[-1].replace('.mat', '.csv'))
            pd.DataFrame(rect_to_polar_mapping[1], index=None, columns=None).to_csv(savecsvfile_rect_polar_x, index=None)   
            
            savecsvfile_polar_rect_y = os.path.join(savefolder, 'polar_rect_mapping_y_' + os.path.split(unwrapmatfile)[-1].replace('.mat', '.csv'))
            pd.DataFrame(polar_to_rect_mapping[0], index=None, columns=None).to_csv(savecsvfile_polar_rect_y, index=None)
            savecsvfile_polar_rect_x = os.path.join(savefolder, 'polar_rect_mapping_x_' + os.path.split(unwrapmatfile)[-1].replace('.mat', '.csv'))
            pd.DataFrame(polar_to_rect_mapping[1], index=None, columns=None).to_csv(savecsvfile_polar_rect_x, index=None)   
        
    
    
        
        
    
    
    
    
        # compute the opposite. 
#        rect_to_polar_mapping
#        rect_ve_img = imread('C:\\Users\\felix\\Documents\\PostDoc\\Shankar\\Cell_seg_masks_parsing_tests\\prob_mask_img-L491_ve_000_unwrap_params_rect.tif')[0]
#        
#        
#        XX, YY = np.meshgrid(range(rect_params.shape[1]), 
#                             range(rect_params.shape[0]))
#        
#        polar_rect_ve_img = np.zeros(polar_params.shape[:-1])
#        polar_i = rect_to_polar_mapping[0][YY,XX]
#        polar_j = rect_to_polar_mapping[1][YY,XX]
#        
#        polar_rect_ve_img[polar_i, 
#                          polar_j] = rect_ve_img
                          
#        
#        
#        fig, ax = plt.subplots(figsize=(15,15))
#        ax.imshow(polar_rect_ve_img)
#        plt.show()
#        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        