# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 17:25:13 2020

@author: felix
"""
import numpy as np 

def extract_centroids(binary, min_area=20, max_area=10000):
    
    from skimage.measure import label, regionprops
    
    labelled = label(binary, connectivity=1) # restrict this. 
    labelled = filter_min_max_areas(labelled, min_thresh=min_area, max_thresh=max_area)
    
    regions = regionprops(labelled)
    
    centroids = np.vstack([re.centroid for re in regions])
    
    return labelled, centroids


def iou_mask(mask1, mask2):
    
    intersection = np.sum(np.abs(mask1*mask2))
    union = np.sum(mask1) + np.sum(mask2) - intersection # this is the union area. 
    
    overlap = intersection / float(union + 1e-8)
    
    return overlap 


def create_spixel_mask(tra_frame, cell_segs_time):

    time, spixels = tra_frame
    cell_segs_frame = cell_segs_time[time]
    
    mask = np.zeros(cell_segs_frame.shape, dtype=np.bool)
    for ss in spixels:
        mask[cell_segs_frame == ss] = 1
        
    return mask
    

def color_cell_segmentation(img, cells, cell_ids = None):
    from skimage.segmentation import mark_boundaries
    
    baseimg = np.dstack([img,img,img])
    
    if cell_ids is None:
        pass
    else:
        for c in cell_ids:
            baseimg[cells==c] = np.uint8(np.random.rand(3)*255)[None,:] # random color.
    
    baseimg = mark_boundaries(baseimg, cells)
    
    return baseimg
            
def warp_segmentation(seg, flow):
    
    import numpy as np
    import cv2
    height, width = seg.shape[:2]
    R2 = np.dstack(np.meshgrid(np.arange(width), np.arange(height)))
    
    # remapping and interpolation 
    pixel_map = R2 + flow
    pixel_map = np.array(pixel_map, dtype=np.float32)
    
    new_seg = cv2.remap(np.float32(seg), pixel_map[...,0], 
                                         pixel_map[...,1], interpolation=cv2.INTER_LINEAR)

    return np.uint16(new_seg)

def create_seg_mask(labelled, ids):
    
    mask = np.zeros(labelled.shape, dtype=np.bool)
    for ss in ids:
        mask[labelled == ss] = 1
        
    return mask

def warp_seg_binary(binary, flow):
    
    pts = np.array(np.where(binary>0)).T
    pts_ = pts + flow[pts[:,0].astype(np.int), pts[:,1].astype(np.int)][:,::-1]
    
    new_mask = np.zeros_like(binary)
    new_mask[pts_[:,0].astype(np.int), pts_[:,1].astype(np.int)] = 1
    
    return new_mask

def get_bbox_region(mask, img=None):
    
    YY_ind, XX_ind = np.indices(mask.shape)
    start_y, start_x = YY_ind[mask], XX_ind[mask]
    
    bbox = np.hstack([np.min(start_y), np.min(start_x), np.max(start_y), np.max(start_x)]).astype(np.int)
    patch = []
    if img is not None:
        patch = img[bbox[0]:bbox[2], bbox[1]:bbox[3]].copy()
    
    return bbox, patch

def multi_hypothesis_tracking(track, next_ids, cell_link_lists, cell_link_lists_coverage, cell_link_lists_coverage_iou, cells, flow, vid, min_iou=0.15, lookuptime=1, patch_size=32, accept_score=0.5, lam=0.75):
    
    from densesift import SingleSiftExtractor
    from skimage.transform import resize
    from skimage.measure import regionprops
    
    # create sift extractor.
    siftextractor = SingleSiftExtractor(patch_size)

    start_pt = track[-1]
    start_time = start_pt[0]
    start_spixel = start_pt[1]
    max_time = len(cells)
    
    start_mask = create_seg_mask(cells[start_time], start_spixel)
#    start_mask_bbox = regionprops(start_mask.astype(np.int))[0].bbox #min_row, min_col, max_row, max_col
    start_mask_bbox, start_mask_image = get_bbox_region(start_mask, img=vid[start_time])

    start_feat = siftextractor.process_image(resize(start_mask_image, (patch_size, patch_size)))[0]
    start_feat = start_feat / np.linalg.norm(start_feat,2)
#    print(start_time)
#    plt.plot(start_feat)
#    plt.figure()
#    plt.imshow(start_mask)
#    plt.plot([start_mask_bbox[1], start_mask_bbox[1], start_mask_bbox[3], start_mask_bbox[3], start_mask_bbox[1]],
#             [start_mask_bbox[0], start_mask_bbox[2], start_mask_bbox[2], start_mask_bbox[0], start_mask_bbox[0]])
#    plt.show()
#    
#    plt.figure()
#    plt.imshow(start_mask_image)
#    plt.show()
    
    #############################################################################################
    # construct initial hypotheses based on the data. 
    #############################################################################################
    hypotheses_0 = [[[start_time+1, [ss]]] for ss in next_ids] # lets grow these hypotheses and see how many future tracks are possible. 
    hypotheses_quals_0 = [0 for ii in range(len(next_ids))] # as many as there are hypotheses. 
    
    upper_time = np.minimum( start_time+1+lookuptime, max_time-1)
    n_tps = upper_time - (start_time+1)
#    print(n_tps)
    hypotheses = list(hypotheses_0)
    hypotheses_quals = list(hypotheses_quals_0)
    
    # propagate the flow mask for the required number of frames. # warp segmentation is preferred for its accuracy. 
    # this compound process severly warps border cells? 
    prop_mask = [warp_segmentation(start_mask, 
                                   -np.sum(flow[start_time:start_time+tt+1], axis=0)) > 0.5  for tt in range(n_tps+1)]
                        
    # the prop_mask is actually accurate -> if no pixels is availabel any more -> that means it was lost!.
    prop_mask_pixels_sum = np.hstack([np.sum(pp) for pp in prop_mask])
    n_absent = np.sum(prop_mask_pixels_sum<=10)
#    print(n_tps, len(prop_mask_pixels_sum), n_absent)
    n_tps = n_tps - n_absent # account for absent time points.
#    print(n_tps)
    
    if n_tps <= 0:
        
        # cannot extend!. 
        track_new = list(track)
        track_new.append([]) # termination 
        return track_new
    
    else:
    
    #    for zz, pp in enumerate(prop_mask):
    #        plt.figure()
    #        plt.imshow(pp)
    #        plt.show()
             
        for iter_ in range(n_tps):
            frame = start_time + 1 + iter_
            
            hypotheses_new = []
            hypotheses_quals_new = []
    
            n_hypotheses = len(hypotheses)
            for kk in range(n_hypotheses):
                
                hypotheses_kk = hypotheses[kk]
                last = hypotheses_kk[-1] # get the last timepoint. 
                last_qual = hypotheses_quals[kk]
                
                if len(last) > 0:
                    last_c = last[1]
                    cc = last_c[0]
                    next_cc = cell_link_lists[frame][cc]
    
                    # only one of the following should be met....... 
                    if len(next_cc) == 1:
    #                    print('appending')
                        new_c = list(hypotheses_kk)
                        new_c.append([frame+1, next_cc])
                        hypotheses_new.append(new_c)
                        hypotheses_quals_new.append(np.hstack([last_qual, 1]))
                        
                    elif len(next_cc) == 0: 
    #                    print('terminating')
                        new_c = list(hypotheses_kk)
                        new_c.append([])
                        hypotheses_new.append(new_c) # terminate. 
                        hypotheses_quals_new.append(np.hstack([last_qual, 1]))
    #                    print('searching new')
    #                    potential_next = cell_link_lists_coverage[frame][cc]
    #                    potential_next_iou = cell_link_lists_coverage_iou[frame][cc]
    #                    
    #                    if len(potential_next) > 0:
    #                        potential_next = potential_next[potential_next_iou>=min_iou]
    #                        
    #                        if len(potential_next) > 0:
    #                            for nn_cc in potential_next:
    #                                new_c = list(hypotheses_kk)
    #                                new_c.append([frame+1, [nn_cc]])
    #                                hypotheses_new.append(new_c)
    #                                hypotheses_quals_new.append(np.hstack([last_qual, 0]))
    #                        else:
    #                            print('terminating')
    #                            new_c = list(hypotheses_kk)
    #                            new_c.append([])
    #                            hypotheses_new.append(new_c) # terminate. 
    #                            hypotheses_quals_new.append(np.hstack([last_qual, 1]))
                    if len(next_cc) > 1:
                        for nn_cc in next_cc:
                            new_c = list(hypotheses_kk)
                            new_c.append([frame+1, [nn_cc]])
                            hypotheses_new.append(new_c)
                            hypotheses_quals_new.append(np.hstack([last_qual, 1]))
                else:
                    hypotheses_new.append(hypotheses_kk)
                    hypotheses_quals_new.append(hypotheses_quals[kk])
                            
    #                        print(frame, len(hypotheses_new))
    #        print('--------------')
    #                        if len(hypotheses_new) == 0: # shouldnt' ever be the case!. 
    #                            print(frame)
            hypotheses_quals = list(hypotheses_quals_new)
            hypotheses = list(hypotheses_new)
    
        if len(hypotheses) == 0:
            track_new = list(track)
            track_new.append([]) # termination 
            return track_new
        
        else:
        # =============================================================================
        #     score the final hypotheses, keep those the best and backtrack to create the most possible track. 
        # =============================================================================
        #    print(hypotheses)
        
        
            # very weird bug is occurring with multihypothesis tracking. 
            spixels_assign_track_dict = [{} for tp in range(n_tps+1)] # this will specify how each spixel has been linked from previous points.
                
            hypothesis_scores = []
            hypothesis_scores_track = []
        #    appearance_scores = []
        #    size_scores = []
        #    avg_scores = []
            for hh_k, hh in enumerate(hypotheses):
        #        print(hh)
                hh_scores = []
        #        aa_scores = []
        #        ss_scores = []
        #        av_scores = []
                for pt_ii, pt in enumerate(hh):
                    if len(pt) > 0:
                        pt_time = pt[0]
                        pt_reg= pt[1]
                        
                        existing_ids = spixels_assign_track_dict[pt_ii].keys()
                        for reg in pt_reg:
                            if reg not in existing_ids:
                                spixels_assign_track_dict[pt_ii][reg] = [hh_k]
                            else:
                                spixels_assign_track_dict[pt_ii][reg].append(hh_k)
                        
                        # compute the IoU score.
                        pt_mask = create_seg_mask(cells[pt_time], pt_reg)
                        pt_mask_score = iou_mask(prop_mask[pt[0] - (start_time+1)], pt_mask)
        #                hh_scores.append(pt_mask_score)
                        
                        # compute appearance score:
                        pt_mask_image = get_bbox_region(pt_mask, img=vid[pt[0]])[1]
                        prop_mask_image = get_bbox_region(prop_mask[pt[0] - (start_time+1)], img=vid[pt[0]])[1]
                        
                        prop_feat = siftextractor.process_image(resize(prop_mask_image, (patch_size, patch_size)))[0]; prop_feat = prop_feat / np.linalg.norm(prop_feat,2)
                        pt_feat = siftextractor.process_image(resize(pt_mask_image, (patch_size, patch_size)))[0]; pt_feat = pt_feat / np.linalg.norm(pt_feat,2)
                        
                        pt_mask_appearance_score = np.dot(prop_feat, pt_feat)
                        
                        hh_score = np.clip( lam * pt_mask_score + (1-lam) * pt_mask_appearance_score, 0 , 1)
                        hh_scores.append(hh_score)
                        
        #                aa_scores.append(np.dot(prop_feat, pt_feat))
        #                size_scores.append(np.sum(pt_mask)/ float(np.sum(prop_mask[pt[0] - (start_time+1)] + 1e-8)))
        #                av_scores.append(np.mean([pt_mask_score, np.dot(prop_feat, pt_feat), 
        #                                          1 - np.abs(np.sum(pt_mask) - np.sum(prop_mask[pt[0] - (start_time+1)]))/ float(np.sum(prop_mask[pt[0] - (start_time+1)] + 1e-8))]))
                hypothesis_scores.append(hh_scores)
                hypothesis_scores_track.append(np.max(hh_scores))
        #        appearance_scores.append(aa_scores)
        #        size_scores.append(ss_scores)
        #        avg_scores.append(av_scores)
                
        #    # final hypotheses..... 
        #    print(hypotheses)
        #    
        ##    print('++++++++++++++++')
        ##    print(hypotheses_quals)
        #    print('++++++++++++++++')
        #    print(hypothesis_scores)
        #    print('++++++++++++++++')
        #    print(spixels_assign_track_dict)
        #        
        # =============================================================================
        #     Evaluate the tracks
        # =============================================================================
            valid_hypotheses_select = np.hstack(hypothesis_scores_track) >= accept_score
            
            if np.sum(valid_hypotheses_select) > 0:
                valid_hypotheses_ids = np.arange(len(hypotheses))[valid_hypotheses_select] 
                
                if len(valid_hypotheses_ids) == 1: 
                    # there is only the one possiblity!
                    track_new = track + hypotheses[valid_hypotheses_ids[0]]
                    return track_new
                else:
                    print('building best tracklet after MHT')
                    valid_hypotheses_scores = np.hstack(hypothesis_scores_track)[valid_hypotheses_ids]
                    valid_hypotheses = [hypotheses[hyp_ii] for hyp_ii in valid_hypotheses_ids]
                    
                    # choose the best one and go from the last valid point building the final track. 
                    best_track = valid_hypotheses[np.argmax(valid_hypotheses_scores)] # we now use this construct..... the remainder
                    used_hyp_ids = [valid_hypotheses_ids[np.argmax(valid_hypotheses_scores)]]
                    
                    for tt in range(len(best_track)-1, -1, -1):
                        # iterating back in time
                        tra = best_track[tt]
                        if len(tra) > 0:
                            tra_tp = tra[0] # don't really need this.
                            tra_reg = tra[1]
                            
                            for reg in tra_reg:    
                                reg_tra_ids = spixels_assign_track_dict[tt][reg]
                                for tr in reg_tra_ids:
                                    if tr in used_hyp_ids:
                                        pass
                                    else:
                                        used_hyp_ids.append(tr)
                                        # modify best track inplace. 
                                        other_track = hypotheses[tr]
                                        for xx in range(len(other_track)):
                                            best_track[xx][1] += other_track[xx][1] 
                                            best_track[xx][1] = list(np.unique(best_track[xx][1]))
                    
        #            print(best_track)
                    track_new = track + best_track
                    return track_new
        #            return []
            else:
                # terminate tracking
                track_new = list(track)
                track_new.append([]) # termination 
                return track_new
            #     else:
            
            
def tracks2array(tracks, n_frames, cell_segs):
    
    YY, XX = np.indices(cell_segs.shape[1:])
    
#    times = np.hstack([int(t.split('_')[1]) for t in tracks.columns[1:]]) - 1
    times = np.arange(n_frames)
    all_tra_ids = range(len(tracks))
    
    tra_array = np.zeros((len(all_tra_ids), len(times), 2))
    tra_array[:] = np.nan
    
    for ii in all_tra_ids:
        
        tra = tracks[ii]
        
        for tt in tra:
            if len(tt) > 0: 
                frame = tt[0]
                spixel_region = tt[1][0]
                tra_y = np.mean(YY[cell_segs[frame]==spixel_region])
                tra_x = np.mean(XX[cell_segs[frame]==spixel_region])
                
                tra_array[ii, frame, 0] = tra_x
                tra_array[ii, frame, 1] = tra_y
                
    return tra_array


def tracks2array_multi(tracks, n_frames, cell_segs):
    
    YY, XX = np.indices(cell_segs.shape[1:])
    
#    times = np.hstack([int(t.split('_')[1]) for t in tracks.columns[1:]]) - 1
    times = np.arange(n_frames)
    all_tra_ids = range(len(tracks))
    
    tra_array = np.zeros((len(all_tra_ids), len(times), 2))
    tra_array[:] = np.nan
    
    for ii in all_tra_ids:
        
        tra = tracks[ii]
        
        for tt in tra:
            if len(tt) > 0: 
                frame = tt[0]
                spixel_region = tt[1]
                
                spixel_region_mask = np.zeros(cell_segs[0].shape, dtype=np.bool)
                
                for ss in spixel_region:
                    spixel_region_mask[cell_segs[frame]==ss] = 1
                
                tra_y = np.mean(YY[spixel_region_mask])
                tra_x = np.mean(XX[spixel_region_mask])
                
                tra_array[ii, frame, 0] = tra_x
                tra_array[ii, frame, 1] = tra_y
                
    return tra_array

def track_cell_ids_2_array(tracks, n_frames):
    
    times = np.arange(n_frames)
    all_tra_ids = range(len(tracks))
          
    tra_array = np.zeros((len(all_tra_ids), len(times))).astype(np.int) # to save space. 0 = background id anyway. 
#    tra_array[:] = np.nan
    
    for ii in all_tra_ids:
        
        tra = tracks[ii]
        
        for tt in tra:
            if len(tt) > 0: 
                frame = tt[0]
                spixel_region = int(tt[1][0])
                
                tra_array[ii, frame] = spixel_region
                
    return tra_array
    
"""
to do: a function to look up the 3D centroid distances. 
"""  
def lookup_3D_region_centroids(labelled, unwrap_params):

    uniq_regions = np.setdiff1d(np.unique(labelled), 0 )

    centroids = []
    
    for re in uniq_regions:
        
        binary = labelled == re
        binary_3D = unwrap_params[binary,:].copy()
        
        centroids.append(np.mean(binary_3D, axis=0))
        
    centroids = np.vstack(centroids)
    
    return centroids

def filter_min_max_areas(cells, min_thresh=0, max_thresh=np.inf): 
    
    cells_ = cells.copy()
    mask_centroids, mask_areas = get_centroids_and_area(cells)
    
    min_cell_area_filter = mask_areas < min_thresh
    max_cell_area_filter = mask_areas> max_thresh
    
    if min_cell_area_filter.sum() > 0:
        min_cell_area_filter_ids = np.setdiff1d(np.unique(cells), 0)[min_cell_area_filter]
        
        for iid in min_cell_area_filter_ids:
            cells_[cells==iid] = 0
    
    if max_cell_area_filter.sum() > 0:
        max_cell_area_filter_ids = np.setdiff1d(np.unique(cells), 0)[max_cell_area_filter]
        
        for iid in max_cell_area_filter_ids:
            cells_[cells==iid] = 0
    
    return cells_

def get_centroids_and_area(labelled):
    
    from skimage.measure import regionprops
    
    regions = regionprops(labelled)
    
    region_areas = np.hstack([re.area for re in regions])
    region_centroids = np.vstack([re.centroid for re in regions])
    
    return region_centroids, region_areas


def mkdir(folder):

    import os 
    if not os.path.exists(folder):
        print('creating')
        os.makedirs(folder)
    
    
if __name__=="__main__":
    
    import numpy as np
    import pylab as plt 
    import skimage.io as skio 
    import skimage.morphology as skmorph
    from tqdm import tqdm 
    import scipy.io as spio 
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.metrics.pairwise import pairwise_distances
    from scipy.optimize import linear_sum_assignment
    import os 
    
    
    """
    input requires the binary + the unwrap_params ( the 3D lookup )
    """
    
    """
    set a few parameters up for deriving centroids and tracking. 
    """
    debug_viz = True
    
# #    binaryfile = 'L491_working_version_tp1-50.tif'
# #    unwrapfile = 'C:\\Users\\felix\\Documents\\PostDoc\\Shankar\\LightSheet_Unwrap-params_Analysis\\L491_ve_000_unwrap_params_unwrap_params-geodesic.mat'
#     unwrapfile = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L505\\L505_ve_120_unwrap_params_unwrap_params-geodesic.mat' # get this file. 

# #    binaryfile = 'L455_E3 EDITED BINARY -20201005T094221Z-001\\L455_E3 EDITED BINARY\\L455_E3_Composite_Fusion_Seam_Edited_MS_ver2-1x.tif'
# #    binaryfile = 'L455_E3 EDITED BINARY -20201005T094221Z-001\\L455_E3 EDITED BINARY\\L455_E3_Composite_Fusion_Seam_Edited_MS_ver3-1x.tif' # updated binary 
#     # binaryfile = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L505\\L505_ve_120_unwrap_ve_geodesic_rect_binary_polarremap_final_v1_1x_paper.tif'
#     binaryfile = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L505\\L505_ve_120_unwrap_ve_geodesic_rect_binary_polarremap_final_v1_1x_paper_MS_edit.tif'

# # =============================================================================
# #     L505
# # =============================================================================
    
#     unwrapfile = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L505\\L505_ve_120_unwrap_params_unwrap_params-geodesic.mat' # get this file. 
#     # binaryfile = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L505\\ver2020-12-30\\L505_ve_120_MS_edit_ver2_30_12_2020.tif'
#     # binaryfile = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L505\\ver2020-12-30_2\\L505_ve_120_MS_edit_ver3_30_12_2020.tif'
#     # binaryfile = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L505\\ver2020-12-31\\L505_ve_120_MS_edit_ver4_30_12_2020.tif'
#     binaryfile = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L505\\ver2020-12-31_2\\L505_ve_120_MS_edit_ver5_30_12_2020.tif'

# # =============================================================================
# #     L864
# # =============================================================================
#     unwrapfile = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L864\\L864_ve_180_unwrap_params_unwrap_params-geodesic.mat' # get this file. 
# #     binaryfile = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L864\\Final_Binary_2020-12-29\\L864_final_edited_binary_RGB.tif'
#     binaryfile = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L864\\2020-12-30\\L864_composite_matt_binary_edit_ver1.tif'
    

# =============================================================================
#   L455_Emb2
# =============================================================================
    # unwrapfile = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L455_Emb2\\L455_Emb2_ve_225_unwrap_params_unwrap_params-geodesic.mat' # get this file. 
    # binaryfile = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L455_Emb2\\annot_2021-02-21\\L455_E2_Mared_Composite_Distal_Edited-1-77.tif'
    
    unwrapfile = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L455_Emb2\\L455_Emb2_ve_225_unwrap_params_unwrap_params-geodesic.mat' # get this file. 
    # binaryfile = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L455_Emb2\\2021-03-22\\22Mar_up2_binary-2x-direct_MS_edit1.tif'
    # binaryfile = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L455_Emb2\\2021-03-25\\25Mar_up2_binary-2x-direct_MS_edit2.tif'
    # binaryfile = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L455_Emb2\\2021-03-25_v2\\25Mar_up2_binary-2x-direct_MS_edit3.tif'
    binaryfile = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Shankar-AVE_Migration\\Annotation\\Curations\\L455_Emb2\\2021-03-26\\26Mar_up2_binary-2x-direct_MS_edit3.tif'

    savefolder = os.path.join(os.path.split(binaryfile)[0],
                              'refine'); mkdir(savefolder)
    base_fname = os.path.split(binaryfile)[-1].split('.tif')[0] # use this to replace relevant filenames
    
    print(savefolder)
    print(base_fname)
    
    # for this video i put the gray first? 
#    gray_vid = skio.imread(binaryfile)[:,0]
#    binary_vid = skio.imread(binaryfile)[:,1]
    
# # normal order 
#     gray_vid = skio.imread(binaryfile)[:,0]
#     binary_vid = skio.imread(binaryfile)[:,1]
    
    gray_vid = skio.imread(binaryfile)[:,1]
    binary_vid = skio.imread(binaryfile)[:,0]
    
    plt.figure()
    plt.subplot(121)
    plt.imshow(gray_vid[0])
    plt.subplot(122)
    plt.imshow(binary_vid[0])
    plt.show()
    
    
    unwrap_params_obj = spio.loadmat(unwrapfile)
    unwrap_params = unwrap_params_obj['ref_map_polar_xyz']

    """
    we should double check if improved if smoothed. 
    """


    
    polar_valid_mask_ii, polar_valid_mask_jj = np.indices(gray_vid.shape[1:])
    polar_center = np.hstack(gray_vid.shape[1:]) / 2. 
    
    polar_center_dist = np.sqrt((polar_valid_mask_ii-polar_center[0])**2  + (polar_valid_mask_jj-polar_center[1])**2)
    polar_valid_mask = polar_center_dist <= gray_vid.shape[1]/2./1.1
    
    # initialise placeholders to store everything
    labelled_vid = []
    centroids_frame = []
    centroids_frame_3D = []
    
    
    
# =============================================================================
#     Step 1: extract all the centroids ready for tracking. 
# =============================================================================
    for tp in np.arange(len(binary_vid))[:]:
#        tp = 0
        
        binary_tp = binary_vid[tp] > 5 # make binary
#        binary_tp = skmorph.binary_dilation(binary_tp, skmorph.square(1))
        
        # may not necessarily need to dilate? -> since this is straightup editted in polar. !
        binary_tp = np.logical_not(binary_tp)
        binary_tp = np.logical_and(binary_tp, polar_valid_mask) # o i see ... 
        
        """
        impose a validity mask.
        """
        regions_tp, centroids_tp = extract_centroids(binary_tp,
                                                      min_area=3, 
                                                      max_area=15000) # check this!. 
        
        labelled_vid.append(regions_tp)
        centroids_frame.append(centroids_tp)
        
        """
        get the equivalent 3D centroid coordinates for non-zero ids. 
        """
        centroids_tp_3D = lookup_3D_region_centroids(regions_tp, unwrap_params)
        centroids_frame_3D.append(centroids_tp_3D)
        
        if debug_viz:
            fig, ax = plt.subplots(figsize=(8,8))
            plt.title(str(tp+1))
            ax.imshow(regions_tp, cmap='coolwarm')
            ax.plot(centroids_tp[:,1], 
                    centroids_tp[:,0], 'wo')
            plt.show()
            
    cells = np.array(labelled_vid)
    centroids_frame = np.array(centroids_frame)
    centroids_frame_3D = np.array(centroids_frame_3D)

    """
    check plot in 3D?
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(centroids_frame_3D[0][...,0], 
                centroids_frame_3D[0][...,2], 
                centroids_frame_3D[0][...,1], 
                marker='o')
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()
    
    
    """
    """
    print('cells', cells.shape)
    
    cellsavefile = 'cells_' + binaryfile
    cellsavefile = os.path.join(savefolder, 'cells_' + base_fname+'.tif')
    skio.imsave(cellsavefile, np.uint16(cells)) # save out this for later usage. 
    

# =============================================================================
#     Step 2: build the associations, based on iou coverage of regions. 0 = bg.
# =============================================================================
    # these are the standard settings we use. 
    dist_check_3D = 30
    dist_check_2D = 30 # should not exceed this much in 2D !. -> counteract problems with the unwrapping. 
    iou_check = True
    good_iou_thresh = 0.5
    min_area_coverage = 0.1
# =============================================================================
#     First build up pairwise associations telling us how cells_T match to cells_T+1 -> does not check for unique_matches
#     These are frame by frame associations. -> therefore what we need are:
#           1. spixel to spixels map in next frame (good iou match!)
#           2. 
# =============================================================================
    cell_link_lists = [] # what does this cell link to in the next frame with iou>=0.5
    cell_link_lists_in = [[]] # what links into this cell from the previous frame with iou >= 0.5
    cell_link_lists_coverage = [] # what links out of this cell by area coverage (10%) -> change to IoU. 
    cell_link_lists_coverage_in = [[]] # what links into this cell by area coverage (10%) -> change to IoU. 
    cell_link_lists_coverage_iou = [] # what is the corresponding iou of links? 
    
    
    """
    build the link using a cost matrix on distance in 2D vs 3D? 
    """
    # build up cell linkage lists. aka. the bipartite connections. 
    for frame_iter in tqdm(range(len(cells)-1)[:]):
        
        # this grabs the actual area. 
        cells_0 = np.setdiff1d(np.unique(cells[frame_iter]), 0)
        cells_1 = np.setdiff1d(np.unique(cells[frame_iter+1]), 0)
        
        cell_tras = {c:[] for c in cells_0 } # this is the id that it links to . 
        cell_tras_in = {c:[] for c in cells_1 } # these keep a record of what comes in. 
        cell_tras_coverage = {c:[] for c in cells_0 }
        cell_tras_coverage_in = {c:[] for c in cells_1 }
        cell_tras_coverage_iou = {c:[] for c in cells_0 }
        
#        cell_masks_frame_iter_next = warp_segmentation(cells[frame_iter], flow[frame_iter]) # warp the once. 
        """
        compare the distances direct in 3D to avoid artefact 
        """
        pos_0_2D = centroids_frame[frame_iter]
        pos_1_2D = centroids_frame[frame_iter+1]
        
        pos_0 = centroids_frame_3D[frame_iter]
        pos_1 = centroids_frame_3D[frame_iter+1]
        
        # assign unique nearest neighbors
        dist_matrix = pairwise_distances(pos_0, pos_1)
        dist_matrix_2D = pairwise_distances(pos_0_2D, pos_1_2D)
        ind_i, ind_j = linear_sum_assignment(dist_matrix)
        
        ind_dists = np.hstack([dist_matrix[ind_i[ii], ind_j[ii]] for ii in range(len(ind_i))])
        ind_dists_2D = np.hstack([dist_matrix_2D[ind_i[ii], ind_j[ii]] for ii in range(len(ind_i))])

        val_dist = ind_dists <= dist_check_3D 
        val_dist_2D = ind_dists_2D <= dist_check_2D 
        
        val_dist = np.logical_and(val_dist, val_dist_2D)
        
        # get the valid indices. 
        ind_i = ind_i[val_dist]
        ind_j = ind_j[val_dist]
        
        """
        link the valid indices into the cell ids. 
        """
        cells_0_i = cells_0[ind_i]
        cells_0_j = cells_1[ind_j]
        
        for ii in np.arange(len(ind_i)):
            cell_tras[cells_0_i[ii]] = [cells_0_j[ii]]
        
        # do a distance check in 3D. and draw a link to check.
        plt.figure(figsize=(5,5))
        plt.imshow(gray_vid[frame_iter], cmap='gray')
        plt.plot(pos_0_2D[ind_i,1],
                  pos_0_2D[ind_i,0], 'g.')
        plt.plot(pos_1_2D[ind_j,1],
                  pos_1_2D[ind_j,0], 'r.')
        
        for ii in range(len(ind_i)):
            plt.plot([pos_0_2D[ind_i[ii],1], pos_1_2D[ind_j[ii],1]], 
                    [pos_0_2D[ind_i[ii],0], pos_1_2D[ind_j[ii],0]], 'w-')
        
        plt.show()
        
        
        
        
        
        
#        for ind_c, c in enumerate(cell_tras.keys()):
#            
#            # find the next candidates and append to the tree roots. 
#            mask_1 = cells[frame_iter] == c
##            mask_1 = cell_masks_frame_iter_next == c
##            mask_1 = warp_segmentation(mask_1, flow[frame_iter])>0 # predicting from optflow. 
#            
#            # use the predicted mask
#            area_mask_1 = np.sum(mask_1)
#            
#            next_id_mask = cells[frame_iter+1][mask_1]
#            next_id_mask_cell_ids = np.unique(next_id_mask)
#            
#            # think this is restrictive? 
#            next_areas = np.bincount(next_id_mask)[next_id_mask_cell_ids] / float(area_mask_1)
#            cand_ids = next_id_mask_cell_ids[next_areas>=min_area_coverage]
#            cand_ids = np.setdiff1d(cand_ids, 0)
            
#            # check iou of current with the next one. 
#            if iou_check and len(cand_ids) > 0 : 
#                iou_cands = []
#                
#                for cand_id in cand_ids:
#                    mask_2 = cells[frame_iter+1] == cand_id
#                    iou_12 = iou_mask(mask_1, mask_2)
#                    iou_cands.append(iou_12)
#                    
#                    cell_tras_coverage_in[cand_id].append(c)
#                    
#                iou_cands = np.hstack(iou_cands)
#                iou_cands_good = iou_cands>=good_iou_thresh
#                
#                if np.sum(iou_cands_good) > 0:
#                    cell_tras[c] = [cand_ids[np.argmax(iou_cands)]] # ensure only one IoU link.
#
#                for ccc in cell_tras[c]:
#                    cell_tras_in[ccc].append(c)
#                
#                cell_tras_coverage[c] = cand_ids
#                cell_tras_coverage_iou[c] = iou_cands 
#            else:
#                cell_tras[c] = cand_ids
#                cell_tras_coverage[c] = cand_ids
#        
#        # these give the out nodes to each edge. 
        cell_link_lists.append(cell_tras)
#        cell_link_lists_coverage.append(cell_tras_coverage)
#        
#        # these give the in edges to the nodes.
#        cell_link_lists_in.append(cell_tras_in)
#        cell_link_lists_coverage_in.append(cell_tras_coverage_in)
#        
#        cell_link_lists_coverage_iou.append(cell_tras_coverage_iou)
    
    """
    Linking the associations to create tracks for checking. (try with no MTH.)
    """
    lookaheadtime = 5

    # initialise list storing which spixel to which track (to perform fast lookup.)
    spixels_to_track_ids = [{cc:[] for cc in list(c.keys())} for c in cell_link_lists]
    avail_ids = [np.hstack(list(c.keys())) for c in cell_link_lists] 
    # create a list of all the nodes that have not yet been linked and is therefore available to start new tracks. 
#    spixel_test_region = 310
#    spixel_test_region = 309

    cell_tracks_all = []
    cell_track_id_counter = 0
    
    # for debugging only go with the first frame availabilities. 
    for frame_master in range(len(cell_link_lists))[:]:
    
        avail_ids_frame = avail_ids[frame_master]
        
        if len(avail_ids_frame) > 0:
            cell_tracks = [[[frame_master, [c]]] for c in avail_ids_frame] 
            
            # up to date with the tracks. 
            # potential new track id. 
            cell_track_ids = cell_track_id_counter + np.arange(len(cell_tracks)) # we will be adding at least this number of tracks. 
            cell_track_id_counter += len(cell_tracks)
    
            # iterate over all the starting cells to build up trajectories. 
            for ind_c in tqdm(range(len(cell_tracks))[:]): 
                
                c = cell_tracks[ind_c] # specify the root of the track and start building from here. 
                    
                ############# multi-hypothesis tracking to track through occlusion ##############
                stable_track = list(cell_tracks[ind_c]) # create a copy.... for now. 
#                create_new_track = 1 

                for frame in range(frame_master, len(cell_link_lists))[:]: # as a guide only. 
#                        print(frame)
                    # try to get the next match in time
                    last_entry = stable_track[-1]
                    
                    if len(last_entry) == 0:
                        # termination of track. don't build anymore. 
                        break
                    else:
                        last_c = stable_track[-1][1] # last spixels. 
                        last_c_time = stable_track[-1][0] # time. 
                        
                        
                        if frame == last_c_time:
                            
                            # =============================================================================
                            #    Check whether the current track should be continued to be tracked or early termination                           
                            # =============================================================================
                            tra_id_clashes = []
                            
                            for cc in last_c:
                                cc_tra_id = spixels_to_track_ids[frame][cc]
                                tra_id_clashes+=cc_tra_id
                            
                            if len(tra_id_clashes) > 0:
                                # terminate track. (significantly speeds up everything. )
                                stable_track.append([])
                                # stop 
                                break
                            else:
                                next_c = []
        #                                    print(last_c)
                                if len(last_c) == 1:
                                    # unambiguous. 
                                    cc = last_c[0]
        #                            spixels_to_track_ids[frame][cc].append(cell_track_ids[ind_c])
        #                            # make sure to delete from avail as we have extended this tracklet 
        #                            avail_ids[frame] = np.delete(avail_ids[frame], np.where(avail_ids[frame]==cc)) 
                                    next_cc = cell_link_lists[frame][cc] # try to get a sure match....... 
                                    if len(next_cc) == 0:
#                                        # check on alternative possibilities based on coverage. 
#                                        potential_next = cell_link_lists_coverage[frame][cc]
#                                        potential_next_iou = cell_link_lists_coverage_iou[frame][cc]
#                                        
#                                        if len(potential_next) > 0:
#                                            potential_next = potential_next[potential_next_iou>=0.15] # still needs at least iou of 0.15. (prevent spurious)
#                                            if len(potential_next) > 0:
#                                                # do MHT. 
#    #                                                print('multi-hypothesis-track')
#    #                                            stable_track = multi_hypothesis_tracking(stable_track, potential_next)
##                                                stable_track = multi_hypothesis_tracking(stable_track, potential_next, cell_link_lists, cell_link_lists_coverage, cell_link_lists_coverage_iou, cells, flow, vid, min_iou=0.5, lookuptime=lookaheadtime, accept_score=0.5, lam=1.)
##                                            else:
                                        stable_track.append([])
                                        break # no need to extend. 
                                    elif len(next_cc) == 1:
                                        # extend tracks. normally
                                        stable_track.append([frame+1, next_cc])
                                    
                                    else:
                                        # if otherwise i.e. greater than 1 we can just use multi hypothesis tracking?
#                                        stable_track = multi_hypothesis_tracking(stable_track, next_cc, cell_link_lists, cell_link_lists_coverage, cell_link_lists_coverage_iou, cells, flow, vid, min_iou=0.5, lookuptime=lookaheadtime, accept_score=0.5, lam=1.)
                                        stable_track.append([])
                                        break
                                elif len(last_c) == 0:
                                    # seems redundant 
                                    break
                                else:
    #                                    print('multi-hypothesis-track')
#                                    stable_track = multi_hypothesis_tracking(stable_track, next_cc, cell_link_lists, cell_link_lists_coverage, cell_link_lists_coverage_iou, cells, flow, vid, min_iou=0.5, lookuptime=lookaheadtime, accept_score=0.5, lam=1.)
                                    stable_track.append([])
                            
#                break
            ############# Update the used set of pixels ##############
#                if create_new_track == 1:
                for tra in stable_track:
                    if len(tra) > 0:
                        frame = tra[0]
                        frame_spixels = tra[1]
                        
                        # condition required as cell_link_lists is 1 less than the vid length!. 
                        if frame < len(cells)-1:
                            for cc in frame_spixels:
                                spixels_to_track_ids[frame][cc].append(cell_track_ids[ind_c])
                                # make sure to delete from avail as we have extended this tracklet 
                                avail_ids[frame] = np.delete(avail_ids[frame], np.where(avail_ids[frame]==cc))
                     
                cell_tracks[ind_c] = list(stable_track) # save back the track..... 
              
            # at the end append all tracks into the register. 
            cell_tracks_all += cell_tracks
##                 update the master track file 
#            cell_tracks_all += cell_tracks[ind_c]
#                cell_track_id_counter += 1
                
    print(len(cell_tracks_all))

    track_lens = [len(ttt) for ttt in cell_tracks_all]

    plt.figure()
    plt.hist(track_lens)
#    plt.ylim([0,170])
    plt.show()

    
# =============================================================================
#     For convenience build the cell tracks all as n_tracks x n_time of cell ids. 
# =============================================================================
    cell_tracks_all_time = track_cell_ids_2_array(cell_tracks_all, len(cells)) # coerces. 
    
# =============================================================================
#    Build the tracks and plot  
# =============================================================================
#    temporal_merged_tras = temporal_merged_tras + [merged_cell_tras[xxx] for xxx in non_merged_temporal_tracks]
    tracks_merged = tracks2array_multi(cell_tracks_all, len(cells), cells)
        

    fig, ax = plt.subplots(figsize=(20,20))
    ax.imshow(gray_vid[0], cmap='gray', alpha=1.)
    ax.plot(tracks_merged[:,0,0], 
            tracks_merged[:,0,1], 'wo')
    
    for ii in range(len(tracks_merged)):
        ax.plot(tracks_merged[ii, :, 0],
                tracks_merged[ii, :, 1], lw=3)
    plt.show()
    
    
    
    ### check the track lengths by plotting it. 
    fig, ax = plt.subplots(figsize=(20,20))
    ax.imshow(gray_vid[0], alpha=0.5)
    
    plot_pts = []
    plot_vals = []

    for ii in range(len(tracks_merged)):
        tra = tracks_merged[ii]
        start = np.arange(len(tra))[~np.isnan(tra[:,0])][0]
        plot_pts.append(tra[start])
        plot_vals.append(np.sum(~np.isnan(tra[:,0])))
        
    plot_pts = np.vstack(plot_pts)
    plot_vals = np.hstack(plot_vals)
        
    ax.scatter(plot_pts[:, 0],
                plot_pts[:, 1], c=plot_vals, cmap='coolwarm')
    plt.show()
    
    
    """
    save tracks out in the same format. 
    """
    
    import pandas as pd 
    
    tracktable_csv = []
    
    for ii in np.arange(len(tracks_merged)):
        
        tra = tracks_merged[ii]
        tra[np.isnan(tra)] = 0 
        
        tracktable_csv.append(np.hstack([ii, 1, tra[:,0]]))
        tracktable_csv.append(np.hstack([ii, 1, tra[:,1]]))
        
    tracktable_csv = np.vstack(tracktable_csv)
    
    
    tracktable = pd.DataFrame(tracktable_csv, index=None,
                              columns = np.hstack(['track_id', 'checked', np.hstack(['Frame_%d' %(jj) for jj in np.arange(tracks_merged.shape[1])])]))
    tracktable['track_id'] = tracktable['track_id'].astype(np.int)
    tracktable['checked'] = tracktable['checked'].astype(np.int)
    
#    tracktable.to_csv('Holly_polar_distal_corrected_nn_tracking-full-dist30.csv', index=None)
    tracktable.to_csv(os.path.join(savefolder, 
                                    '%s-manual_polar_distal_corrected_nn_tracking-full-dist30.csv' %(base_fname)), index=None)
    
    # save not only the tracking but also all associated. intermediate information including e.g. centroids and parsed cell regions. 
    
                              
    
    
    """
    so cell tracks all contains the actual links of all the tracks?
    
    # save this with associated cell segmentation image
    """
#    spio.savemat('cell_tracks_w_ids_' + 'Holly_polar_distal_corrected_nn_tracking-full-dist30.mat', 
#                 {'cell_tracks_xy': tracks_merged, 
#                  'cell_tracks_ids': cell_tracks_all_time})
    
    spio.savemat(os.path.join(savefolder, 
                              'cell_tracks_w_ids_' + '%s-manual_polar_distal_corrected_nn_tracking-full-dist30.mat' %(base_fname)), 
                  {'cell_tracks_xy': tracks_merged, 
                  'cell_tracks_ids': cell_tracks_all_time})
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    