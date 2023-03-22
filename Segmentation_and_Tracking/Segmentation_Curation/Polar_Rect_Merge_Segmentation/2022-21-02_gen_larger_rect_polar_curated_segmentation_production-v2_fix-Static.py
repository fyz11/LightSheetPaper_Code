# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 17:25:13 2020

@author: felix
"""
import numpy as np 

def extract_centroids(binary):
    
    from skimage.measure import label, regionprops
    
    labelled = label(binary)
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

def parse2array(tracks, start_col_id=1):
    
#    times = np.hstack([int(t.split('_')[1]) for t in tracks.columns[1:]]) - 1
    times = np.arange(len(tracks.columns[start_col_id:]))
#    track_ids = np.unique(tracks.iloc[:,0].values)
    print(times)
    all_tra_ids = tracks.iloc[:,0].values
    uniq_tra_ids = np.unique(all_tra_ids)
    
    tra_array = np.zeros((len(uniq_tra_ids), len(times), 2))
    print(tra_array.shape)
    
    for ii, uniq_tra in enumerate(uniq_tra_ids):
        
        select = all_tra_ids == uniq_tra
        data = tracks.loc[select].values
#        print(uniq_tra, data.shape)
        data = data[:,start_col_id:].astype(np.float)
        data[data==0] = np.nan
        
#        print(data.shape, uniq_tra)
        tra_array[ii] = data[:2].T
        
    return uniq_tra_ids, tra_array
    

def compute_start_end_times(tracks):
    
    start_end_times = []
    times = np.arange(tracks.shape[1])
    
    for tra in tracks:
        t_tra = times[~np.isnan(tra[...,0])]
        if len(t_tra) > 0:
            start_end_times.append([t_tra[0],
                                    t_tra[-1]])
        else:
            start_end_times.append([0,
                                    0])
    
    start_end_times = np.vstack(start_end_times)
    
    return start_end_times


def parse_cell_tracks_only( all_track_ids, all_track_info, len_cutoff=1):
    
    start_end_times = compute_start_end_times(all_track_info)
    tracklentimes = start_end_times[:,1] + 1 - start_end_times[:,0]
    
    select = tracklentimes > len_cutoff
    
    return start_end_times[select], tracklentimes[select], all_track_ids[select], all_track_info[select]


"""
rotation pts script. 
"""
def rotate_pts(pts, angle=0, center=[0,0]):
    
    angle_rads = angle / 180. * np.pi
    rot_matrix = np.zeros((2,2))
    rot_matrix[0,0] = np.cos(angle_rads)
    rot_matrix[0,1] = -np.sin(angle_rads)
    rot_matrix[1,0] = np.sin(angle_rads)
    rot_matrix[1,1] = np.cos(angle_rads)
    
#    print(rot_matrix)
    center_ = np.hstack(center)[None,:]
    pts_ = rot_matrix.dot((pts-center_).T).T + center_
    
    return pts_


def rotate_tracks_xy_order(tracks, angle=0, center=[0,0]):

    tracks_rotate = []
    
    for tra in tracks:
#        print(tra.shape)
        tra_rot = rotate_pts(tra, angle=angle, center=center)
        tracks_rotate.append(tra_rot)
        
    return np.array(tracks_rotate)


def track_rmse_2_tracks(tra_1, tra_2):
    
    # using masked arrays 
    tra_1_ = np.ma.masked_array(tra_1, mask=np.isnan(tra_1))
    tra_2_ = np.ma.masked_array(tra_2, mask=np.isnan(tra_2))
    
    tra_12_diff = np.sqrt(((tra_1_-tra_2_)**2).mean()) # to check the validity of this. 
    
    return tra_12_diff


def track_rmse_2_tracks_nonma(tra_1, tra_2):
    
    # using masked arrays 
#    tra_1_ = np.ma.masked_array(tra_1, mask=np.isnan(tra_1))
#    tra_2_ = np.ma.masked_array(tra_2, mask=np.isnan(tra_2))
    
    tra_12_diff = np.sqrt((np.nanmean(tra_1-tra_2)**2)) # to check the validity of this. 
    
    return tra_12_diff
    

def track_corr_2_tracks(tra_1, tra_2):
    
    # using masked arrays 
#    tra_1_ = np.ma.masked_array(tra_1, mask=np.isnan(tra_1))
#    tra_2_ = np.ma.masked_array(tra_2, mask=np.isnan(tra_2))
    
    tra_1_ = tra_1[1:] - tra_1[:-1]
    tra_2_ = tra_2[1:] - tra_2[:-1]
    
    tra_12_corr = ((tra_1_*tra_2_).sum(axis=-1)) / (np.linalg.norm(tra_1_, axis=-1) * np.linalg.norm(tra_2_, axis=-1) )
    tra_12_corr_N = np.sum(~np.isnan(tra_12_corr))
    
    tra_12_corr = np.nansum(tra_12_corr)/ float(tra_12_corr_N + 1e-8)
    
    return tra_12_corr
    

def get_centroids_and_area(labelled):
    
    from skimage.measure import regionprops
    
    regions = regionprops(labelled)
    
    region_areas = np.hstack([re.area for re in regions])
    region_centroids = np.vstack([re.centroid for re in regions])
    
    return region_centroids, region_areas

# this version fills in holes in the area. 
def create_contour_mask_reg(reg_mask, ksize=2, tol=None):
    
    from skimage.measure import find_contours, approximate_polygon
    from skimage.draw import polygon 
    
    # pre processing of reg_mask? -> this is absolutely necessary ... otherwise no good. 
    reg_mask = skmorph.binary_closing(reg_mask, skmorph.disk(ksize))
    
    m,n = reg_mask.shape
    cont_main_mask = np.zeros((m,n), dtype=np.bool)
    cont = find_contours(reg_mask>0, 0)
    
    if len(cont) > 0: 
        len_cont = np.hstack([len(cc) for cc in cont])
        cont_main = cont[np.argmax(len_cont)]
        
        # simplify this contour. 
        if tol is not None:
            cont_main = approximate_polygon(cont_main, tolerance=tol)
            
        rr,cc = polygon(cont_main[:,0], 
                        cont_main[:,1], 
                        shape=(m,n))
        
        cont_main_mask[rr,cc] = 1
        
    return cont_main_mask


def create_contour_mask_reg_largest_area(reg_mask, ksize=2, tol=None):
    
    from skimage.measure import find_contours, approximate_polygon, label, regionprops
    from skimage.draw import polygon 
    
    # pre processing of reg_mask? -> this is absolutely necessary ... otherwise no good. 
    reg_mask = skmorph.binary_closing(reg_mask, skmorph.disk(ksize))
    
    if reg_mask.sum() > 0:
        reg_mask_labelled = label(reg_mask>0)
        reg_mask_props = regionprops(reg_mask_labelled)
        
        reg_mask_areas = np.hstack([re.area for re in reg_mask_props])
        reg_mask_ids = np.setdiff1d(np.unique(reg_mask_labelled), 0)
        
        cont_main_mask = reg_mask_labelled == reg_mask_ids[np.argmax(reg_mask_areas)]
        
    else:
        cont_main_mask = reg_mask.copy()
        
    return cont_main_mask


"""
custom cell matching script.
"""
def match_cells(labels1, labels2, com1=None, com2=None, K=10):
    
    """
    brute force or nearest neighbour match.
    """
    from scipy.optimize import linear_sum_assignment
    uniq1 = np.setdiff1d(np.unique(labels1), 0)
    uniq2 = np.setdiff1d(np.unique(labels2), 0)
    
    n1 = len(uniq1)
    n2 = len(uniq2)
    
    sim_matrix = np.zeros((n1,n2))
    dice_matrix = np.zeros((n1,n2))
    
    if com1 is not None:
        com1 = np.array(com1)
        
    if com2 is not None:
        com2 = np.array(com2)
    
    if com1 is not None:
        from sklearn.neighbors import NearestNeighbors
        
        nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(com1)
        _, indices = nbrs.kneighbors(com2)
        
        for j in range(len(com2)):
            cand_i = indices[j]
 
            if len(cand_i) > 0:
                for i in range(len(cand_i)):
                    mask1 = labels1 == uniq1[cand_i[i]]
                    mask2 = labels2 == uniq2[j]
                    intersection = np.sum(np.abs(mask1*mask2))
                    union = np.sum(mask1) + np.sum(mask2) - intersection
                    
                    # jaccard. 
                    overlap = intersection / float(union + 1e-8)
                    
                    # dice?
                    dice = 2*intersection / float(np.sum(mask1) + np.sum(mask2) + 1e-8)
                    
#                    print(overlap)
                    sim_matrix[cand_i[i],j] = np.clip(overlap, 0, 1)
                    dice_matrix[cand_i[i],j] = np.clip(dice, 0, 1)
    else:
        for i in range(n1):
            for j in range(n2):
                mask1 = labels1 == uniq1[i]
                mask2 = labels2 == uniq2[j]
                intersection = np.sum(np.abs(mask1*mask2))
                union = np.sum(mask1) + np.sum(mask2) - intersection
                
                overlap = intersection / float(union + 1e-8)
                dice = 2*intersection / float(np.sum(mask1) + np.sum(mask2) + 1e-8)
                
                sim_matrix[i,j] = np.clip(overlap, 0, 1)
                dice_matrix[cand_i[i],j] = np.clip(dice, 0, 1)
            
    ind_i, ind_j = linear_sum_assignment(1-sim_matrix) # need to reverse this (distance)
    
    # return at the end also the sim matrix. 
    return ind_i, ind_j, sim_matrix[ind_i, ind_j], dice_matrix[ind_i, ind_j], sim_matrix
    
def refine_polar_mapped_mask(binary_label, tol=None):
    
    cell_ids = np.setdiff1d(np.unique(binary_label), 0)
    
    binary_label_refine = np.zeros_like(binary_label)
    
    for cell_id in cell_ids:
        
#        cell_id_mask = create_contour_mask_reg(binary_label==cell_id, tol=tol)
        cell_id_mask = create_contour_mask_reg_largest_area(binary_label==cell_id, ksize=2, tol=None)
        binary_label_refine[cell_id_mask>0] = cell_id
        
    return binary_label_refine
        

def unique_pairs(pairs):
    
    pairs_ = np.sort(pairs, axis=1)

    uniq_pairs = np.vstack({tuple(row) for row in pairs_}) 
    uniq_pairs = np.vstack(sorted(uniq_pairs, key=lambda x: x[0]))
    uniq_pairs_counts = np.zeros(len(uniq_pairs))
    
    for pp in pairs_:
        for ref_pp_ii, ref_pp in enumerate(uniq_pairs):
            if np.sum(pp == ref_pp) == 2:
                uniq_pairs_counts[ref_pp_ii] += 1
                break
    
    return uniq_pairs, uniq_pairs_counts


# =============================================================================
#   utility scripts. 
# =============================================================================
#def parse_cells_from_binary(binary, ksize):
    
def map_rect_to_polar(img, mapping, target_shape):
    
    img_polar = np.zeros(target_shape)
    img_polar[mapping[0], 
              mapping[1]] = img.copy()
    
    return img_polar


def parse_cells_from_polar_binary(binary, ksize=3, valid_factor = 1.1):
    
    from skimage.measure import label 
    # from skimage.morphology import remove_small_objects
    
    # create a valid mask ... to exclude border regions. 
    polar_valid_mask_ii, polar_valid_mask_jj = np.indices(binary.shape)
    polar_center = np.hstack(binary.shape[:]) / 2. 

    polar_center_dist = np.sqrt((polar_valid_mask_ii-polar_center[0])**2  + (polar_valid_mask_jj-polar_center[1])**2)
    polar_valid_mask = polar_center_dist <= binary.shape[0]/2./valid_factor
    
    
    cell_polar_binary_frame = np.logical_not(skmorph.binary_dilation(binary, skmorph.square(ksize))) #
    cell_polar_binary_frame = np.logical_and(polar_valid_mask, cell_polar_binary_frame)
    # cell_polar_binary_frame = remove_small_objects(cell_polar_binary_frame, 10)
    
     # produce the initial labelling. 
    binary_polar_label = label(cell_polar_binary_frame, connectivity=1)
    
    # get the areas to flag up super unreliable? 
    binary_polar_label_centroids, binary_polar_label_areas = get_centroids_and_area(binary_polar_label)
    
    return binary_polar_label, (binary_polar_label_centroids, binary_polar_label_areas)

    
def parse_cells_from_rect_binary(binary, ksize=1):
    
    from skimage.measure import label 
    # from skimage.morphology import remove_small_objects
    
    cell_rect_binary_frame = np.logical_not(skmorph.binary_dilation(binary, skmorph.disk(ksize)))
    # cell_rect_binary_frame = remove_small_objects(cell_rect_binary_frame, 10)
    
    cell_rect_binary_frame_label = label(cell_rect_binary_frame, connectivity=1)
    binary_rect_label_centroids, binary_rect_label_areas = get_centroids_and_area(cell_rect_binary_frame_label)
    
    return cell_rect_binary_frame_label, (binary_rect_label_centroids, binary_rect_label_areas)


def create_distal_radial_mask(shape, dist_thresh):
    
    YY,XX = np.indices(shape) 
    cc = np.hstack(shape)/2.
    dists = np.sqrt( ( YY - cc[0]) ** 2  + ( XX - cc[1]) ** 2)
    
    return dists < dist_thresh


def compare_polar_rect_binaries( polar_cells, rect_cells, ksize_rect=3, min_area=5, exclude_ids=[]):
    
    """
    might need to engineer the masking a bit ? to avoid heavy overlap? 
    """
    # get the cell ids. 
    polar_cell_ids = np.setdiff1d(np.unique(polar_cells), 0) 
#    rect_cell_ids = np.setdiff1d(np.unique(rect_cells), 0)
    
    # set a max polar id to help create new ids without affecting old ones. 
    max_polar_id = polar_cell_ids.max()
    merged_mask = np.zeros_like(polar_cells) # saving the final merged mask. -> starts off initially zero. 
    
    for ii in range(len(polar_cell_ids))[:]:
        
        cell_ii = polar_cell_ids[ii]
        polar_reg_mask = polar_cells == cell_ii
        
        if cell_ii not in exclude_ids:
        
            # go on
            polar_reg_mask_cover_rect_id = np.setdiff1d(np.unique(rect_cells[polar_reg_mask>0]), 0) 
            
            
            # do a check to see how many remain following an area check. 
            if len(polar_reg_mask_cover_rect_id) > 1:
            
                areas_covered = []
                
                for jj in range(len(polar_reg_mask_cover_rect_id)):
                    
                    cell_jj_id = polar_reg_mask_cover_rect_id[jj]
                    rect_reg_mask = rect_cells == cell_jj_id

                    # get the largest area using contour. 
    #                rect_reg_mask = create_contour_mask_reg(rect_reg_mask,ksize=1)
                    rect_reg_mask = create_contour_mask_reg_largest_area(rect_reg_mask, ksize=ksize_rect)
                    areas_covered.append(rect_reg_mask.sum())
                    
                areas_covered = np.hstack(areas_covered)
                
                n_valid = areas_covered > min_area
                
                if n_valid.sum() > 1:
                    # there may be a case for splitting... but we need to be careful and take care of the wrapping around. 
                    polar_reg_mask_cover_rect_id_process = polar_reg_mask_cover_rect_id[n_valid]
                    
                    for jj in range(len(polar_reg_mask_cover_rect_id_process)):
                    
                        cell_jj_id = polar_reg_mask_cover_rect_id_process[jj]
                        rect_reg_mask = rect_cells == cell_jj_id
                        
    #                     get the largest area using contour. 
    #                    rect_reg_mask = create_contour_mask_reg(rect_reg_mask, ksize=1)
                        rect_reg_mask = create_contour_mask_reg_largest_area(rect_reg_mask, ksize=3)
    #                    areas_covered.append(rect_reg_mask.sum())
                        if rect_reg_mask.sum() > 0:
#                            rect_reg_mask = np.logical_and(rect_reg_mask, polar_reg_mask) # how essential is this. 
                            
                            merged_mask[rect_reg_mask>0] = max_polar_id + 1 
                            max_polar_id = max_polar_id + 1
                            
                else:
                    if polar_reg_mask.sum() > min_area:
                        merged_mask[polar_reg_mask>0] = polar_cell_ids[ii]
                    
            else:
                if polar_reg_mask.sum() > min_area:
                    merged_mask[polar_reg_mask>0] = polar_cell_ids[ii]
                        
        else:
            # we are keeping and should be in the final merged mask. 
            if polar_reg_mask.sum() > min_area:
                merged_mask[polar_reg_mask>0] = polar_cell_ids[ii]
            
            
    return merged_mask


def refine_left_seam_xaxis(polar_labels, polar_labels_original, min_occur=2, min_pair_frequency=2, start_x_factor=1.05, refine_strip_size=10, use_polar=False):
    
    """
    try to refine the 180 degree boundary while referencing and matching. 
    """
    
    # create local strips. 
    strip_size = refine_strip_size
    start_x = int(polar_labels.shape[0] - polar_labels.shape[0]/start_x_factor)
    cc = np.hstack(polar_labels.shape) / 2.
    end_x = int(cc[1])
    
    boundary_x_samples = np.linspace( start_x, end_x, end_x+1-start_x).astype(np.int)
    
    # query the labels across the strips. . 
    cell_ids_boundary_line = np.array([polar_labels[int(cc[0])-strip_size:int(cc[0])+strip_size+1, 
                                                    boundary_x_samples[xx]:boundary_x_samples[xx+1]].ravel() for xx in np.arange(len(boundary_x_samples))[:-1]])

    # tabulate the adjacent ids that occur nearest the middle point along the boundary line. 
    cell_ids_boundary = []
    
    for cc in cell_ids_boundary_line: 
        
        # take the top 2 unique..
        uniq_cells = np.setdiff1d(np.unique(cc), 0) 
        
        if len(uniq_cells) > 0: 
            uniq_cells_counts = np.hstack([np.sum(cc==ccc) for ccc in uniq_cells])
            
            if np.sum(uniq_cells_counts > min_occur ) > 1: # check for minimum occurrence.  
                # then we can tabulate the most populous 2
                majority_ids = uniq_cells[np.argsort(uniq_cells_counts)[::-1]]
                cell_ids_boundary.append([majority_ids[:2]])

    cell_ids_boundary = np.vstack(cell_ids_boundary)
    
    # uniquefy the pairs at the cell boundary and tabulate the frequency 
    cell_ids_boundary_uniq, cell_ids_boundary_uniq_counts = unique_pairs(cell_ids_boundary)
    cell_ids_boundary_uniq = cell_ids_boundary_uniq[cell_ids_boundary_uniq_counts>min_pair_frequency] # this is another failsafe.. in terms of frequency of occurrence. 
    
    """
    Apply the filtering to get a candidate list for merging. 
    """
    final_merge_ids = []
    
    for cc in cell_ids_boundary_uniq[:]:
        
        cc_0, cc_1 = cc 
        
        # look up associated region in polar view. 
        # check 1: same majority polar id for both regions. 
        mask_0 = polar_labels == cc_0 
        mask_1 = polar_labels == cc_1 
        
        mask_polar_lookup_0 = polar_labels_original[mask_0]
        mask_polar_lookup_1 = polar_labels_original[mask_1]
        
        uniq_polar_0 = np.setdiff1d(np.unique(mask_polar_lookup_0), 0)
        uniq_polar_1 = np.setdiff1d(np.unique(mask_polar_lookup_1), 0)
        
        """
        resolve majority cases. 
        """
        if len(uniq_polar_0) == 1:
            uniq_polar_0 = uniq_polar_0[0]
        elif len(uniq_polar_0) > 1:
            # pick majority 
            occurrence_0 = np.hstack([np.sum(mask_polar_lookup_0==uu) for uu in uniq_polar_0])
            uniq_polar_0 = uniq_polar_0[np.argmax(occurrence_0)]
        else:
            uniq_polar_0 = -1
            
        if len(uniq_polar_1) == 1:
            uniq_polar_1 = uniq_polar_1[0] 
        elif len(uniq_polar_1) > 1:
            occurrence_1 = np.hstack([np.sum(mask_polar_lookup_1==uu) for uu in uniq_polar_1])
            uniq_polar_1 = uniq_polar_1[np.argmax(occurrence_1)]
        else:
            uniq_polar_1 = -1
            
        # check 2: the iou between this and the polar overlap is similar. ? -> do you need this? 
        """
        if both ids are the same and there is not a nan ...  
        """
        if ~np.isnan(uniq_polar_0) and ~np.isnan(uniq_polar_1):
            
            if uniq_polar_0 == uniq_polar_1:
                # then they should be part of the same id ... 
                final_merge_ids.append(np.hstack([cc, uniq_polar_0]))
                
#                plt.figure()
#                canvas_show = np.zeros_like(polar_labels)
#                canvas_show[polar_labels==cc[0]] = cc[0]
#                canvas_show[polar_labels==cc[1]] = cc[1]
#                
#                canvas_show = label2rgb(canvas_show, bg_label=0)
#                
#                plt.imshow(canvas_show)
#                plt.imshow(refined_polar_mask, alpha=0.25)
#                plt.show()
                  
    polar_labels_ = polar_labels.copy()
                
    if len(final_merge_ids) > 0:
        final_merge_ids = np.vstack(final_merge_ids)
         
        # implement the changes. 
        max_id_cell = np.max(polar_labels)
        
        for ff in final_merge_ids:
            
            cc_0, cc_1, cc_new = ff
            
            polar_labels_[polar_labels==cc_0] = 0
            polar_labels_[polar_labels==cc_1] = 0
            
            if use_polar:
                polar_labels_[polar_labels_original ==cc_new] = max_id_cell + 1
            else:
                comb_mask = np.logical_or(polar_labels==cc_0, polar_labels==cc_1)
                polar_labels_[comb_mask] = max_id_cell + 1
                
            max_id_cell = max_id_cell + 1 
            
    return polar_labels_
        
# =============================================================================
#   filtering operations on regions 
# =============================================================================

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
        
    
def filter_non_solid_areas(cells, thresh=0.7):
    
    from skimage.measure import regionprops
    
    cells_ = cells.copy()
    cells_prop = regionprops(cells)
    
    cells_solidity = np.hstack([re.solidity for re in cells_prop])
    
    cells_solidity_filter = cells_solidity < thresh
    
    if cells_solidity_filter.sum() > 0:
        cells_solid_filter_ids = np.setdiff1d(np.unique(cells), 0)[cells_solidity_filter]
        
        for iid in cells_solid_filter_ids:
            cells_[cells==iid] = 0
    
    return cells_
    
    
def aggregate_binary_masks(binary_list):
    
    from skimage.measure import label
    
    labelled_list = [label(binary) for binary in binary_list]
    consensus_segs = labelled_list[0]

    if len(labelled_list) > 1:
        
        for ll in range(len(labelled_list)-1):
            candidate = labelled_list[ll+1]
            uniq_ids_consensus = np.setdiff1d(np.unique(consensus_segs), 0)
            
            consensus_segs_temp = consensus_segs.copy()
            
            for reg in uniq_ids_consensus:
                reg_mask = consensus_segs == reg
                next_reg_mask = candidate[reg_mask]
                
                next_reg_mask_uniq_ids = np.setdiff1d(np.unique(next_reg_mask), 0)
                
                if len(next_reg_mask_uniq_ids) > 1: 
                    # replace. 
                    consensus_segs_temp[reg_mask] = 0
                    max_cell_id = np.max(consensus_segs_temp)
                    
                    for jj, re in enumerate(next_reg_mask_uniq_ids):
#                        print(re)
#                        print(next_reg_mask)
                        consensus_segs_temp[candidate==re] = max_cell_id + jj + 1 # important to increment, else will have duplicates. 
                        
            consensus_segs = consensus_segs_temp.copy()

    return consensus_segs


# =============================================================================
# a script to allow reallocation of 'black pixels' in polar view. within the boundary of the outer circle only to be reassigned ....
# =============================================================================

def grow_polar_regions_nearest(labels, metric='manhattan', debug=False):
    
    from scipy.ndimage.morphology import binary_fill_holes
    from sklearn.neighbors import NearestNeighbors
    """
    script will grow the cell regions so that there is 0 distance between each cell region. 
    """
    bg = labels>0
    bg = skmorph.binary_closing(bg, skmorph.disk(5))
    valid_mask = binary_fill_holes(bg) # this needs to be a circle... 
    
    uncovered_mask = np.logical_and(valid_mask, labels==0)
    covered_mask = np.logical_and(valid_mask, labels > 0)
    
    if debug: 
        plt.figure()
        plt.title('uncovered')
        plt.imshow(uncovered_mask)
        plt.show()
        
        plt.figure()
        plt.title('covered')
        plt.imshow(covered_mask)
        plt.show()
        
    YY, XX = np.indices(labels.shape)
    
    uncovered_coords = np.vstack([YY[uncovered_mask], 
                                  XX[uncovered_mask]]).T
        
    covered_coords = np.vstack([YY[covered_mask], 
                                XX[covered_mask]]).T
        
#    print(uncovered_coords.shape)
#    print(covered_coords.shape)
    
    """
    build a kNN tree for assigning the remainder. 
    """
    neigh = NearestNeighbors(n_neighbors=1, metric=metric)
    neigh.fit(covered_coords)

    nbr_indices = neigh.kneighbors(uncovered_coords, return_distance=False)
    nbr_indices = nbr_indices.ravel()
    
    """
    look up what labels should be assigned
    """
    labels_new = labels.copy()
    
    nbr_pts = covered_coords[nbr_indices]
    
    labels_new[uncovered_coords[:,0], 
               uncovered_coords[:,1]] = labels[nbr_pts[:,0], nbr_pts[:,1]].copy()
    
    return labels_new


def grow_polar_regions_nearest_3D(labels, 
                                  unwrap_params, 
                                  dilate=5, 
                                  metric='manhattan', debug=False):
    
    from scipy.ndimage.morphology import binary_fill_holes
    from sklearn.neighbors import NearestNeighbors
    
    """
    script will grow the cell regions so that there is 0 distance between each cell region. 
    """
    bg = labels>0
    bg = skmorph.binary_closing(bg, skmorph.disk(dilate))
    valid_mask = binary_fill_holes(bg) # this needs to be a circle... 
    
    uncovered_mask = np.logical_and(valid_mask, labels==0)
    covered_mask = np.logical_and(valid_mask, labels > 0)
    
    if debug_viz: 
        plt.figure()
        plt.title('uncovered')
        plt.imshow(uncovered_mask)
        plt.show()
        
        plt.figure()
        plt.title('covered')
        plt.imshow(covered_mask)
        plt.show()
        
    YY, XX = np.indices(labels.shape)
    
    uncovered_coords = np.vstack([YY[uncovered_mask], 
                                  XX[uncovered_mask]]).T
        
    covered_coords = np.vstack([YY[covered_mask], 
                                XX[covered_mask]]).T
    
    uncovered_coords_3D = unwrap_params[uncovered_coords[:,0], 
                                        uncovered_coords[:,1]]
    covered_coords_3D = unwrap_params[covered_coords[:,0], 
                                      covered_coords[:,1]]
    
    """
    build a kNN tree for assigning the remainder. 
    """
    neigh = NearestNeighbors(n_neighbors=1, metric=metric)
    neigh.fit(covered_coords_3D)

    nbr_indices = neigh.kneighbors(uncovered_coords_3D, return_distance=False)
    nbr_indices = nbr_indices.ravel()
    
    """
    look up what labels should be assigned
    """
    labels_new = labels.copy()
    
    nbr_pts = covered_coords[nbr_indices]
    
    labels_new[uncovered_coords[:,0], 
               uncovered_coords[:,1]] = labels[nbr_pts[:,0], nbr_pts[:,1]].copy()
    
    return labels_new


def create_segmentations_mask(cells, ids):
    
    cells_ = np.zeros_like(cells)
    
    for idd in ids:
        cells_[cells==idd] = idd
        
    return cells_
    
#def refine_polar_mapped_mask(binary_label, tol=None):
#    
#    cell_ids = np.setdiff1d(np.unique(binary_label), 0)
#    
#    binary_label_refine = np.zeros_like(binary_label)
#    
#    for cell_id in cell_ids:
#        
#        cell_id_mask = create_contour_mask_reg(binary_label==cell_id, tol=tol)
#        binary_label_refine[cell_id_mask>0] = cell_id
#        
#    return binary_label_refine
    

def largest_area_id(cells):
    
    from skimage.measure import regionprops
    
    regprops = regionprops(cells)
    areas = np.hstack([re.area for re in regprops])
    reg_ids = np.setdiff1d(np.unique(cells), 0)
    max_reg_id = reg_ids[np.argmax(areas)]
    
    return max_reg_id, cells == max_reg_id


def largest_component(binary):
    
    from skimage.measure import regionprops, label
    
    cells = label(binary, connectivity=1)
    regprops = regionprops(cells)
    areas = np.hstack([re.area for re in regprops])
    reg_ids = np.setdiff1d(np.unique(cells), 0)
    max_reg_id = reg_ids[np.argmax(areas)]
    
    return max_reg_id, cells == max_reg_id

def erode_large_regions( binary, 
                        erode_ksize=5,
                        size_region = 100):

    from skimage.measure import label
    from scipy.ndimage.morphology import binary_fill_holes

    binary_new = np.zeros(binary.shape,dtype=np.bool)
    lab = label(binary)
    uniq_labs = np.setdiff1d(np.unique(lab), 0)
    
    for ll in uniq_labs:
        ll_mask = lab == ll
        # ll_mask = binary_fill_holes(ll_mask)

        if ll_mask.sum() >= size_region:
            ll_mask = skmorph.binary_erosion(ll_mask, skmorph.disk(erode_ksize))
            binary_new[ll_mask] = 1
        else:
            binary_new[ll_mask] = 1

    return binary_new


def find_embryo_folders(infolder, key, exclude=None):
    
    import os 
    
    folders = os.listdir(infolder)
    folders_found = []
    
    for ff in folders:
        if key in ff:
            include = 1
            if exclude is not None:
                for ex in exclude:
                    if ex in ff:
                        # hello
                        include = 0
            if include == 1:
                folders_found.append(os.path.join(infolder,ff))
    
    if len(folders_found)>0:
        folders_found = np.hstack(folders_found)
        
    return folders_found

def match_folder(query, ref_list):
    
    import os 
    import numpy as np 
    
    look_vect = np.hstack([query in os.path.split(rr)[-1] for rr in ref_list])
    
    return look_vect


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


# def get_folder_index(query_list, ref_list):
    
#     import os 
#     import numpy as np 
    
#     index = np.arange(len(ref_list))
#     all_index = []
    
#     for query in query_list: 
#         ind = index[ref_list==query]
#         all_index.append(ind)
#     all_index = np.hstack(all_index)
    
#     return all_index


def get_folder_index(query_list, ref_list):
    
    import os 
    import numpy as np 
    
    index = np.arange(len(ref_list))
    all_index = []
    
    for query in query_list: 
        ind_bool = np.hstack([query in rr for rr in ref_list])
        ind = index[ind_bool]
        all_index.append(ind)
    all_index = np.hstack(all_index)
    
    return all_index


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
    from skimage.measure import regionprops
    import pandas as pd 
    import skimage.transform as sktform
    from tqdm import tqdm 
    
    from skimage.color import label2rgb
    from skimage.segmentation import find_boundaries
    from skimage.filters import gaussian
    import os 
    
    from skimage.measure import label
    import glob
    import re
        
# =============================================================================
#     some parameter tuning to use. 
# =============================================================================
    
    # distal_dist_polar_keep = 100 # designates the distance around the polar distal tip ids that we don't replace. 
    
    """
    This parameter will need some tuning. 
    """
    # distal_dist_polar_keep = [100, 30, 60, 60] # not great  7th one or aka 6th N7 embryo don't do distal correct. 
    
    #### for A. 
    # distal_dist_polar_keep = [30, 30, 30, 40] # not great  7th one or aka 6th N7 embryo don't do distal correct. 
    
    #### for B. Yap.  40 was ok for 2nd
    
    distal_dist_polar_keep = [70, 60, 30, 40, 70, 55] # not great  7th one or aka 6th N7 embryo don't do distal correct. 
    # distal_dist_polar_keep = [60] # for 7
    #### for C. Pre-induction 
    # distal_dist_polar_keep = [45, 40, 60, 40,50]
    file_no = 5
    
    
    max_cell_area = 20000 
    upsample_factor = 2 # how much we need to increase this by ?
    # upsample_factor = 1
    
    polar_erode_ksize = 1 # whats a good setting?
    # polar_erode_ksize = 4 
    rect_erode_ksize = 3 # default 
    min_cell_area = 10 # former 5. 
    debug_viz = True
    
    distal_correct = True
    # distal_correct = False
    
# =============================================================================
#     Find the relevant files to allow automatic processing
# =============================================================================

    # date = '2022-05-03'
    date = '2023-02-21'
    
    # annotfolder = 'F:\\Shankar-2\\Control_Embryos_Lefty\\Lefty_fixed_outlines_edit1'
    # annotfolder = r'F:\Work\Research\Shankar\Additional_Fixed_Datasets\Addtional_Confocal_Data_for_Fixed_Paper\A_PI DAPI and Actin only\Outline_A_PI DAPI  and Actin only'
    annotfolder = r'F:\Work\Research\Shankar\Additional_Fixed_Datasets\Addtional_Confocal_Data_for_Fixed_Paper\B_E5_5 YAP staining\B_5_5_ YAP outlining ver1'
    # annotfolder = r'F:\Work\Research\Shankar\Additional_Fixed_Datasets\Addtional_Confocal_Data_for_Fixed_Paper\C_Pre_Induction YAP Staining\C_Pre_Induction YAP Staining outlining ver1'
    
    # find the various embryo folders and use this to get. 
    # allfolders = find_embryo_folders(annotfolder, key='D4', exclude=['Merged'])
    allfolders = find_embryo_folders(annotfolder, key='Lit', exclude=['Merged'])
    allfolders = np.sort(allfolders)
    allfolders = np.hstack([ff for ff in allfolders if '.zip' not in ff])
    
    """
    Skip the one polar only folder.
    """
    # exclude the only polar annotation only folder. 
    exclude_folder = ['Lit335_Emb3']
    allfolders = np.hstack([ff for ff in allfolders if os.path.split(ff)[-1] not in exclude_folder])    
    
    
    
    # allembryos = np.hstack([os.path.split(ff)[-1].split('_')[0] for ff in allfolders])
    # allembryos = np.hstack([os.path.split(ff)[-1] for ff in allfolders])
    allembryos = np.hstack([os.path.split(ff)[-1].split('_outline')[0] for ff in allfolders])
    
    # all_emb_folder = 'F:\\Shankar-2\\Control_Embryos_Lefty'
    # all_emb_folder = r'F:\Work\Research\Shankar\Additional_Fixed_Datasets\Addtional_Confocal_Data_for_Fixed_Paper\A_PI DAPI and Actin only'
    all_emb_folder = r'F:\Work\Research\Shankar\Additional_Fixed_Datasets\Addtional_Confocal_Data_for_Fixed_Paper\B_E5_5 YAP staining'
    # all_emb_folder = r'F:\Work\Research\Shankar\Additional_Fixed_Datasets\Addtional_Confocal_Data_for_Fixed_Paper\C_Pre_Induction YAP Staining'
    
    all_emb_unwrap_folders = find_embryo_folders(all_emb_folder, key='_reorient_gui')
    all_emb_unwrap_folders = np.sort(all_emb_unwrap_folders)


    # curated_folder = os.path.join(annotfolder, 'A_PI_merge_edit1')
    curated_folder = os.path.join(annotfolder, 'B_YAP_merge_edit1')
    # curated_folder = os.path.join(annotfolder, 'C_PRE_merge_edit1')
    curated_files = glob.glob(os.path.join(curated_folder, '*.tif'))

    # outfolder = os.path.join(annotfolder, 'A_PI_merge_edit1_polar_auto')
    outfolder = os.path.join(annotfolder, 'B_YAP_merge_edit1_polar_auto')
    # outfolder = os.path.join(annotfolder, 'C_PRE_merge_edit1_polar_auto')
    mkdir(outfolder)

# =============================================================================
#     Iterate over the files. 
# =============================================================================

    # check and populate this. 
    # gray_ch = 0 
    # binary_ch = 1
    gray_ch = 1 
    binary_ch = 0


# =============================================================================
#     INPUT 1 : EDITTED BINARY FILE 
# =============================================================================
    from skimage.morphology import skeletonize
    
    for ii in np.arange(len(curated_files))[file_no:file_no+1]: # 1 is the worst... why ?
    
        binaryfile = curated_files[ii]
        cells = skio.imread(binaryfile)
        
        if len(cells.shape) > 2:
            cells = cells[binary_ch].copy()
        cells = skeletonize(cells>0); cells = skmorph.binary_dilation(cells, skmorph.disk(1))
        cells = np.uint8(cells * 255)

        # since only an image. 
        # binary_cells = cells[binary_ch]
        # gray_cells = cells[gray_ch]
        binary_cells = cells.copy() 
        gray_cells = cells.copy()
        
        print(binary_cells.shape)
        
        savefolder, basefname = os.path.split(binaryfile)
        basefname = basefname.split('.tif')[0]
        
        print(savefolder, basefname)
        print('=====')

        
        # # get the embfolder? 
        # emb_folder = allfolders[folder_ii]
        # emb_name = os.path.split(emb_folder)[-1]
        
        """
        use regex to get the query emb name 
        """
        query_lit = re.findall('Lit\d+', basefname)[0]
        # query_emb = re.findall('Emb\d+', basefname)[0]
        try:
            query_emb = re.findall('E\d+', basefname)[0]
        except:
            query_emb = re.findall('Emb\d+', basefname)[0]
        query_name = query_lit + '_' + query_emb
        
        # # special for 7 
        # query_lit = re.findall('D\d+', basefname)[0]
        # query_emb = re.findall('e\d+', basefname)[0]
        # query_name = query_lit #+ '_' + query_emb
        
        """
        for C. 
        """
        # query_name = basefname.split('PRE_')[0]+'PRE_' + query_lit
        
        
        folder_ind = get_folder_index([query_name], allembryos)[0]
        emb_folder = allfolders[folder_ind]
        emb_name = os.path.split(emb_folder)[-1]

        # # get the pre binary version. 
        # polarfile = '../Curations/L455_Emb2/2021-03-17/Matt_Distal_Edit1_Composite_Distal_Edited-1-77.tif'
        # this is a reference polar -> which we can use Matt's old one? 
        # # C:\Users\fyz11\Documents\Work\Projects\Shankar-AVE_Migration\Annotation\Curations\L455_Emb2\2021-03-17
        allannotfiles = os.listdir(emb_folder)
        polarfile = [os.path.join(emb_folder, ff) for ff in allannotfiles if 'polar' in ff][0]
        # C:\Users\fyz11\Documents\Work\Projects\Shankar-AVE_Migration\Annotation\Curations\L455_Emb2\2021-03-17
        polar_cells = skio.imread(polarfile)
        # polar_cells = skio.imread(polarfile)
        polar_cells = polar_cells[binary_ch].copy()
        polar_cells = polar_cells[None,:] # pad to time? 


# =============================================================================
#   INPUT 2 : POLAR -> RECT AND RECT -> POLAR TRANSFORMATION FILE FOR THE EMBRYO.
# =============================================================================
        
        # find the corresponding emb_name in the proper folder. 
        # # unwrap_folder = all_emb_unwrap_folders[match_folder(emb_name+'_', all_emb_unwrap_folders)][0] # to turn into a string. 
        # all_emb_unwrap_folders_suffix = np.hstack([ff.split('-')[0] for ff in all_emb_unwrap_folders])
        
        unwrap_folder = all_emb_unwrap_folders[match_folder(emb_name.split('_outline')[0]+'_', all_emb_unwrap_folders)][0]
        # # for 'C' only 
        # unwrap_folder = all_emb_unwrap_folders[match_folder(emb_name.split('_outline')[0]+'-', all_emb_unwrap_folders)][0]
        
        tformfolder = os.path.join(unwrap_folder, 'unwrapped', 'polar_to_rect_tfs')
        
        unwrap_params_file = glob.glob(os.path.join(unwrap_folder, 'unwrapped', '*.mat'))[0] 
        unwrap_obj = spio.loadmat(unwrap_params_file)
        unwrap_params_polar = unwrap_obj['ref_map_polar_xyz']

        polar_rect_map_file_000_xx = glob.glob(os.path.join(tformfolder, 'polar_rect_mapping_x_*.csv'))[0]
        polar_rect_map_file_000_yy = glob.glob(os.path.join(tformfolder, 'polar_rect_mapping_y*.csv'))[0]
        
        rect_polar_map_file_000_xx = glob.glob(os.path.join(tformfolder, 'rect_polar_mapping_x*.csv'))[0]
        rect_polar_map_file_000_yy = glob.glob(os.path.join(tformfolder, 'rect_polar_mapping_y*.csv'))[0]
        
        # get the various transformations. 
        polar_rect_map_000_polar_rect_ = np.dstack([pd.read_csv(polar_rect_map_file_000_yy), 
                                                    pd.read_csv(polar_rect_map_file_000_xx)]).transpose(2,0,1)
        polar_rect_map_000_rect_polar_ = np.dstack([pd.read_csv(rect_polar_map_file_000_yy), 
                                                    pd.read_csv(rect_polar_map_file_000_xx)]).transpose(2,0,1)
        
    # =============================================================================
    #   ITERATION LOOP over time. 
    # =============================================================================
        
        # artifically pad the static frame. 
        binary_cells = binary_cells[None,:].copy()


        n_frames = 1
        
        cells_refine_all = [] # create an empty array to save the refined. 
        cells_boundary_refine_all = []
        
        polar_frames_all = []
        
        for frame_no in tqdm(np.arange(n_frames)[:]):
            
            polar_rect_map_000_rect_polar = polar_rect_map_000_rect_polar_.copy()
            polar_rect_map_000_polar_rect = polar_rect_map_000_polar_rect_.copy()
            # =============================================================================
            #  a) create a polar version through rect to polar mapping        
            # =============================================================================
            # this is the rect annotation 
            binary_img_frame_rect = binary_cells[frame_no].copy()
            
            if debug_viz == True:
            
                plt.figure(figsize=(10,10))
                plt.imshow(binary_img_frame_rect, cmap='gray')
                plt.show()    
            
            
            # mapping to polar to have a polar version. with upsampling.
            if upsample_factor == 1: 
                binary_img_frame_polar = map_rect_to_polar(binary_img_frame_rect, 
                                                            polar_rect_map_000_rect_polar, 
                                                            target_shape = polar_rect_map_000_polar_rect.shape[1:] )
                
            else:
            
                # applying the resampling. 
                # attempting a larger image. 
    #            target_shape = (np.hstack(polar_rect_map_000_polar_rect.shape[1:])).astype(np.int)
                target_shape = (np.hstack(polar_rect_map_000_polar_rect.shape[1:]) * 2).astype(np.int)
            
                # we need to increase this too. 
                binary_img_frame_rect = sktform.resize(binary_img_frame_rect, ((upsample_factor*binary_img_frame_rect.shape[0], upsample_factor*binary_img_frame_rect.shape[1])), preserve_range=True) > 0
            
                plt.figure()
                plt.title('upsample')
                plt.imshow(binary_img_frame_rect)
                plt.show()
            
                polar_rect_map_000_rect_polar = sktform.resize(polar_rect_map_000_rect_polar, ((polar_rect_map_000_rect_polar.shape[0], upsample_factor*polar_rect_map_000_rect_polar.shape[1], upsample_factor*polar_rect_map_000_rect_polar.shape[2])), preserve_range=True).astype(np.int)
            
                binary_img_frame_polar = map_rect_to_polar(binary_img_frame_rect, 
                                                          polar_rect_map_000_rect_polar * upsample_factor, 
                                                          target_shape = target_shape) 

                # binary_img_frame_polar = skmorph.binary_dilation(binary_img_frame_polar>0, skmorph.disk(1))
                
                plt.figure()
                plt.title('upsample')
                plt.imshow(binary_img_frame_polar)
                plt.show()
    #            binary_img_frame_polar = sktform.resize(binary_img_frame_polar, new_target_shape, preserve_range=True) > 0  
    #     
    #            gray_img_frame_rect = sktform.resize(gray_cells[frame_no], ((binary_img_frame_rect.shape[0], binary_img_frame_rect.shape[1])), preserve_range=True) 
    ##            binary_img_frame_polar = skmorph.remove_small_objects(binary_img_frame_polar, 100)
    #            gray_img_frame_polar = map_rect_to_polar(gray_img_frame_rect, 
    #                                                     polar_rect_map_000_rect_polar * upsample_factor, 
    #                                                     target_shape = target_shape) 
    #            
    #            gray_img_frame_polar = gaussian(gray_img_frame_polar, sigma= 3)
    #            
    #            
    #            plt.figure(figsize=(8,8))
    #            plt.imshow(binary_img_frame_polar)
    #            
    #            plt.figure(figsize=(8,8))
    #            plt.imshow(gray_img_frame_polar)
    #            
    #            plt.show()
    #            
    #            
    #            polar_frames_all.append(gray_img_frame_polar)
                
            
    #    polar_frames_all = np.array(polar_frames_all)
            if debug_viz == True:

                plt.figure(figsize=(10,10))
                plt.imshow(binary_img_frame_polar, cmap='gray')
                plt.show()    
                
                plt.figure(figsize=(10,10))
                plt.imshow(binary_img_frame_polar>0, cmap='gray')
                plt.show()  
            
            
            # =============================================================================
            #  b) gen individual cell images from rect and polar mapped separately.        
            # =============================================================================
            cells_rect_label, (cells_rect_centroids, cells_rect_areas) = parse_cells_from_rect_binary(binary_img_frame_rect, 
                                                                                                      ksize=rect_erode_ksize)
            
            if debug_viz == True:
            
                plt.figure(figsize=(10,10))
                plt.imshow(label2rgb(cells_rect_label, bg_label=0))
                plt.plot(cells_rect_centroids[:,1], 
                        cells_rect_centroids[:,0], 'k.')
                plt.show()    
    #        
            if upsample_factor == 1:  
    #            # map this into polar
                cells_rect_label = map_rect_to_polar(cells_rect_label, 
                                                      polar_rect_map_000_rect_polar, 
                                                      target_shape = polar_rect_map_000_polar_rect.shape[1:])
                cells_rect_label = np.rint(cells_rect_label).astype(np.int)
            else:
                # applying the resampling. 
                # attempting a larger image. 
                target_shape = (np.hstack(polar_rect_map_000_polar_rect.shape[1:])*1).astype(np.int)
    #            print(target_shape.shape)
            
                # we need to increase this too. 
    #            cells_rect_label = sktform.resize(cells_rect_label, ((upsample_factor*cells_rect_label.shape[0], upsample_factor*cells_rect_label.shape[1])), preserve_range=True) > 0
                
    #            print(cells_rect_label.shape)
    #            print(polar_rect_map_000_rect_polar.shape)
    #            polar_rect_map_000_rect_polar = sktform.resize(polar_rect_map_000_rect_polar, ((polar_rect_map_000_rect_polar.shape[0], upsample_factor*polar_rect_map_000_rect_polar.shape[1], upsample_factor*polar_rect_map_000_rect_polar.shape[2])), preserve_range=True).astype(np.int)
    ##            polar_rect_map_000_rect_polar[0] = np.clip(polar_rect_map_000_polar_rect[0]
    #        
                # this type of mapping creates aliasing? 
                cells_rect_label = map_rect_to_polar(cells_rect_label, 
                                                      polar_rect_map_000_rect_polar*1, 
                                                      target_shape = target_shape).astype(np.int)
                cells_rect_label = sktform.resize(cells_rect_label, np.hstack(target_shape)*upsample_factor, preserve_range=True, order=0) # using nearest.
                cells_rect_label = np.rint(cells_rect_label).astype(np.int)
                
                
            if debug_viz == True:
                
                cells_rect_centroids, cells_rect_areas = get_centroids_and_area(cells_rect_label)  
                # plt.figure(figsize=(10,10))
                # plt.title('parsed mapped rect binary')
                # plt.imshow(label2rgb(cells_polar_label, bg_label=0))
                # plt.plot(cells_polar_centroids[:,1], 
                #           cells_polar_centroids[:,0], 'k.')
                # plt.show()        
                
                plt.figure(figsize=(10,10))
                plt.title('mapped rect parsed')
                plt.imshow(label2rgb(cells_rect_label, bg_label=0))
                plt.plot(cells_rect_centroids[:,1], 
                          cells_rect_centroids[:,0], 'k.')
                plt.show()    
                
                
            ##### good up to here? 
                
                
        # # =============================================================================
        # #             distal_polar_mask = create_distal_radial_mask(cells_polar_label.shape, dist_thresh=distal_dist_polar_keep)
        #         # distal_polar_ids = np.setdiff1d(np.unique(cells_polar_label[distal_polar_mask>0]), 0)
        # # =============================================================================
        #     distal_polar_mask = create_distal_radial_mask(cells_rect_label.shape, dist_thresh=distal_dist_polar_keep)
        #     distal_polar_ids = np.setdiff1d(np.unique(cells_rect_label[distal_polar_mask>0]), 0)
                
        #     distal_cells_rect_label_mask = create_segmentations_mask(cells_rect_label, distal_polar_ids) # set these to be 0
        #     cells_rect_label[distal_cells_rect_label_mask>0] = 0 # this is fine. 
            
        #     distal_cells_rect_label_mask_binary = distal_cells_rect_label_mask>0
        #     # for this we do some closing operations then get the largest component. 
        #     distal_cells_rect_label_mask_binary_valid = skmorph.binary_closing(distal_cells_rect_label_mask_binary, 
        #                                                                         skmorph.disk(5))
        #     distal_cells_rect_label_mask_binary_valid = largest_component(distal_cells_rect_label_mask_binary_valid)[1]
        #     distal_cells_rect_label_mask_binary = np.logical_and(distal_cells_rect_label_mask_binary, 
        #                                                          distal_cells_rect_label_mask_binary_valid)
            
        #     # distal_cells_rect_label_mask_binary = distal_cells_rect_label_mask_binary*distal_polar_mask # guarantee just here. 
            
        # # =============================================================================
        # #   Preferential erosion of large regions.     
        # # =============================================================================
        #     # distal_cells_rect_label_mask_binary = skmorph.binary_erosion(distal_cells_rect_label_mask_binary, 
        #     #                                                              skmorph.disk(3))
            
        #     distal_cells_rect_label_mask_binary = erode_large_regions( distal_cells_rect_label_mask_binary, 
        #                                                               erode_ksize=3,
        #                                                               size_region = 50)
            
            
        #     distal_cells_rect_label_mask_binary_label = label(distal_cells_rect_label_mask_binary, connectivity=1)
        #     distal_polar_ids_new = np.setdiff1d(np.unique(distal_cells_rect_label_mask_binary_label), 0)
            
        #     # put the new version into the cells_rect_label
        #     # cells_rect_label[distal_cells_rect_label_mask_binary_valid>0] = 0 # delete old ids. 
            
        #     # replace this region 
        #     for jj in np.arange(len(distal_polar_ids_new)):
        #         max_cell_id = np.max(cells_rect_label)
        #         # region_mask = cells_polar_label == distal_polar_ids[jj]
        #         region_mask = distal_cells_rect_label_mask_binary_label == distal_polar_ids_new[jj]
        #         cells_rect_label[region_mask>0] = max_cell_id + 1
                
        #     # # fixing some issues?
        #     # bg = cells_rect_label == 0
        #     # bg = skmorph.binary_dilation(bg, skmorph.disk(1))
        #     # cells_rect_label_fg = cells_rect_label*np.logical_not(bg)
        #     # cells_rect_label = label(cells_rect_label_fg, connectivity=1)

        #     cells_rect_centroids, cells_rect_areas = get_centroids_and_area(cells_rect_label)        
            
        #     # plt.figure()
        #     # plt.imshow(binary_img_frame_polar)
        #     # plt.show()
        #     # print(binary_img_frame_polar.max())
            
        #     # plt.figure()
        #     # plt.imshow(binary_img_frame_polar)
        #     # plt.show()
            
            
        #     # plt.figure()
        #     # plt.imshow(cells_rect_label)
        #     # plt.show()
            
            cells_polar_label, (cells_polar_centroids, cells_polar_areas) = parse_cells_from_polar_binary(binary_img_frame_polar, ksize=polar_erode_ksize, valid_factor = 1.1)
            
            """
            visualise these. 
            """
            if debug_viz == True:
                
                plt.figure(figsize=(10,10))
                plt.title('parsed mapped rect binary')
                plt.imshow(label2rgb(cells_polar_label, bg_label=0))
                plt.plot(cells_polar_centroids[:,1], 
                          cells_polar_centroids[:,0], 'k.')
                plt.show()        
                
                
                plt.figure(figsize=(10,10))
                plt.title('mapped rect parsed')
                plt.imshow(label2rgb(cells_rect_label, bg_label=0))
                plt.plot(cells_rect_centroids[:,1], 
                          cells_rect_centroids[:,0], 'k.')
                plt.show()        
            
            # good up to here. 
            
            # =============================================================================
            #  c) compare the individual rect and polar parsed cells, using the polar as the main reference and pulling in rect for replacement where appropriate. 
            # =============================================================================
            distal_polar_mask = create_distal_radial_mask(cells_polar_label.shape, dist_thresh=distal_dist_polar_keep[ii])
            distal_polar_ids = np.setdiff1d(np.unique(cells_polar_label[distal_polar_mask>0]), 0) # this should still be good? 
            
            # # o i see ... this explicitly avoids comparison of distal....... -> i think this is the one that messes things up?
            # merged_mask = compare_polar_rect_binaries(cells_polar_label, 
            #                                           cells_rect_label, 
            #                                           ksize_rect=3, 
            #                                           min_area=min_cell_area, 
            #                                           exclude_ids=distal_polar_ids)
            
            merged_mask = cells_rect_label.copy()
            
            if distal_correct:
                # =============================================================================
                #   c ii) as above can erroneously correct the distal tip -> here we put back the distal tip part in case its missed. 
                # =============================================================================
                """
                polar -> get the distal cell ids from polar that is 100% correct. 
                """
                distal_polar_mask = create_distal_radial_mask(cells_polar_label.shape, dist_thresh=distal_dist_polar_keep[ii])
                
                """
                inject holly's distal in here. # we don't need this necessarily. 
                binary_img_frame_polar = np.uint8(sktform.resize(polar_cells[frame_no], binary_img_frame_polar.shape, preserve_range=True))
                print(binary_img_frame_polar.max())
                """
                cell_polar_binary_ref = np.uint8(sktform.resize(polar_cells[frame_no], binary_img_frame_polar.shape, preserve_range=True))
                cells_polar_label_ref, _ = parse_cells_from_polar_binary(cell_polar_binary_ref, 
                                                                          ksize=polar_erode_ksize, 
                                                                          valid_factor = 1.1)
                
                # distal_polar_ids = np.setdiff1d(np.unique(cells_polar_label[distal_polar_mask>0]), 0)
                distal_polar_ids = np.setdiff1d(np.unique(cells_polar_label_ref[distal_polar_mask>0]), 0)
                
                # construct the identified region. 
                # distal_polar_cell_mask = create_segmentations_mask(cells_polar_label, distal_polar_ids)
                distal_polar_cell_mask = create_segmentations_mask(cells_polar_label_ref, distal_polar_ids)
                distal_polar_cell_ids_counts = np.hstack([np.sum(distal_polar_cell_mask==cc) for cc in distal_polar_ids])
                
                distal_polar_ids = distal_polar_ids[ distal_polar_cell_ids_counts > min_cell_area] # this is good to have a mininum cell area. 
                # regenerate the mask.
                # distal_polar_cell_mask = create_segmentations_mask(cells_polar_label, distal_polar_ids)
                distal_polar_cell_mask = create_segmentations_mask(cells_polar_label_ref, distal_polar_ids)
                
                """
                rect polar -> locate the equivalent region. 
                """
                # get the corresponding region in rect. 
                distal_rect_polar_cell_region = merged_mask[distal_polar_mask>0]
                distal_rect_polar_cell_ids = np.setdiff1d(np.unique(distal_rect_polar_cell_region), 0)
        
                # count the occurrence and remove very small occurrences
                distal_rect_polar_cell_ids_counts = np.hstack([np.sum(distal_rect_polar_cell_region==cc) for cc in distal_rect_polar_cell_ids])
        
                
                # based on this construct the filtered region mask
        #        distal_rect_polar_cell_ids = distal_rect_polar_cell_ids[ distal_rect_polar_cell_ids_counts > min_cell_area]
                distal_rect_cell_mask = create_segmentations_mask(merged_mask, distal_rect_polar_cell_ids) 
                
                if debug_viz:
                    
                    plt.figure(figsize=(10,10))
                    plt.imshow(label2rgb(distal_polar_cell_mask, bg_label=0))
                    plt.show()
                    
                    plt.figure(figsize=(10,10))
                    plt.imshow(label2rgb(distal_rect_cell_mask, bg_label=0))
                    plt.show()
                    
                """
                replace the distal region by heuristic .... 
                """
                merged_mask[distal_rect_cell_mask>0] = 0 # delete old ids. 
                # replace this region 
                for jj in np.arange(len(distal_polar_ids)):
                    max_cell_id = np.max(merged_mask)
                    # region_mask = cells_polar_label == distal_polar_ids[jj]
                    region_mask = cells_polar_label_ref == distal_polar_ids[jj]
                    merged_mask[region_mask>0] = max_cell_id + 1
                    
                    
                # have another check.
                if debug_viz:
                    
                    plt.figure(figsize=(10,10))
                    plt.imshow(label2rgb(merged_mask, bg_label=0))
                    plt.show()
            
            
            # grab the centroids for display 
            merged_mask_centroids, merged_mask_areas = get_centroids_and_area(merged_mask)
            
            if debug_viz == True:
                plt.figure(figsize=(10,10))
                plt.imshow(merged_mask, cmap='coolwarm')
        #        plt.imshow(distal_polar_mask, cmap='coolwarm', alpha=0.5)
                plt.show()
                
                
                color_merged_mask = label2rgb(merged_mask, bg_label=0)
        
                plt.figure(figsize=(10,10))
                plt.imshow(color_merged_mask)
                plt.plot(merged_mask_centroids[:,1], 
                          merged_mask_centroids[:,0],'k.')
        #        plt.imshow(distal_polar_mask, cmap='coolwarm', alpha=0.5)
                plt.show()
            
            
            # =============================================================================
            #  d) attempt autofixing of the 180 degrees 'seam' 
            # =============================================================================
            merged_mask_refine = refine_left_seam_xaxis(merged_mask, 
                                                        cells_polar_label, 
                                                        min_occur=2, 
                                                        min_pair_frequency=2, 
                                                        start_x_factor=1.05, refine_strip_size=5, use_polar=False)
            
            color_merged_mask_refine = label2rgb(merged_mask_refine, bg_label=0)
            
            if debug_viz == True:
                plt.figure(figsize=(10,10))
                plt.imshow(color_merged_mask_refine)
    #            plt.imshow(distal_polar_mask, cmap='coolwarm', alpha=0.5)
                plt.show()
            
            # =============================================================================
            #  e) recompute centroids and areas to clean up. 
            #       1 - remove too small / too large regions
            #       2 - remove non-solid regions.  =============================================================================
            
            """
            do regrowing here. 
            """
    #        merged_mask_refine = grow_polar_regions_nearest(merged_mask_refine, metric='euclidean', debug=debug_viz)
            
            merged_mask_refine = filter_min_max_areas(merged_mask_refine, min_thresh=min_cell_area, max_thresh=np.inf)    
            # this bit is very weird. 
            # merged_mask_refine = grow_polar_regions_nearest(merged_mask_refine, metric='manhattan', debug=debug_viz)
            if upsample_factor == 1:
                unwrap_params_polar_ = unwrap_params_polar.copy()
            else:
                unwrap_params_polar_ = sktform.resize(unwrap_params_polar, (merged_mask_refine.shape[0], merged_mask_refine.shape[1], unwrap_params_polar.shape[2]), preserve_range=True)
            
            merged_mask_refine = grow_polar_regions_nearest_3D(merged_mask_refine, 
                                                                unwrap_params_polar_, 
                                                                metric='manhattan', debug=debug_viz)
            merged_mask_refine = filter_min_max_areas(merged_mask_refine, min_thresh=min_cell_area, max_thresh=max_cell_area)        
            merged_mask_refine = filter_non_solid_areas(merged_mask_refine, thresh=0.25) # this removes the outer? 
             
    #        merged_mask_refine = grow_polar_regions_nearest(merged_mask_refine, metric='euclidean', debug=debug_viz)
    #        merged_mask_refine = filter_min_max_areas(merged_mask_refine, min_thresh=min_cell_area, max_thresh=max_cell_area)        
    #        merged_mask_refine = grow_polar_regions_nearest(merged_mask_refine, metric='euclidean', debug=debug_viz)
    #        merged_mask_refine = filter_non_solid_areas(merged_mask_refine, thresh=0.5)

            """
            downsample? -> can we do this safely... 
            """
    #        if upsample_factor == 1:
    #            continue
    #        else:
    #            merged_mask_refine = sktform.resize(merged_mask_refine, 
    #                                                np.hstack(merged_mask_refine.shape)//2, 
    #                                                preserve_range=True, order=0)
    #            merged_mask_refine = merged_mask_refine.astype(np.int)
            
            color_merged_mask_refine = label2rgb(merged_mask_refine, bg_label=0)
            merged_mask_refine_centroids, merged_mask_refine_areas = get_centroids_and_area(merged_mask_refine)
            
            
            if debug_viz == True:
                plt.figure(figsize=(12,12))
                plt.imshow(color_merged_mask_refine, alpha=1.)
                plt.plot(merged_mask_refine_centroids[:,1], 
                          merged_mask_refine_centroids[:,0], 'wo', mec='k')
                plt.show()
                
                plt.figure(figsize=(12,12))
                plt.imshow(find_boundaries(merged_mask_refine), alpha=1.)
                plt.plot(merged_mask_refine_centroids[:,1], 
                          merged_mask_refine_centroids[:,0], 'wo', mec='k')
                plt.show()
            
            
            # =============================================================================
            #  f) generate new binary from the refined mask.
            # =============================================================================
            
    #        # step 1: regrow the regions so there is no gap. 
    #        merged_mask_refine_regrow = grow_polar_regions_nearest(merged_mask_refine, metric='manhattan')
            
            # step 2: recomputing a 1-line border. 
            merged_mask_refine_regrow_cell_boundary = find_boundaries(merged_mask_refine)
            
            
            if debug_viz == True:
                
                plt.figure(figsize=(12,12))
                plt.imshow(merged_mask_refine_regrow_cell_boundary, cmap='gray', alpha=1.)
                plt.show()
            
            
            cells_refine_all.append(merged_mask_refine) # create an empty array to save the refined. 
            cells_boundary_refine_all.append(merged_mask_refine_regrow_cell_boundary)
            
        cells_refine_all = np.array(cells_refine_all)
        cells_boundary_refine_all = np.array(cells_boundary_refine_all)

        savefile_cells = os.path.join(outfolder, 'cells_'+basefname+'.tif')
        print('saving, ', savefile_cells)
        savefile_binary = os.path.join(outfolder, 'binary_'+basefname+'.tif')
        print('saving, ', savefile_binary)
        
        
        # save these specifically for the single file.
        skio.imsave(savefile_cells, 
                    np.uint16(cells_refine_all[0]))
        skio.imsave(savefile_binary, 
                    np.uint8(255*cells_boundary_refine_all[0]))
        
#     skio.imsave('%s_auto_reedit-up2-2x-direct_distal.tif' %(basefname), np.uint16(cells_refine_all))
#     skio.imsave('%s_auto_reedit-up2_binary-2x-direct_distal.tif' %(basefname), np.uint8(255*cells_boundary_refine_all))
    
    
    
    
    
    
    