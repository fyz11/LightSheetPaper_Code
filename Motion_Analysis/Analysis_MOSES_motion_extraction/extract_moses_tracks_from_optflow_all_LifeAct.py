#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:26:50 2019

@author: felix
"""

def meantracks2D_to_3D( meantracks, ref_map):
    """
    ref map gives the matching corresponding look up table in 3D coordinates. 
    """
    
    meantracks3D = ref_map[meantracks[:,:,0], meantracks[:,:,1],:]
    
    return meantracks3D
    

def exponential_decay_correction_time(vid, average_fnc=None, f_scale=100.):
    """
    Fits an equation of form y=Ae^(Bt)+C on the mean intensities of video
    """
    from scipy.stats import linregress
    from scipy.optimize import least_squares
    
    if average_fnc is None:
        average_fnc = np.mean
        
        
    I_vid = np.hstack([average_fnc(v) for v in vid])
    I_time = np.arange(len(I_vid))
    
    # fit equation. y =A*e^(-Bt)
    log_I_vid = np.log(I_vid)
    slope, intercept, r_value, p_value, std_err = linregress(I_time, log_I_vid)

    # initial fit. 
    A = np.exp(intercept)
    B = slope
    
    # refined robust fitting. 
    def exp_decay(t,x):
        return (x[0] * np.exp(x[1] * t) + x[2])
        
    def res(x, t, y):
        return exp_decay(t,x) - y
    
    x0 = [A, B, 0]
    res_robust = least_squares(res, x0, loss='soft_l1', f_scale=f_scale, args=(I_time, I_vid))
        
    robust_y = exp_decay(I_time, res_robust.x)
    correction = float(robust_y[0]) / robust_y
    
#    plt.figure()
#    plt.plot(robust_y)
#    plt.show()
    vid_corrected = np.zeros(vid.shape, np.float32)
    
    for frame in range(vid.shape[0]):
        vid_corrected[frame, ...] = vid[frame, ...] * correction[frame]
    
    return vid_corrected, res_robust.x


def locate_unwrapped_im_folder(rootfolder, key='geodesic-rotmatrix'):
    
    found_dir = []
    
    for dirName, subdirList, fileList in os.walk(rootfolder):
#        print('Found directory: %s' % dirName)
        for directory in subdirList:
            if key in directory and 'unwrap_params' not in directory:
                found_dir.append(os.path.join(dirName, directory))
                
    return found_dir


if __name__=="__main__":
    
    import numpy as np 
    import pylab as plt 
    import glob 
    import os 
    import scipy.io as spio
    from skimage.io import imsave
    
    import MOSES.Utility_Functions.file_io as fio
    from MOSES.Optical_Flow_Tracking.optical_flow import farnebackflow, TVL1Flow, DeepFlow
    from MOSES.Optical_Flow_Tracking.superpixel_track import compute_vid_opt_flow, compute_grayscale_vid_superpixel_tracks, compute_grayscale_vid_superpixel_tracks_w_optflow
    from MOSES.Visualisation_Tools.track_plotting import plot_tracks
    from flow_vis import make_colorwheel, flow_to_color
    
    from skimage.filters import threshold_otsu
    from tqdm import tqdm 
    
    optical_flow_params = dict(pyr_scale=0.5, levels=5, winsize=25, iterations=5, poly_n=5, poly_sigma=1.2, flags=0)
    
    
#    rootfolder = '/media/felix/Srinivas4/LifeAct'
    rootfolder = '/media/felix/Srinivas4/MTMG-TTR'
#    rootfolder = '/media/felix/Srinivas4/MTMG-HEX'
#    rootfolder = '/media/felix/Srinivas4/MTMG-HEX/new_hex'
# =============================================================================
#     1. Load all embryos 
# =============================================================================
    embryo_folders = np.hstack([os.path.join(rootfolder,folder) for folder in os.listdir(rootfolder) if os.path.isdir(os.path.join(rootfolder, folder)) and 'new_hex' not in folder])
    
# =============================================================================
#     2. find valid embryo folders i.e. has a geodesic-rotmatrix. 
# =============================================================================
    n_spixels = 1000 # set the number of pixels. 
    
    for embryo_folder in embryo_folders[:]:
        
        # find the geodesic-rotmatrix. 
        unwrapped_folder = locate_unwrapped_im_folder(embryo_folder, key='geodesic-rotmatrix')
        
        if len(unwrapped_folder) == 1:
            print('processing: ', unwrapped_folder[0])
            
            unwrapped_folder = unwrapped_folder[0]
            # determine the savefolder. 
            saveflowfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'optflow')
            savetrackfolder = os.path.join(embryo_folder, 'MOSES_analysis', 'tracks_all_time')
            fio.mkdir(savetrackfolder) # create this folder. 
            
            unwrap_im_files = glob.glob(os.path.join(unwrapped_folder, '*.tif'))
            
            for ii in tqdm(range(len(unwrap_im_files))[:]):
                
                unwrap_im_file = unwrap_im_files[ii]
                print(unwrap_im_file)
                vid = fio.read_multiimg_PIL(unwrap_im_file)
                
                max_vid_im = vid.max(axis=0)
            
                """
                1. load the flow file. 
                """
                flowfile = os.path.join(saveflowfolder, 'deepflow_'+os.path.split(unwrap_im_file)[-1].replace('.tif', '.mat'))
                flow = spio.loadmat(flowfile)['flow']
#                flow_mag = np.sqrt(flow[...,0] ** 2 + flow[...,1]**2)

                # use the image pixels to determine where the tracks should be taken.                 
#                flow_mag_mask = max_vid_im < 2.5
                flow_mag_mask = max_vid_im < 1 # this is a guide only, remember we are just training to recognise the movement of the VE movement. 
                
                """
                2. set the number of superpixels and produce tracks. 
                """
                meantracks = compute_grayscale_vid_superpixel_tracks_w_optflow(vid, flow, 
                                                                               n_spixels=n_spixels, 
                                                                               direction='F', 
                                                                               dense=False, 
                                                                               mindensity=1)
                
#                # mask out the irrelevant tracks by the image intensity which deepflow method suffers from. 
                meantracks_filt = flow_mag_mask[meantracks[:,0,0], 
                                                meantracks[:,0,1]]
                meantracks_ = meantracks.copy()
                meantracks_[meantracks_filt] = meantracks_[meantracks_filt,0][:,None]
                
##                flow, meantracks = compute_grayscale_vid_superpixel_tracks(vid, 
##                                                                           opt_flow_fn=farnebackflow, 
##                                                                           n_spixels=n_spixels, 
##                                                                           mask=None, 
##                                                                           adaptive_mask=False, 
##                                                                           ksize_mask=3, 
##                                                                           direction='F', 
##                                                                           dense=False, 
##                                                                           mindensity=1, 
##                                                                           params=optical_flow_params)

                fig, ax = plt.subplots(figsize=(10,10))
                ax.imshow(vid[0], cmap='gray')
                plot_tracks(meantracks_, ax=ax, color='r')
                plt.show()
                 
                """
                3. save the tracks with min_I?
                """
                trackfile = os.path.join(savetrackfolder, 'deepflow-meantracks-%d_' %(n_spixels)+os.path.split(unwrap_im_file)[-1].replace('.tif', '.mat'))
                print(trackfile)
                spio.savemat(trackfile, {'meantracks': meantracks})
                
                
                

    
##    infolder = '/media/felix/Srinivas4/LifeAct/L401_1View/unwrapped/geodesic-rotmatrix'
##    infolder = '/media/felix/Srinivas4/MTMG-HEX/L926_TP1-45/unwrapped/geodesic-rotmatrix'
#    infolder = '/media/felix/Srinivas4/MTMG-TTR/L860/unwrapped/geodesic-rotmatrix'
##    infolder = '/media/felix/Srinivas4/LifeAct/L491_Emb2_TP1-100/unwrapped/geodesic-rotmatrix'
#    videofiles = glob.glob(os.path.join(infolder, '*.tif'))
#    
#    for videofile in videofiles[-4:-3]:
##    for videofile in videofiles[-3:-2]:
#        vid = fio.read_multiimg_PIL(videofile)
#        
##         do the correction of intensity?
#        vid_correct, fit_params = exponential_decay_correction_time(vid[::-1], average_fnc=None, f_scale=100.)
#        vid_correct = vid_correct[::-1]
#        
#        vid_correct = np.uint8(np.clip(255*vid_correct/vid_correct.max(), 0, 255))
#        
##        imsave('test_exp_correct-meannorm.tif', vid_correct)
#        
##        fig, ax = plt.subplots()
##        ax.plot([np.mean(v) for v in vid], color='k', label='original')
##        ax.plot([np.mean(v) for v in vid_correct], color='g', label='corrected')
##        plt.show()
#        flow_original = compute_vid_opt_flow(vid, DeepFlow)
#        flow_correct = compute_vid_opt_flow(vid_correct, DeepFlow)
#        flow_farneback_original = compute_vid_opt_flow(vid, 
#                                                       farnebackflow, 
#                                                       mask=None, 
#                                                       adaptive=False, 
#                                                       ksize=3, 
#                                                       params=optical_flow_params)
#        flow_farneback_correct = compute_vid_opt_flow(vid_correct, 
#                                                      farnebackflow, 
#                                                      mask=None, 
#                                                      adaptive=False, 
#                                                      ksize=3, 
#                                                      params=optical_flow_params)
#        
##        flow_mag = np.sqrt(flow_correct.mean(axis=0)[...,0] ** 2+ flow_correct.mean(axis=0)[...,1] ** 2)
#        flow_mag_correct = np.sqrt(flow_farneback_correct.mean(axis=0)[...,0] ** 2+ flow_farneback_correct.mean(axis=0)[...,1] ** 2)
#        
##        flow_farneback = compute_vid_opt_flow(vid, farnebackflow, mask=None, adaptive=False, ksize=3, params=optical_flow_params)
##        flow_farneback_mag = np.sqrt(flow_farneback[...,0] ** 2+ flow_farneback[...,1] ** 2)
#        
##        mean_flow = flow_correct.mean(axis=0)
#        X, Y = np.indices(flow_correct.shape[1:-1])
#    
#        sampling = 8
#        x_pos = X[::sampling,::sampling]
#        y_pos = Y[::sampling,::sampling]
#        
#        fig, ax = plt.subplots(figsize=(15,15))
#        ax.imshow(vid_correct[0])
#    #    ax.quiver(y_pos, x_pos, flow_diff[::sampling,::sampling,0], -flow_diff[::sampling,::sampling,1])
#        ax.quiver(y_pos, x_pos, 
#                  flow_correct.mean(axis=0)[::sampling,::sampling,0], 
#                  -flow_correct.mean(axis=0)[::sampling,::sampling,1])
#    #    ax.quiver(y_pos, x_pos, flow_diff.mean(axis=0)[::sampling,::sampling,0], -flow_diff.mean(axis=0)[::sampling,::sampling,1])
#    #    plt.xlim([0, flow_diff.shape[1]])
#    #    plt.ylim([flow_diff.shape[0], 0])
#    #    plt.show()
#        plt.xlim([0, flow_correct[0].shape[1]])
#        plt.ylim([flow_correct[0].shape[0], 0])
#        plt.show()
#        
#        
#        X, Y = np.indices(flow_correct.shape[1:-1])
#    
#        sampling = 8
#        x_pos = X[::sampling,::sampling]
#        y_pos = Y[::sampling,::sampling]
#        
#        fig, ax = plt.subplots(figsize=(15,15))
#        ax.imshow(vid_correct[0])
#    #    ax.quiver(y_pos, x_pos, flow_diff[::sampling,::sampling,0], -flow_diff[::sampling,::sampling,1])
#        ax.quiver(y_pos, x_pos, 
#                  flow_farneback_correct.mean(axis=0)[::sampling,::sampling,0], 
#                  -flow_farneback_correct.mean(axis=0)[::sampling,::sampling,1])
#    #    ax.quiver(y_pos, x_pos, flow_diff.mean(axis=0)[::sampling,::sampling,0], -flow_diff.mean(axis=0)[::sampling,::sampling,1])
#    #    plt.xlim([0, flow_diff.shape[1]])
#    #    plt.ylim([flow_diff.shape[0], 0])
#    #    plt.show()
#        plt.xlim([0, flow_correct[0].shape[1]])
#        plt.ylim([flow_correct[0].shape[0], 0])
#        plt.show()
    
#    vidfolder = '/media/felix/SAMSUNG/Shankar/Holly_Data/Hex-data/unwrapped_smooth/bleach_corrected'
#    videofiles = glob.glob(os.path.join(vidfolder, '*.tif'))
#    
#    
#    # look at the Hex. 
#    hexfiles = videofiles[1:2]
#    
#    
#    for hexfile in hexfiles:
#        hexfile = '/media/felix/Elements/Shankar/Unwrapped Holly/L491_Emb2_unwrapped1.tif'
#        _, vid = fio.read_multiimg_stack(hexfile)
#        
#        optflow, meantracks = compute_grayscale_vid_superpixel_tracks(vid, 
#                                                                      opt_flow_fn=farnebackflow, 
#                                                                      n_spixels=1000, 
#                                                                      mask=None, 
#                                                                      adaptive_mask=False, 
#                                                                      ksize_mask=3, 
#                                                                      direction='F', 
#                                                                      dense=False, 
#                                                                      mindensity=1,
#                                                                      params=optical_flow_params)
#        
#        fig, ax = plt.subplots(figsize=(10,10))
#        ax.imshow(vid[0])
#        ax.plot(meantracks[:,0,1], 
#                meantracks[:,0,0], 'k.')
#        plot_tracks(meantracks, ax, color='r', lw=1)
#        plt.show()
#        
#        
#        mean_flow = flow_to_color(optflow.mean(axis=0))
#        
#        plt.figure()
#        plt.imshow(mean_flow)
#        plt.show()
#        
#        
#        # create a colour wheel
#        wheel = np.zeros((100,100,2)); wheel_x, wheel_y = np.indices(wheel[...,0].shape)
#        wheel[...,0] = wheel_x - wheel.shape[0]//2 
#        wheel[...,1] = wheel_y - wheel.shape[1]//2 
#        
#        wheel_color = flow_to_color(wheel, convert_to_bgr=False)
#        
#        plt.figure()
#        plt.imshow(wheel_color.transpose(1,0,2))
#        plt.show()
#        
## =============================================================================
##      Arrow plots of motion    
## =============================================================================
#        X, Y = np.indices(optflow[0][...,0].shape)
#    
#        sampling = 10
#        x_pos = X[::sampling,::sampling]
#        y_pos = Y[::sampling,::sampling]
#        
#        fig, ax = plt.subplots(figsize=(15,15))
#        ax.imshow(vid[0])
#    #    ax.quiver(y_pos, x_pos, flow_diff[::sampling,::sampling,0], -flow_diff[::sampling,::sampling,1])
#        ax.quiver(y_pos, x_pos, 
#                  optflow.mean(axis=0)[::sampling,::sampling,0], 
#                  -optflow.mean(axis=0)[::sampling,::sampling,1])
#    #    ax.quiver(y_pos, x_pos, flow_diff.mean(axis=0)[::sampling,::sampling,0], -flow_diff.mean(axis=0)[::sampling,::sampling,1])
#    #    plt.xlim([0, flow_diff.shape[1]])
#    #    plt.ylim([flow_diff.shape[0], 0])
#    #    plt.show()
#        plt.xlim([0, optflow[0].shape[1]])
#        plt.ylim([optflow[0].shape[0], 0])
#        plt.show()
#            
#        speed = np.hstack([np.linalg.norm([np.mean(f[...,0]), np.mean(f[...,1])]) for f in optflow])
##        speed = np.hstack([np.mean(f[:,:,1]) for f in optflow])
#        
#        
#        track_disps = meantracks[:,:] - meantracks[:,0][:,None]
#        track_speed = np.linalg.norm(track_disps, axis=-1)
#        av_track_speed = np.mean(track_speed, axis=0)
#        
#        fig, ax = plt.subplots()
#        plt.plot(speed)
#        plt.plot(av_track_speed)
#        plt.show()
#        
#        
## =============================================================================
##       grab the 3d geometry
## =============================================================================
##        unwrap_params = spio.loadmat('/media/felix/SAMSUNG/Shankar/Holly_Data/Hex-data/step7-smooth_mtmg_man1-ve/')
#        unwrap_params = spio.loadmat('/media/felix/Elements/Shankar/Unwrapped Holly/unwrap_params_tp50.mat')
#        
#        ref_map = unwrap_params['ref_map']
#        ref_space = unwrap_params['ref_space']
#        ref_map = ref_map.reshape(ref_space.shape[0], 
#                                  ref_space.shape[1],
#                                  3)
#        
#        
#        meantracks3D = meantracks2D_to_3D( meantracks, ref_map)
#        track_disps_3D = meantracks3D[:,:,] - meantracks3D[:,0][:,None]
#        track_speed_3D = np.linalg.norm(track_disps_3D, axis=-1)
#        av_track_speed_3D = np.mean(track_speed_3D, axis=0)
#        
#        fig, ax = plt.subplots()
#        plt.plot(av_track_speed)
#        plt.plot(av_track_speed_3D)
#        plt.show()
#        
#        
#        """
#        Plot 3D (x,y,z)
#        """
#        from mpl_toolkits.mplot3d import Axes3D
#        
#        fig = plt.figure()
#        ax = fig.add_subplot(111, projection='3d')
#        ax.view_init(elev=90., azim=0)
#        for kk in range(len(meantracks3D)):
#            ax.plot(meantracks3D[kk,:,0], 
#                    meantracks3D[kk,:,1],
#                    meantracks3D[kk,:,2])
#        ax.set_aspect('equal')
##        ax.set_xlim([100,800])
##        ax.set_ylim([])
#        plt.show()
#        
#        
## =============================================================================
##     denoising into states using hdp-hmm
## =============================================================================
#        import pyhsmm
#        import pyhsmm.basic.distributions as distributions
#        from pyhsmm.util.text import progprint_xrange
#        from scipy.signal import savgol_filter
#        
#        obs_dim = 1
#        Nmax = 5
#        
#        obs_hypparams = {'mu_0':np.zeros(obs_dim),
#                'sigma_0':np.eye(obs_dim),
#                'kappa_0':0.25,
#                'nu_0':obs_dim+2}
#        
#        obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
#        posteriormodel = pyhsmm.models.WeakLimitStickyHDPHMM(
#                kappa=1.,alpha=6.,gamma=6.,init_state_concentration=1.,
#                obs_distns=obs_distns)
##        posteriormodel.add_data(speed)
#        posteriormodel.add_data(np.diff(savgol_filter(av_track_speed_3D,5,1)))
#        
#        for idx in progprint_xrange(100):
#            posteriormodel.resample_model()
#        
##        obs_hypparams = {'mu_0':np.zeros(obs_dim),
##                        'sigma_0':np.eye(obs_dim),
##                        'kappa_0':0.3,
##                        'nu_0':obs_dim+5}
##        dur_hypparams = {'alpha_0':2*30,
##                         'beta_0':2}
##        
##        obs_distns = [distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
##        dur_distns = [distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]
##        
##        posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
##                alpha=.1,gamma=.1, # better to sample over these; see concentration-resampling.py
##                init_state_concentration=6., # pretty inconsequential
##                obs_distns=obs_distns,
##                dur_distns=dur_distns)
##    
##    
##        posteriormodel.add_data(speed,trunc=60)
##        for idx in progprint_xrange(100):
##            posteriormodel.resample_model()        
#        states = posteriormodel.stateseqs[0]
#        
#        fig, ax = plt.subplots()
#        time = np.arange(len(states))
#        for state in np.unique(states):
#            ax.plot(time[states==state], 
#                    av_track_speed[:-1][states==state], 'o',
#                    label=str(state))
#        plt.legend()
#    
#    
    
