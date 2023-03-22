# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 19:42:08 2021

@author: fyz11
"""

def load_pickle(ff):
    
    with open(ff, 'rb') as filehandler:
        reloaded_data = pickle.load(filehandler)
        
    return reloaded_data

if __name__=="__main__":
    
    import numpy as np 
    import pylab as plt 
    import networkx as nx 
    import glob 
    import os 
    import itertools
    import pickle
    from scipy.spatial.distance import squareform
    import seaborn as sns 
    from scipy.stats import pearsonr
    
    # analysis_model_save_folder = 'Lasso_Feature_Analysis-Anterior_Posterior-8quads'
    # analysis_model_save_folder = 'Lasso_Feature_Analysis-Anterior_Posterior-8quads-PreMig'
    # analysis_model_save_folder = 'Lasso_Feature_Analysis-Anterior_Posterior-8quads-PreMig-0.01'
    # analysis_model_save_folder = 'Lasso_Feature_Analysis-Anterior_Posterior-8quads-Mig-0.01'
    # analysis_model_save_folder = 'Lasso_Feature_Analysis-Anterior_Posterior-8quads-PreMig-0.05'
    # analysis_model_save_folder = 'Lasso_Feature_Analysis-Anterior_Posterior-8quads-Mig-0.05'
    # analysis_model_save_folder = 'Lasso_Feature_Analysis-Anterior_Posterior-8quads-PreMig-Auto'
    # analysis_model_save_folder = 'Lasso_Feature_Analysis-Anterior_Posterior-8quads-Mig-Auto'
    
    # analysis_model_save_folder = 'Corr_Feature_Analysis-Anterior_Posterior-8quads-dynamic'
    # analysis_model_save_folder = 'Corr_Feature_Analysis-Anterior_Posterior-8quads'
    # analysis_model_save_folder = 'Corr_Feature_Analysis-Anterior_Posterior-32quads_static'
    # analysis_model_save_folder = 'Corr_Feature_Analysis-Anterior_Posterior-32quads-static' # seem to recover the embryo geometry.... 
    
    
    """
    use these 
    """
    # analysis_model_save_folder = 'Corr_Feature_Analysis-Anterior_Posterior-8quads'
    # analysis_model_save_folder = 'Corr_Feature_Analysis-Anterior_Posterior-8quads-dynamic'
    # analysis_model_save_folder = 'Corr_Feature_Analysis-Anterior_Posterior-32quads-static'
    # analysis_model_save_folder = 'Corr_Feature_Analysis-Anterior_Posterior-32quads-dynamic'
    # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-8quads-static'
    # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-8quads-dynamic'
    # analysis_model_save_folder = 'Lasso0-1_Feature_Analysis-Anterior_Posterior-8quads-static'
    # analysis_model_save_folder = 'Lasso0-01_Feature_Analysis-Anterior_Posterior-8quads-static'
    # analysis_model_save_folder = 'Lasso0-1_Feature_Analysis-Anterior_Posterior-8quads-dynamic'
    # analysis_model_save_folder = 'Lasso0-01_Feature_Analysis-Anterior_Posterior-8quads-dynamic'
    # analysis_model_save_folder = 'Lasso0-1_Feature_Analysis-Anterior_Posterior-8quads-static_2'
    # analysis_model_save_folder = 'Lasso0-01_Feature_Analysis-Anterior_Posterior-8quads-static_2'
    # analysis_model_save_folder = 'Lasso0-01_Feature_Analysis-Anterior_Posterior-8quads-dynamic_2'
    
    # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-8quads-static_2'
    # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-8quads-dynamic_2'
    
    # analysis_model_save_folder = 'Corr_Feature_Analysis-Anterior_Posterior-8quads-static-2'
    # analysis_model_save_folder = 'Corr_Feature_Analysis-Anterior_Posterior-8quads-dynamic-2'
    # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-8quads-static_2'
    # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-8quads-dynamic_2'
    # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-8quads-static_2-rescale_coef'
    # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-8quads-dynamic_2-rescale_coef'
    # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-8quads-static_2-stdscale_coef'
    # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-8quads-dynamic_2-stdscale_coef'
    # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-8quads-static_2-rescale_coef_post'
    # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-8quads-dynamic_2-rescale_coef_post'
    
    # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-8quads-static_2-with_coef-std'
    # analysis_model_save_folder = 'sqrtLasso_Feature_Analysis-Anterior_Posterior-8quads-dynamic_2-with_coef-std'
    
    # analysis_model_save_folder = '2021-02-24_sqrtLasso_Feature_Analysis-AP-8quads-static-with_coef-std_realxyz'
    # analysis_model_save_folder = '2021-02-24_sqrtLasso_Feature_Analysis-AP-8quads-dynamic-with_coef-std_realxyz'
    # analysis_model_save_folder = '2021-02-24_sqrtLasso_Feature_Analysis-AP-8quads-static-with_coef-std_realxyz-nocum'
    # analysis_model_save_folder = '2021-02-24_sqrtLasso_Feature_Analysis-AP-8quads-dynamic-with_coef-std_realxyz-nocum'
    
    # analysis_model_save_folder = '2021-02-24_sqrtLasso_Feature_Analysis-AP-8quads-static-with_coef-std_ref'
    # analysis_model_save_folder = '2021-02-24_sqrtLasso_Feature_Analysis-AP-8quads-dynamic-with_coef-std_ref'
    # analysis_model_save_folder = '2021-02-24_sqrtLasso_Feature_Analysis-AP-8quads-dynamic-with_coef-std_ref'
    
    
    """
    This is intended in the paper? ---- update 03/01/2023. 
    """
    # analysis_model_save_folder = '2021-05-19_sqrtLasso_Feature_Analysis-AP-8quads-static-with_coef-std-nooutliers'
    # analysis_model_save_folder = '2021-05-19_sqrtLasso_Feature_Analysis-AP-8quads-dynamic-with_coef-std-nooutliers'
    
    """
    This is the updated with all 5 lifeact embryos. 
    """
    analysis_model_save_folder = '2023-01-03_sqrtLasso_Feature_Analysis-AP-8quads-dynamic-with_coef-std_ref'
    # analysis_model_save_folder = '2023-01-03_sqrtLasso_Feature_Analysis-AP-8quads-dynamic-with_coef-std_ref'
    
    labellingtype = 'dynamic'
    # labellingtype = 'static'
    
    
    # e.g. migration. 
    # parse the networks -> compute the graph edit distance. 
    all_network_files = np.hstack(glob.glob(os.path.join(analysis_model_save_folder,'*.p')))
    all_network_fnames = np.hstack([os.path.split(ff)[-1].split('.p')[0] for ff in all_network_files])
    # all_network_fnames_select = np.hstack([ff for ff in np.arange(len(all_network_fnames)) if 'Pre-migration' in all_network_fnames[ff]])
    # all_network_fnames_select = np.hstack([ff for ff in np.arange(len(all_network_fnames)) if 'Migration to Boundary' in all_network_fnames[ff]])
    
    if labellingtype == 'dynamic':
        all_network_fnames_select = np.hstack([ff for ff in np.arange(len(all_network_fnames)) if '_dynamic' in all_network_fnames[ff]])
    if labellingtype == 'static': 
        all_network_fnames_select = np.hstack([ff for ff in np.arange(len(all_network_fnames)) if '_static' in all_network_fnames[ff]])
    # all_network_fnames_select = np.arange(len(all_network_fnames))
    
    all_network_files = all_network_files[all_network_fnames_select]
    all_network_fnames = all_network_fnames[all_network_fnames_select]
    # all_network_fnames = np.hstack([ff for ff in all_network_fnames if 'Migration to Boundary' in ff])
    # all_metadata = np.array([ff.split('')])
    
    
    print('=============')
    print(all_network_files)
    print('=============')
    
# =============================================================================
#     consider instead the matrix as a feature vector. 
# =============================================================================
    
    G = [load_pickle(ff)['lasso_matrix'] for ff in all_network_files]
    G_feats_std = np.array([load_pickle(ff)['X_feats_std'] for ff in all_network_files])
    mean_G_feats_std = np.nanmean(G_feats_std, axis=0) # do we have to do the coefficient separately for each stage? 
    
    mean_G_feats_std_mig = np.nanmean(G_feats_std[:len(G)//2], axis=0)
    mean_G_feats_std_pre = np.nanmean(G_feats_std[len(G)//2:], axis=0)
    
    # G_norm = []
    # for ii in np.arange(len(G)):
    #     if ii <len(G)//2:
    #         G_norm.append(G[ii]*mean_G_feats_std_mig[None,:])
    #     else:
    #         G_norm.append(G[ii]*mean_G_feats_std_pre[None,:])
    # G = G_norm
    
    # G = [load_pickle(ff)['corr_matrix'] for ff in all_network_files]
    # G_pval = [load_pickle(ff)['corr_matrix_pval'] for ff in all_network_files]
    param_names = load_pickle(all_network_files[0])['lasso_param_names']
    
    for g_ii in np.arange(len(G)):
        
        if g_ii < len(G)//2:
            fig, ax = plt.subplots(figsize=(15,15))
            plt.title('Coef Matrix, Normalised Feats, %s' %( all_network_fnames[g_ii]), fontsize=32)
            plt.imshow(G[g_ii]*mean_G_feats_std_mig[None,:], vmin=-1, vmax=1, cmap='coolwarm')
            plt.xticks(np.arange(len(G[g_ii])), param_names, rotation=45, horizontalalignment='right', fontsize=24)
            plt.yticks(np.arange(len(G[g_ii])), param_names, rotation=0, horizontalalignment='right', fontsize=24)
            # fig.savefig(os.path.join(analysis_model_save_folder, save_analysis_name+'.svg'), bbox_inches='tight')
            plt.show()
        else:
            fig, ax = plt.subplots(figsize=(15,15))
            plt.title('Coef Matrix, Normalised Feats, %s' %( all_network_fnames[g_ii]), fontsize=32)
            plt.imshow(G[g_ii]*mean_G_feats_std_pre[None,:], vmin=-1, vmax=1, cmap='coolwarm')
            plt.xticks(np.arange(len(G[g_ii])), param_names, rotation=45, horizontalalignment='right', fontsize=24)
            plt.yticks(np.arange(len(G[g_ii])), param_names, rotation=0, horizontalalignment='right', fontsize=24)
            # fig.savefig(os.path.join(analysis_model_save_folder, save_analysis_name+'.svg'), bbox_inches='tight')
            plt.show()
    

    # apply the pval to filter.
    # G = [G[ii]*(G_pval[ii]<0.01) for ii in np.arange(len(G))]
    G_flat = np.array([gg.ravel() for gg in G])
    
    dist_matrix = np.zeros((len(G), len(G)))
    
    for ii, jj in itertools.combinations(np.arange(len(G)), 2):
        G_ii = G[ii] #* G_feats_std[ii] #mean_G_feats_std[None,:]
        G_jj = G[jj] #* G_feats_std[jj] #mean_G_feats_std[None,:]
        # G_ii = G[ii] * mean_G_feats_std[None,:]
        # G_jj = G[jj] * mean_G_feats_std[None,:]
        
        # G_ii = np.clip(G[ii],-1,1) # clip to depress the total influence of curvature. 
        # G_jj = np.clip(G[jj],-1,1)
        
        # G_ii[preserved] = 0 
        # G_jj[preserved] = 0
        # G_ii[np.abs(G_ii) < 0.2] = 0
        # G_jj[np.abs(G_jj) < 0.2] = 0
        
        diff = np.sum(np.abs(G_ii-G_jj))
        # diff = pearsonr(G_ii.ravel(), G_jj.ravel())[0]
        dist_matrix[ii, jj] = diff
        dist_matrix[jj, ii] = diff
        
# =============================================================================
#   Do a hierarchical clustering.  
# =============================================================================
    import seaborn as sns 
    import pandas as pd 
    
    # # col_names = load_pickle(all_network_files[0])['lasso_param_names']
    # col_names = all_network_fnames
    
    # df = pd.DataFrame(dist_matrix, 
    #                   index=None, 
    #                   columns=all_network_fnames)
        
        
    # fig, ax = plt.subplots(figsize=(15,15))
    # g = sns.clustermap(dist_matrix, 
    #                    method='average',
    #                    cmap='vlag',
    #                    row_colors=row_colors)
    # g.ax_row_dendrogram.remove()    
    # set up the colors. 
    dist_colors = sns.color_palette('magma_r', 8)[::2]
    AP_colors = sns.color_palette('coolwarm',3)
    AP_colors = [AP_colors[1],AP_colors[0],AP_colors[2]]
    
# =============================================================================
#     Parse the conditions and get the colors. 
# =============================================================================
    quad_colors_dist = np.hstack([[ii,ii] for ii in np.arange(4)])
    ant_posterior_dist = np.hstack([[2,1] for ii in np.arange(4)])
    
    quad_colors_dist = np.hstack([quad_colors_dist,quad_colors_dist])
    ant_posterior_dist = np.hstack([ant_posterior_dist, ant_posterior_dist])

        
    row_colors_lut = dict(zip(set(quad_colors_dist), dist_colors))
    row_colors = pd.DataFrame(quad_colors_dist,index=all_network_fnames)[0].map(row_colors_lut)        
    # row_colors_lut = {ii:dist_colors[quad_colors_dist[ii]] for ii in np.arange(len(G_quad))}
    # row_colors = pd.Series(np.arange(len(G_quad)), index=df.columns).map(row_colors_lut)
    row_colors_lut_2 = dict(zip([0,1,2], AP_colors))
    row_colors_2 = pd.DataFrame(ant_posterior_dist,index=all_network_fnames)[0].map(row_colors_lut_2)  


    fig, ax = plt.subplots(figsize=(15,15))
    g = sns.clustermap(pd.DataFrame(dist_matrix,#/np.std(dist_matrix), 
                                    index=all_network_fnames,
                                    columns=all_network_fnames), 
                        # method='average', # We use the ward? 
                        # method='ward', 
                        method='complete',
                       # method='single',
                       metric='euclidean',
                       cmap='vlag',
                       col_colors=[row_colors, row_colors_2])
    g.ax_row_dendrogram.remove() 
    
    if labellingtype=='dynamic':
        plt.savefig(os.path.join(analysis_model_save_folder, 'dynamic_dendrogram.svg'), bbox_inches='tight')
    if labellingtype=='static':
        plt.savefig(os.path.join(analysis_model_save_folder, 'static_dendrogram.svg'), bbox_inches='tight')
    plt.show()
    
    
    # ant_posterior_dist = np.zeros(len(G_quad), dtype=np.int)
    
    # anteriors = [[8,1,2], 
    #              [16,9,10],
    #              [24,17,18], 
    #              [32,25,26]]
    # anteriors = np.array(anteriors).ravel()
    # for aa in anteriors:
    #     ant_posterior_dist[G_quad==aa] = 2
        
    # posteriors = [[4,5,6], 
    #               [12,13,14],
    #               [20,21,22], 
    #               [28,29,30]]
    # posteriors = np.array(posteriors).ravel()
    # for pp in posteriors:
    #     ant_posterior_dist[G_quad==pp] = 1    
    
    
    
    # from sklearn.decomposition import PCA, FastICA
    # from sklearn.manifold import TSNE, SpectralEmbedding, MDS, Isomap
    # from sklearn.metrics.pairwise import pairwise_distances
    # import kmapper as km 
    
    
    # # pca_model = PCA(n_components=G_flat.shape[0])
    # # pca_model = FastICA(n_components=3)
    # # pca_model = MDS(n_components=3)
    # pca_model = Isomap(n_components=3, n_neighbors=2, metric='manhattan')
    # # pca_model.fit(G_flat)
    
    # # # plt.figure()
    # # # plt.plot(pca_model.explained_variance_ratio_)
    # # # plt.show(0)
    
    # # Y_pca = pca_model.transform(G_flat)
    # # pca_model = TSNE(n_components=3, perplexity=15,  learning_rate=200)
    # # pca_model = SpectralEmbedding(n_components=2, affinity='rbf')
    # # pca_model = SpectralEmbedding(n_components=3)
    # # pca_model = SpectralEmbedding(n_components=3,  affinity='rbf')
    # # pca_model = SpectralEmbedding(n_components=3, affinity='precomputed')
    # # pca_model = SpectralEmbedding(n_components=2)
    # # Y_pca = pca_model.fit_transform(pairwise_distances(G_flat, metric='l1'))
    # # Y_pca = pca_model.fit_transform(pairwise_distances(G_flat))
    # # Y_pca = pca_model.fit_transform(G_flat)
    # Y_pca = pca_model.fit_transform(dist_matrix)
    
    # import seaborn as sns 
    # plot_colors = sns.color_palette('Greens', 32)
    
    # fig, ax = plt.subplots(figsize=(10,10))
    
    # for ii in np.arange(0,len(G_quad)):
    #     ax.plot(Y_pca[ii,0], 
    #             Y_pca[ii,1],'o',
    #             color = dist_colors[quad_colors_dist[ii]],
    #             mec = AP_colors[ant_posterior_dist[ii]],
    #             mew=4,
    #             # label=all_network_fnames[int(all_network_fnames[ii].split('_')[-1])-1], 
    #             ms=15)
    #             # color=plot_colors[int(all_network_fnames[ii].split('_')[-1])-1])
    #     # plt.text(Y_pca[ii,0], 
    #     #           Y_pca[ii,1],
    #     #           str('Mig_')+str(int(all_network_fnames[ii].split('_')[-1])),
    #     #           ha='center',
    #     #           va='center')


    # from mpl_toolkits.mplot3d import Axes3D
    
    # fig = plt.figure(figsize=(15,15))
    # ax = fig.add_subplot(111, projection='3d')
    # for ii in np.arange(0,len(G_quad)):
    #     ax.scatter(Y_pca[ii,0], 
    #                 Y_pca[ii,1],
    #                 Y_pca[ii,2],
    #                 color = dist_colors[quad_colors_dist[ii]],
    #                 edgecolors = AP_colors[ant_posterior_dist[ii]],
    #                 linewidths =5,
    #                 # label=all_network_fnames[int(all_network_fnames[ii].split('_')[-1])-1], 
    #                 s=500)
    # plt.show()

    # # for ii in np.arange(0,len(all_network_fnames)):
    # #     ax.plot(Y_pca[ii,0], 
    # #             Y_pca[ii,1],'o',
    # #             label=all_network_fnames[ii], 
    # #             ms=10, 
    # #             color=plot_colors[ii])
    # #     plt.text(Y_pca[ii,0], 
    # #               Y_pca[ii,1],
    # #               str('Mig_')+str(ii),
    # #               ha='center',
    # #               va='center')
        
    # # plt.savefig('test.svg',bbox_inches='tight')
        
    # # for ii in np.arange(32,len(Y_pca)):
    # #     ax.plot(Y_pca[ii,0], 
    # #             Y_pca[ii,1],'o',label=all_network_fnames[ii], ms=20, color=plot_colors[ii-32])
    # #     plt.text(Y_pca[ii,0], 
    # #              Y_pca[ii,1],
    # #              str('Pre_')+str(ii-32+1))
    # # # plt.legend()       
    # plt.show()
    
    
    
# =============================================================================
#     Visualize the polar grid create a fake one. ?
# =============================================================================
    
    
    
    # dist_matrix = np.zeros((len(all_network_fnames), len(all_network_fnames)))
    
    
    

    # # # node one graph just to get the nodal names. 
    # # G1 = load_pickle(all_network_files[0])['lasso_graph']
    # # nodes = list(G1.nodes())

    # # G2 = load_pickle(all_network_files[1])['lasso_graph']
    
    # # G1_matrix = nx.to_numpy_matrix(G1, nodelist=nodes)
    
    # # from grakel.utils import graph_from_networkx
    
    
    # # # G_nx = [G1, G2]
    # # G_nx = [load_pickle(all_network_files[ii])['lasso_graph'] for ii in np.arange(len(all_network_files))]

    # # # Transforms list of NetworkX graphs into a list of GraKeL graphs
    # # G = graph_from_networkx(G_nx)
    
    # # # from grakel.kernels import PropagationAttr
    # # from grakel import RandomWalk
    # # from grakel import MultiscaleLaplacian, SvmTheta
    # # # Uses the graphhopper kernel to generate the kernel matrices
    # # # gk = PropagationAttr(normalize=True)
    # # # gk = RandomWalk(normalize=False, kernel_type='exponential')
    # # gk = RandomWalk(normalize=False)
    # # # gk = SvmTheta(normalize=True)
    # # K_G = gk.fit_transform(G)
    
    # # dist_matrix = K_G.max() - K_G
    # null_connect_cost = 0 # this was the threshold set for the coefficients. 
    
    # for i,j in itertools.combinations(np.arange(len(all_network_fnames)), 2):
    #     print(i,j)
        
    #     G1_matrix = load_pickle(all_network_files[i])['corr_matrix']
    #     G2_matrix = load_pickle(all_network_files[j])['corr_matrix']
        
    #     feature_names = load_pickle(all_network_files[j])['lasso_param_names']#: col_name_select
        
    #     # # # G1_matrix = nx.to_numpy_matrix(G1, nodelist=nodes)
    #     # # # G2_matrix = nx.to_numpy_matrix(G2, nodelist=nodes)
    #     # G1_matrix = np.sign(G1_matrix) * np.logical_and(np.abs(G1_matrix) > 0.1, np.abs(G1_matrix) < 0.8)*1
    #     # G2_matrix = np.sign(G2_matrix) * np.logical_and(np.abs(G2_matrix) > 0.1, np.abs(G2_matrix) < 0.8)*1
        
    #     # # the distance is then a matrix comparison 
    #     # G1_matrix[G1_matrix==0] = np.nan
    #     # G2_matrix[G2_matrix==0] = np.nan
        
    #     diff_matrix = np.abs(G1_matrix-G2_matrix)
        
    #     # # make the corrections.
    #     # nan_1 = np.logical_and(np.isnan(G1_matrix), 
    #     #                         ~np.isnan(G2_matrix))
    #     # nan_2 = np.logical_and(~np.isnan(G1_matrix), 
    #     #                         np.isnan(G2_matrix))
    #     # diff_matrix[nan_1] = null_connect_cost
    #     # diff_matrix[nan_2] = null_connect_cost
        
        
    #     dist_matrix[i,j] = np.nansum(diff_matrix)
    #     dist_matrix[j,i] = np.nansum(diff_matrix)
        
 
    # dist_matrix[np.isnan(dist_matrix)] = np.nanmax(dist_matrix)    
    
    
    # plt.figure(figsize=(15,15))
    # plt.imshow(dist_matrix, cmap='coolwarm')
    # plt.xticks(np.arange(len(all_network_fnames)), all_network_fnames, rotation=90, fontsize=18)
    # plt.yticks(np.arange(len(all_network_fnames)), all_network_fnames, rotation=0, fontsize=18)
    # plt.show()
    
    # # hierarchical clustering
    
    # from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    # from scipy.spatial.distance import squareform
    
    # plt.figure(figsize=(12,5))
    # # dissimilarity = 1 - abs(correlations)
    # # Z = linkage(dist_matrix, 'complete')
    # Z = linkage(squareform(dist_matrix), 'ward')
    # # Z = linkage(dist_matrix, 'ward')
    # # Z = linkage(dist_matrix, 'average')
    # # D = dendrogram(Z, labels=all_network_fnames, orientation='top', leaf_rotation=90);
    # D = dendrogram(Z,  orientation='top', leaf_rotation=90);
    # # D['leaves'] = all_network_fnames
    # dist_positions = np.hstack(D['dcoord'])
    # xplot_positions = np.hstack(D['icoord'])[dist_positions==0]
    # plt.xticks(xplot_positions[np.argsort(xplot_positions)],
    #             all_network_fnames[D['leaves']], rotation=45,   ha ='right')
    # plt.show()
    
    # plt.figure(figsize=(15,15))
    # plt.imshow(dist_matrix[D['leaves']][:,D['leaves']], cmap='coolwarm')
    # plt.xticks(np.arange(len(all_network_fnames)), all_network_fnames[D['leaves']], rotation=90, fontsize=18)
    # plt.yticks(np.arange(len(all_network_fnames)), all_network_fnames[D['leaves']], rotation=0, fontsize=18)
    # plt.show()
    
    
        