#! /usr/bin/python3
print('this is cell-analyzer v0.1.0' + '\n')
print('preparing image segmentation run...' + '\n')

import os
import glob
import numpy as np
import pandas as pd
import skimage as sk
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from datetime import datetime as dt
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib import cm
from skimage import exposure, feature, filters, measure, morphology, segmentation
from scipy import ndimage as ndi
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap

import warnings
warnings.filterwarnings("ignore")


def process_image(img, norm_window, min_hole_size, min_cell_size, extrema_blur, peak_sep, name='temp.TIF', save_path = '.'):
    img_dims = np.shape(img)
    print('image dimensions: ', img_dims)

    if len(img_dims) < 3:
        n_chan = 1
        content = img
        v_min, v_max = np.percentile(content, (1,99))
        content_scaled = exposure.rescale_intensity(content, in_range=(v_min, v_max))

    else:
        # handle if first channel is blank
        if np.mean(img[:,:,0]) < 1:
            img = img[:,:,1:]
            img_dims = np.shape(img)
        # handle other blank channels
        n_chan = img_dims[2]
        base = img[:,:,0]
        # restack image, excluding blank channels
        for channel in range(1, n_chan):
            if np.sum(img[:,:,channel]) > (img_dims[0] * img_dims[1] * 0.2):
                base = np.stack((base, img[:,:,channel]), axis=2)

        img = base
        img_dims = np.shape(img)
        n_chan = img_dims[2]

        ### custom colormaps
        N = 256
        blank = np.zeros(N)
        gray = np.linspace(0, 1, N)
        # blue
        blues = np.ones((N,4))
        blues[:,0] = blank
        blues[:,1] = blank
        blues[:,2] = gray
        blue_cmap = ListedColormap(blues)
        # green
        greens = np.ones((N,4))
        greens[:,0] = blank
        greens[:,1] = gray
        greens[:,2] = blank
        green_cmap = ListedColormap(greens)
        # red
        reds = np.ones((N,4))
        reds[:,0] = gray
        reds[:,1] = blank
        reds[:,2] = blank
        red_cmap = ListedColormap(reds)

        # separate and scale channels for vis
        content = np.sum(img, axis=2)
        v_min, v_max = np.percentile(content, (1,99))
        content_scaled = exposure.rescale_intensity(content, in_range=(v_min, v_max))

        if n_chan >= 1:
            dapi = img[:,:,0]
            v_min, v_max = np.percentile(dapi, (1,99))
            dapi_scaled = exposure.rescale_intensity(dapi, in_range=(v_min, v_max))

        if n_chan >= 2:
            gfp = img[:,:,1]
            v_min, v_max = np.percentile(gfp, (1,99))
            gfp_scaled = exposure.rescale_intensity(gfp, in_range=(v_min, v_max))

        if n_chan >= 3:
            txred = img[:,:,2]
            v_min, v_max = np.percentile(txred, (1,99))
            txred_scaled = exposure.rescale_intensity(txred, in_range=(v_min, v_max))

        if n_chan == 4:
            cy5 = img[:,:,3]
            v_min, v_max = np.percentile(cy5, (1,99))
            cy5_scaled = exposure.rescale_intensity(cy5, in_range=(v_min, v_max))

        if n_chan > 4:
            print('handling of more than 4 image channels not supported')

### handle single high-res or stitched low-res images (large dimensions)
    if np.logical_and(np.shape(img)[0] < 2500, np.shape(img)[1] < 2500):
        # correct image and create content mask
        bg = filters.threshold_local(content, norm_window)
        norm = content / bg
        blur = filters.gaussian(norm, sigma=2)
        # blur = filters.gaussian(content, sigma=2)
        otsu = filters.threshold_otsu(blur)
        mask = blur > otsu
        mask_filled = morphology.remove_small_holes(mask, min_hole_size)
        selem = morphology.disk(3)
        mask_opened = morphology.binary_opening(mask_filled, selem)
        mask_filtered = morphology.remove_small_objects(mask_opened, min_cell_size)
        heavy_blur = filters.gaussian(content, extrema_blur)
        blur_masked = heavy_blur * mask_filtered
    else:
        blur = filters.gaussian(content, sigma=2)
        otsu = filters.threshold_otsu(blur)
        mask = blur > otsu
        mask_filtered = mask
        blur_masked = mask * blur

    # find local maxima
    coords = feature.peak_local_max(blur_masked, min_distance=peak_sep)
    coords_T = []
    coords_T += coords_T + [[point[1], point[0]] for point in coords]
    coords_T = np.array(coords_T)
    markers = np.zeros(np.shape(content))
    for i, point in enumerate(coords):
        markers[point[0], point[1]] = i

    # generate labeled cells
    rough_labels = measure.label(mask_filtered)
    distance = ndi.distance_transform_edt(mask_filtered)
    ws = segmentation.watershed(-distance, markers, connectivity=2, watershed_line=True)
    labeled_cells = ws * mask_filtered


    # measure and store image channel props from content mask
    print('# of content channels (n_chan): ', n_chan)
    cell_props = {}

    if n_chan > 1:
        # store and gate dapi
        dapi_props = measure.regionprops(labeled_cells, dapi)
        cell_props['dapi_props'] = dapi_props
        dapi_blur = filters.gaussian(dapi)
        dapi_otsu = filters.threshold_otsu(dapi_blur)
        dapi_mask = dapi_blur > dapi_otsu
        gated_dapi = dapi_mask * labeled_cells

    if n_chan >= 2:
        # store and gate gfp
        gfp_props = measure.regionprops(labeled_cells, gfp)
        cell_props['gfp_props'] = gfp_props
        gfp_blur = filters.gaussian(gfp)
        gfp_otsu = filters.threshold_otsu(gfp_blur)
        gfp_mask = gfp_blur > gfp_otsu
        gated_gfp = gfp_mask * labeled_cells

    if n_chan >= 3:
        # store and gate txred
        txred_props = measure.regionprops(labeled_cells, txred)
        cell_props['txred_props'] = txred_props
        txred_blur = filters.gaussian(txred)
        txred_otsu = filters.threshold_otsu(txred_blur)
        txred_mask = txred_blur > txred_otsu
        gated_txred = txred_mask * labeled_cells

    if n_chan == 4:
        # store and gate cy5
        cy5_props = measure.regionprops(labeled_cells, cy5)
        cell_props['cy5_props'] = cy5_props
        cy5_blur = filters.gaussian(cy5)
        cy5_otsu = filters.threshold_otsu(cy5_blur)
        cy5_mask = cy5_blur > cy5_otsu
        gated_cy5 = cy5_mask * labeled_cells

    content_props = measure.regionprops(labeled_cells, content)
    cell_props['image_content'] = content_props


    # define custom label mask colormap
    plasma = cm.get_cmap('plasma', 256)
    newcolors = plasma(np.linspace(0, 1, 256))
    newcolors[0, :] = [0, 0, 0, 1]
    custom_cmap = ListedColormap(newcolors)


    # plot & return results
    if n_chan == 1:
        # plt.imshow(content_scaled, cmap='gray')
        # plt.title('original content')
        # plt.show()
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9,5))
        ax[0].imshow(content_scaled, cmap='viridis')
        ax[0].set_title('scaled image')
        ax[1].imshow(mask_filtered, cmap='gray')
        ax[1].set_title('mask')
        ax[2].imshow(labeled_cells, cmap=custom_cmap)
        ax[2].plot(coords[:,1], coords[:,0], c='yellow', marker = '*', linestyle='', markersize=2)
        ax[2].set_title('labels')
        plt.tight_layout()
        # plt.show()
        plt.savefig(save_path + '/' + name[:-4] + '_cell_labels.png')
        plt.close()

    elif n_chan == 2:
        # plt.imshow(content_scaled, cmap='gray')
        # plt.title('original content')
        # plt.show()
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,7))
        ax[0, 0].imshow(dapi_scaled, cmap=blue_cmap)
        ax[0, 0].set_title('scaled dapi')
        ax[0, 1].imshow(mask_filtered, cmap='gray')
        ax[0, 1].set_title('image mask')
        ax[0, 2].imshow(labeled_cells, cmap=custom_cmap)
        ax[0, 2].plot(coords[:,1], coords[:,0], c='yellow', marker = '*', linestyle='', markersize=2)
        ax[0, 2].set_title('labels')
        ax[1, 0].imshow(gfp_scaled, cmap=green_cmap)
        ax[1, 0].set_title('scaled gfp')
        ax[1, 1].imshow(gfp_mask, cmap='gray')
        ax[1, 1].set_title('gfp mask')
        ax[1, 2].imshow(gated_gfp, cmap=custom_cmap)
        ax[1, 2].set_title('gated gfp')
        plt.tight_layout()
        # plt.show()
        plt.savefig(save_path + '/' + name[:-4] + '_cell_labels.png')
        plt.close()

    elif n_chan == 3:
        # plt.imshow(content_scaled, cmap='gray')
        # plt.title('original content')
        # plt.show()
        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15,9))
        ax[0, 0].imshow(dapi_scaled, cmap=blue_cmap)
        ax[0, 0].set_title('scaled dapi')
        ax[0, 1].imshow(mask_filtered, cmap='gray')
        ax[0, 1].set_title('image mask')
        ax[0, 2].imshow(labeled_cells, cmap=custom_cmap)
        ax[0, 2].plot(coords[:,1], coords[:,0], c='yellow', marker = '*', linestyle='', markersize=2)
        ax[0, 2].set_title('labels')
        ax[1, 0].imshow(gfp_scaled, cmap=green_cmap)
        ax[1, 0].set_title('scaled gfp')
        ax[1, 1].imshow(gfp_mask, cmap='gray')
        ax[1, 1].set_title('gfp mask')
        ax[1, 2].imshow(gated_gfp, cmap=custom_cmap)
        ax[1, 2].set_title('gated gfp')
        ax[2, 0].imshow(txred_scaled, cmap=red_cmap)
        ax[2, 0].set_title('scaled txred')
        ax[2, 1].imshow(txred_mask, cmap='gray')
        ax[2, 1].set_title('txred mask')
        ax[2, 2].imshow(gated_txred, cmap=custom_cmap)
        ax[2, 2].set_title('gated txred')
        plt.tight_layout()
        # plt.show()
        plt.savefig(save_path + '/' + name[:-4] + '_cell_labels.png')
        plt.close()

    else:
        # plt.imshow(content_scaled, cmap='gray')
        # plt.title('original content')
        # plt.show()
        fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(16,10))
        ax[0, 0].imshow(dapi_scaled, cmap=blue_cmap)
        ax[0, 0].set_title('scaled dapi')
        ax[0, 1].imshow(mask_filtered, cmap='gray')
        ax[0, 1].set_title('image mask')
        ax[0, 2].imshow(labeled_cells, cmap=custom_cmap)
        ax[0, 2].plot(coords[:,1], coords[:,0], c='yellow', marker = '*', linestyle='', markersize=2)
        ax[0, 2].set_title('labels')
        ax[1, 0].imshow(gfp_scaled, cmap=green_cmap)
        ax[1, 0].set_title('scaled gfp')
        ax[1, 1].imshow(gfp_mask, cmap='gray')
        ax[1, 1].set_title('gfp mask')
        ax[1, 2].imshow(gated_gfp, cmap=custom_cmap)
        ax[1, 2].set_title('gated gfp')
        ax[2, 0].imshow(txred_scaled, cmap=red_cmap)
        ax[2, 0].set_title('scaled txred')
        ax[2, 1].imshow(txred_mask, cmap='gray')
        ax[2, 1].set_title('txred mask')
        ax[2, 2].imshow(gated_txred, cmap=custom_cmap)
        ax[2, 2].set_title('gated txred')
        ax[3, 0].imshow(cy5_scaled, cmap='gray')
        ax[3, 0].set_title('scaled cy5')
        ax[3, 1].imshow(cy5_mask, cmap='gray')
        ax[3, 1].set_title('cy5 mask')
        ax[3, 2].imshow(gated_cy5, cmap=custom_cmap)
        ax[3, 2].set_title('gated cy5')
        plt.tight_layout()
        # plt.show()
        plt.savefig(save_path + '/' + name[:-4] + '_cell_labels.png')
        plt.close()

    return img, labeled_cells, cell_props, n_chan



def read_and_process_directory(base_directory, norm_window, min_hole_size, min_cell_size, extrema_blur, peak_sep, formatted_titles, channel_list):
    # process all .tif files in a passed directory and returns results dataframe (.csv)

    # set up paths
    time_stamp = dt.now().strftime('%Y_%m_%d_%H_%M_%S')
    save_path = '_extracted_data_%s' % time_stamp
    save_path = os.path.join(base_directory, save_path)
    print('base: ' + base_directory)
    print('save: ' + save_path)
    print('channel_list: ', channel_list)
    os.mkdir(save_path)

    # get paths for images in base directory
    image_list = glob.glob(os.path.join(base_directory, '*.TIF')) # '.TIF' or '.tif'
    image_list = image_list + glob.glob(os.path.join(base_directory, '*.tif'))

    # initialize results dataframe
    results_df = pd.DataFrame()

    # iteratively read in images by filenames
    for i, img_path in enumerate(image_list):
        img = plt.imread(img_path)
        name = os.path.basename(img_path)

        print('\n')
        print(name)
        print('img ' + str(i + 1) + ' of ' + str(len(image_list)))

        img, labeled_cells, cell_props, n_chan = process_image(img, norm_window, min_hole_size, min_cell_size, extrema_blur, peak_sep, name, save_path)

        # save all cell quant in results dataframe
        img_df = pd.DataFrame()

        for count, key in enumerate(cell_props.keys()):
            channel_data = cell_props[key]

            if np.logical_and(len(np.shape(img)) > 2, str(count) in channel_list):
                if count < n_chan:
                    bg = filters.threshold_local(img[:,:,count], norm_window)

            for c, cell in enumerate(channel_data):

                if c % 10 == 0:
                    pct_done = ((c + (len(channel_data) * count)) / len(channel_data))
                    pct_done = np.round((pct_done / len(cell_props.keys())) * 100, decimals=3)
                    print('img ' + str(pct_done) + '% complete ' + '(channel ' + str(count + 1) + ' of ' + str(len(cell_props.keys())) + ', cell ' + str(c + 1) + ' of ' + str(len(channel_data))+ ')' )

                if count == 0:
                    if formatted_titles:
                        img_df = img_df.append({'_image_name': name, '_img_cell_#': c, '_clone': name[14:17],
                            '_moi': float(name[18:20]), '_days_post_inf': float(name[22:24]),
                            'area_ch' + str(count): channel_data[c]['area'],
                            'log_area_ch' + str(count): np.log10(channel_data[c]['area']),
                            'max_ch' + str(count): channel_data[c]['max_intensity'],
                            'min_ch' + str(count): channel_data[c]['min_intensity'],
                            'mean_ch' + str(count): channel_data[c]['mean_intensity'],
                            'extent_ch' + str(count): channel_data[c]['extent'],
                            'eccentricity_ch' + str(count): channel_data[c]['eccentricity'],
                            'perimeter_ch' + str(count): channel_data[c]['perimeter'],
                            'major_axis_ch' + str(count): channel_data[c]['major_axis_length'],
                            'minor_axis_ch' + str(count): channel_data[c]['minor_axis_length']}, ignore_index=True)

                    else:
                        img_df = img_df.append({'_image_name': name, '_img_cell_#': c,
                            'area_ch' + str(count): channel_data[c]['area'],
                            'log_area_ch' + str(count): np.log10(channel_data[c]['area']),
                            'max_ch' + str(count): channel_data[c]['max_intensity'],
                            'min_ch' + str(count): channel_data[c]['min_intensity'],
                            'mean_ch' + str(count): channel_data[c]['mean_intensity'],
                            'extent_ch' + str(count): channel_data[c]['extent'],
                            'eccentricity_ch' + str(count): channel_data[c]['eccentricity'],
                            'perimeter_ch' + str(count): channel_data[c]['perimeter'],
                            'major_axis_ch' + str(count): channel_data[c]['major_axis_length'],
                            'minor_axis_ch' + str(count): channel_data[c]['minor_axis_length']}, ignore_index=True)

                    img_df['log_area_ch' + str(count)] = img_df['log_area_ch' + str(count)].replace(0, 0.01)

                else:
                    img_df.loc[img_df.index[c], 'area_ch' + str(count)] = channel_data[c]['area']
                    img_df.loc[img_df.index[c], 'log_area_ch' + str(count)] = np.log10(channel_data[c]['area'])
                    img_df.loc[img_df.index[c], 'max_ch' + str(count)] = channel_data[c]['max_intensity']
                    img_df.loc[img_df.index[c], 'min_ch' + str(count)] = channel_data[c]['min_intensity']
                    img_df.loc[img_df.index[c], 'mean_ch' + str(count)] = channel_data[c]['mean_intensity']
                    img_df.loc[img_df.index[c], 'extent_ch' + str(count)] = channel_data[c]['extent']
                    img_df.loc[img_df.index[c], 'eccentricity_ch' + str(count)] = channel_data[c]['eccentricity']
                    img_df.loc[img_df.index[c], 'perimeter_ch' + str(count)] = channel_data[c]['perimeter']
                    img_df.loc[img_df.index[c], 'major_axis_ch' + str(count)] = channel_data[c]['major_axis_length']
                    img_df.loc[img_df.index[c], 'minor_axis_ch' + str(count)] = channel_data[c]['minor_axis_length']
                    img_df['log_area_ch' + str(count)] = img_df['log_area_ch' + str(count)].replace(0, 0.01)

                ### new -- for channel-specific detailed regionprop data to add to img_df
                if str(count) in channel_list:
                    if np.logical_and(count < n_chan, len(np.shape(img)) > 2):
                        cleaned_channel = img[:,:,count] / bg
                        cleaned_channel = filters.gaussian(cleaned_channel, sigma=2)
                        ch_otsu = filters.threshold_otsu(cleaned_channel)
                        ch_feat_mask = morphology.binary_erosion(cleaned_channel > ch_otsu)
                        ch_feat_mask = morphology.remove_small_objects(ch_feat_mask, 2)

                        cell_ch_labels = measure.label((labeled_cells == c) * ch_feat_mask)
                        cell_ch_props = measure.regionprops(cell_ch_labels, img[:,:,count])
                        ch_feat_areas = np.array([r.area for r in cell_ch_props])
                        ch_feat_means = np.array([r.mean_intensity for r in cell_ch_props])
                        ch_feat_maxes = np.array([r.max_intensity for r in cell_ch_props])
                        ch_feat_mins = np.array([r.min_intensity for r in cell_ch_props])
                        img_df.loc[img_df.index[c], 'num_features_ch' + str(count)] = len(cell_ch_props)

                        if len(cell_ch_props) > 0:
                            img_df.loc[img_df.index[c], 'log_num_features_ch' + str(count)] = np.log10(len(cell_ch_props))
                        else:
                            img_df.loc[img_df.index[c], 'log_num_features_ch' + str(count)] = 0.01

                        img_df.loc[img_df.index[c], 'avg_feature_area_ch' +  str(count)] = np.mean(ch_feat_areas)
                        img_df.loc[img_df.index[c], 'log_avg_feature_area_ch' + str(count)] = np.log10(np.mean(ch_feat_areas))
                        img_df.loc[img_df.index[c], 'median_feature_area_ch' + str(count)] = np.median(ch_feat_areas)
                        img_df.loc[img_df.index[c], 'avg_feature_int_ch' +  str(count)] = np.mean(ch_feat_means)
                        img_df.loc[img_df.index[c], 'avg_feature_max_ch' +  str(count)] = np.mean(ch_feat_maxes)
                        img_df.loc[img_df.index[c], 'avg_feature_min_ch' +  str(count)] = np.mean(ch_feat_mins)
                        img_df.loc[img_df.index[c], 'feature_coverage_%_ch' + str(count)] = np.sum(ch_feat_mask * (labeled_cells == c)) / np.sum(labeled_cells == c)
                        img_df.fillna(0, inplace=True)

            try:
                c
            except:
                print('no cells detected')
            else:
                if count == len(cell_props.keys()) - 1:
                    pct_done = 100.00
                    print('img ' + str(pct_done) + '% complete ' + '(channel ' + str(count + 1) + ' of ' + str(len(cell_props.keys())) + ', cell ' + str(c + 1) + ' of ' + str(len(channel_data))+ ')' )

                if np.logical_and(count < n_chan, len(np.shape(img)) > 2):
                    if str(count) in channel_list:
                        img_df['log_avg_feature_area_ch' + str(count)] = img_df['log_avg_feature_area_ch' + str(count)].replace(0, 0.01)
                        img_df['log_num_features_ch' + str(count)] = img_df['log_num_features_ch' + str(count)].replace(0, 0.01)


        results_df = pd.concat([results_df, img_df])
        results_df.fillna(0, inplace=True)

    results_df.to_csv(save_path + '/' + '_cell_datasheet.csv')
    return results_df, save_path, n_chan



def cluster_results(results_df, save_path, n_chan, num_clust, dim_method, formatted_titles, channel_list):
    # simple unsupervised hierarchical clsutering based on cell properties

    if formatted_titles:
        # encode clone number as variable
        clone_key = list(np.unique(results_df['_clone']))
        clone_temp = results_df['_clone'].values
        clone_IDs = [ clone_key.index(item) for item in clone_temp ]
        # dendrogram
        data = results_df.iloc[:, 6:].values
    else:
        # dendrogram
        data = results_df.iloc[:, 3:].values

    plt.figure(figsize=(10, 7))
    plt.title("Cell Property Dendogram")
    shc.set_link_color_palette(['#dbd057', '#75db57', '#57dbaa', '#579bdb', '#8557db'])
    Z = shc.linkage(data, method='ward')
    dend = shc.dendrogram(Z, color_threshold=0.25*np.max(Z), above_threshold_color='#db5f57')
    plt.savefig(save_path + '/' + '_cell_dendrogram.png')
    plt.close()

    # generate cluster IDs
    cluster = AgglomerativeClustering(n_clusters=num_clust, affinity='euclidean', linkage='ward')
    ID = cluster.fit_predict(data)

    # reduction -- UMAP
    reducer = umap.UMAP()
    scaled_data = StandardScaler().fit_transform(data)
    embedding = reducer.fit_transform(scaled_data)
    results_df['umap-2d-one'] = embedding[:,0]
    results_df['umap-2d-two'] = embedding[:,1]

    if dim_method == 'umap_c':
        clusterable_embedding = umap.UMAP(n_neighbors=20, min_dist=0.0, n_components=2, random_state=42).fit_transform(scaled_data)
        embed_labels = cluster.fit_predict(clusterable_embedding)
        results_df['umap-2d-one'] = clusterable_embedding[:,0]
        results_df['umap-2d-two'] = clusterable_embedding[:,1]

    # reduction -- TSNE
    data_embedded = TSNE(n_components=2).fit_transform(data)
    results_df['tsne-2d-one'] = data_embedded[:,0]
    results_df['tsne-2d-two'] = data_embedded[:,1]

    # method-specific cluster labels
    if dim_method == 'tsne':
        x_dim = 'tsne-2d-one'
        y_dim = 'tsne-2d-two'
        results_df['cluster_ID'] = ID[:]
    if dim_method in ['umap', 'umap_c']:
        x_dim = 'umap-2d-one'
        y_dim = 'umap-2d-two'
        results_df['cluster_ID'] = embed_labels[:]

    # add columns for tsne labeling
    if formatted_titles:
        results_df['clone_ID'] = clone_IDs
    else:
        results_df['clone_ID'] = 'A00'


    # clusters
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=x_dim, y=y_dim,
        hue="cluster_ID",
        data=results_df,
        palette=sns.color_palette("hls", num_clust),
        legend="full",
        alpha=0.75
    )
    plt.title('cell clustering')
    plt.savefig(save_path + '/_' + dim_method + '_clusters.png')
    plt.close()

    if formatted_titles:
        # clone IDs
        plt.figure(figsize=(10, 7))
        g = sns.scatterplot(
            x=x_dim, y=y_dim,
            hue="clone_ID",
            data=results_df,
            palette=sns.color_palette("husl", len(clone_key)),
            legend="full",
            alpha=0.75
        )
        plt.title('cell clones')
        for t, l in zip(g.legend().texts, clone_key): t.set_text(l)
        plt.savefig(save_path + '/_' + dim_method + '_clones.png')
        plt.close()

        # days post-infection
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            x=x_dim, y=y_dim,
            hue="_days_post_inf",
            data=results_df,
            palette=sns.color_palette("magma", len(np.unique(results_df['_days_post_inf']))),
            legend="full",
            alpha=0.75
        )
        plt.title('days post infection')
        plt.savefig(save_path + '/_' + dim_method + '_dpi.png')
        plt.close()

        # MOI
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            x=x_dim, y=y_dim,
            hue="_moi",
            data=results_df,
            palette=sns.color_palette("viridis", len(np.unique(results_df['_moi']))),
            legend="full",
            alpha=0.75
        )
        plt.title('EBV moi')
        plt.savefig(save_path + '/_' + dim_method + '_moi.png')
        plt.close()


    for channel in range(n_chan):
        # cell areas
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            x=x_dim, y=y_dim,
            hue="area_ch" + str(channel),
            data=results_df,
            palette=sns.color_palette("mako", as_cmap=True),
            legend=None,
            alpha=0.75
        )
        norm = plt.Normalize(results_df['area_ch' + str(channel)].min(), results_df['area_ch' + str(channel)].max())
        sm = plt.cm.ScalarMappable(cmap="mako", norm=norm)
        sm.set_array([])
        plt.colorbar(sm)
        plt.title('cell area - ' + 'ch' + str(channel))
        plt.savefig(save_path + '/_' + dim_method + '_area_ch' + str(channel) + '.png')
        plt.close()


        # log cell areas
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            x=x_dim, y=y_dim,
            hue="log_area_ch" + str(channel),
            data=results_df,
            palette=sns.color_palette("mako", as_cmap=True),
            legend=None,
            alpha=0.75
        )
        norm = plt.Normalize(results_df['log_area_ch' + str(channel)].min(), results_df['log_area_ch' + str(channel)].max())
        sm = plt.cm.ScalarMappable(cmap="mako", norm=norm)
        sm.set_array([])
        plt.colorbar(sm)
        plt.title('log cell area - ' + 'ch' + str(channel))
        plt.savefig(save_path + '/_' + dim_method + '_log_area_ch' + str(channel) + '.png')
        plt.close()


        # mean intensities
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            x=x_dim, y=y_dim,
            hue="mean_ch" + str(channel),
            data=results_df,
            palette=sns.color_palette("summer", as_cmap=True),
            legend=None,
            alpha=0.75
        )
        norm = plt.Normalize(results_df['mean_ch' + str(channel)].min(), results_df['mean_ch' + str(channel)].max())
        sm = plt.cm.ScalarMappable(cmap="summer", norm=norm)
        sm.set_array([])
        plt.colorbar(sm)
        plt.title('mean cell intensity (a.u) - ' + 'ch' + str(channel))
        plt.savefig(save_path + '/_' + dim_method + '_mean_int_ch' + str(channel) + '.png')
        plt.close()

        # min intensities
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            x=x_dim, y=y_dim,
            hue="min_ch" + str(channel),
            data=results_df,
            palette=sns.color_palette("ocean_r", as_cmap=True),
            legend=None,
            alpha=0.75
        )
        norm = plt.Normalize(results_df['min_ch' + str(channel)].min(), results_df['min_ch' + str(channel)].max())
        sm = plt.cm.ScalarMappable(cmap="ocean_r", norm=norm)
        sm.set_array([])
        plt.colorbar(sm)
        plt.title('min cell intensity (a.u) - ' + 'ch' + str(channel))
        plt.savefig(save_path + '/_' + dim_method + '_min_int_ch' + str(channel) + '.png')
        plt.close()

        # tmax intensities
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            x=x_dim, y=y_dim,
            hue="max_ch" + str(channel),
            data=results_df,
            palette=sns.color_palette("spring_r", as_cmap=True),
            legend=None,
            alpha=0.75
        )
        norm = plt.Normalize(results_df['max_ch' + str(channel)].min(), results_df['max_ch' + str(channel)].max())
        sm = plt.cm.ScalarMappable(cmap="spring_r", norm=norm)
        sm.set_array([])
        plt.colorbar(sm)
        plt.title('max cell intensity (a.u) - ' + 'ch' + str(channel))
        plt.savefig(save_path + '/_' + dim_method + '_max_int_ch' + str(channel) + '.png')
        plt.close()

        # eccentricity
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            x=x_dim, y=y_dim,
            hue="eccentricity_ch" + str(channel),
            data=results_df,
            palette=sns.color_palette("RdYlGn_r", as_cmap=True),
            legend=None,
            alpha=0.75
        )
        norm = plt.Normalize(results_df['eccentricity_ch' + str(channel)].min(), results_df['eccentricity_ch' + str(channel)].max())
        sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=norm)
        sm.set_array([])
        plt.colorbar(sm)
        plt.title('cell eccentricity (a.u) - ' + 'ch' + str(channel))
        plt.savefig(save_path + '/_' + dim_method + '_eccentricity_ch' + str(channel) + '.png')
        plt.close()

        # extent
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            x=x_dim, y=y_dim,
            hue="extent_ch" + str(channel),
            data=results_df,
            palette=sns.color_palette("RdYlBu_r", as_cmap=True),
            legend=None,
            alpha=0.75
        )
        norm = plt.Normalize(results_df['extent_ch' + str(channel)].min(), results_df['extent_ch' + str(channel)].max())
        sm = plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=norm)
        sm.set_array([])
        plt.colorbar(sm)
        plt.title('cell extent (a.u) - ' + 'ch' + str(channel))
        plt.savefig(save_path + '/_' + dim_method + '_extent_ch' + str(channel) + '.png')
        plt.close()


        if np.logical_and(str(channel) in channel_list, n_chan > 1):
            # n_channel_features
            plt.figure(figsize=(10, 7))
            sns.scatterplot(
                x=x_dim, y=y_dim,
                hue="num_features_ch" + str(channel),
                data=results_df,
                palette=sns.color_palette("CMRmap", as_cmap=True),
                legend=None,
                alpha=0.75
            )
            norm = plt.Normalize(results_df['num_features_ch' + str(channel)].min(), results_df['num_features_ch' + str(channel)].max())
            sm = plt.cm.ScalarMappable(cmap="CMRmap", norm=norm)
            sm.set_array([])
            plt.colorbar(sm)
            plt.title('num features - ' + 'ch' + str(channel))
            plt.savefig(save_path + '/_' + dim_method + '_num_features_ch' + str(channel) + '.png')
            plt.close()

            # log_n_channel_features
            plt.figure(figsize=(10, 7))
            sns.scatterplot(
                x=x_dim, y=y_dim,
                hue="log_num_features_ch" + str(channel),
                data=results_df,
                palette=sns.color_palette("CMRmap", as_cmap=True),
                legend=None,
                alpha=0.75
            )
            norm = plt.Normalize(results_df['log_num_features_ch' + str(channel)].min(), results_df['log_num_features_ch' + str(channel)].max())
            sm = plt.cm.ScalarMappable(cmap="CMRmap", norm=norm)
            sm.set_array([])
            plt.colorbar(sm)
            plt.title('log num features - ' + 'ch' + str(channel))
            plt.savefig(save_path + '/_' + dim_method + '_log_num_features_ch' + str(channel) + '.png')
            plt.close()

            # avg_feature_area_ch
            plt.figure(figsize=(10, 7))
            sns.scatterplot(
                x=x_dim, y=y_dim,
                hue="avg_feature_area_ch" + str(channel),
                data=results_df,
                palette=sns.color_palette("mako", as_cmap=True),
                legend=None,
                alpha=0.75
            )
            norm = plt.Normalize(results_df['avg_feature_area_ch' + str(channel)].min(), results_df['avg_feature_area_ch' + str(channel)].max())
            sm = plt.cm.ScalarMappable(cmap="mako", norm=norm)
            sm.set_array([])
            plt.colorbar(sm)
            plt.title('avg feature area - ' + 'ch' + str(channel))
            plt.savefig(save_path + '/_' + dim_method + '_avg_feature_area_ch' + str(channel) + '.png')
            plt.close()

            # median_feature_area_ch
            plt.figure(figsize=(10, 7))
            sns.scatterplot(
                x=x_dim, y=y_dim,
                hue="median_feature_area_ch" + str(channel),
                data=results_df,
                palette=sns.color_palette("mako", as_cmap=True),
                legend=None,
                alpha=0.75
            )
            norm = plt.Normalize(results_df['median_feature_area_ch' + str(channel)].min(), results_df['median_feature_area_ch' + str(channel)].max())
            sm = plt.cm.ScalarMappable(cmap="mako", norm=norm)
            sm.set_array([])
            plt.colorbar(sm)
            plt.title('median feature area - ' + 'ch' + str(channel))
            plt.savefig(save_path + '/_' + dim_method + '_median_feature_area_ch' + str(channel) + '.png')
            plt.close()

            # log_avg_feature_area_ch
            plt.figure(figsize=(10, 7))
            sns.scatterplot(
                x=x_dim, y=y_dim,
                hue="log_avg_feature_area_ch" + str(channel),
                data=results_df,
                palette=sns.color_palette("mako", as_cmap=True),
                legend=None,
                alpha=0.75
            )
            norm = plt.Normalize(results_df['log_avg_feature_area_ch' + str(channel)].min(), results_df['avg_feature_area_ch' + str(channel)].max())
            sm = plt.cm.ScalarMappable(cmap="mako", norm=norm)
            sm.set_array([])
            plt.colorbar(sm)
            plt.title('log avg feature area - ' + 'ch' + str(channel))
            plt.savefig(save_path + '/_' + dim_method + '_log_avg_feature_area_ch' + str(channel) + '.png')
            plt.close()

            # avg_feature_int_ch
            plt.figure(figsize=(10, 7))
            sns.scatterplot(
                x=x_dim, y=y_dim,
                hue="avg_feature_int_ch" + str(channel),
                data=results_df,
                palette=sns.color_palette("rocket", as_cmap=True),
                legend=None,
                alpha=0.75
            )
            norm = plt.Normalize(results_df['avg_feature_int_ch' + str(channel)].min(), results_df['avg_feature_int_ch' + str(channel)].max())
            sm = plt.cm.ScalarMappable(cmap="rocket", norm=norm)
            sm.set_array([])
            plt.colorbar(sm)
            plt.title('avg feature int - ' + 'ch' + str(channel))
            plt.savefig(save_path + '/_' + dim_method + '_avg_feature_int_ch' + str(channel) + '.png')
            plt.close()

            # feature_coverage_%_ch
            plt.figure(figsize=(10, 7))
            sns.scatterplot(
                x=x_dim, y=y_dim,
                hue="feature_coverage_%_ch" + str(channel),
                data=results_df,
                palette=sns.color_palette("Spectral_r", as_cmap=True),
                legend=None,
                alpha=0.75
            )
            norm = plt.Normalize(results_df['feature_coverage_%_ch' + str(channel)].min(), results_df['feature_coverage_%_ch' + str(channel)].max())
            sm = plt.cm.ScalarMappable(cmap="Spectral_r", norm=norm)
            sm.set_array([])
            plt.colorbar(sm)
            plt.title('feature coverage - ' + 'ch' + str(channel))
            plt.savefig(save_path + '/_' + dim_method + '_feature_coverage_pct_ch' + str(channel) + '.png')
            plt.close()

    results_df.to_csv(save_path + '/' + '_cell_datasheet_with_cluster_data.csv')
    return
