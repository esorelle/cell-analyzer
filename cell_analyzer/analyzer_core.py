#! usr/bin/python3

import os
import glob
import numpy as np
import skimage as sk
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
import seaborn as sns
from scipy import ndimage as ndi
from datetime import datetime as dt
from skimage import filters, feature, measure, morphology
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE


def process_image(img, max_cell_area, small_obj_size, name='temp.TIF', save_path = '.'):
    # performs optimized processing steps -- bg norm and watershed

    # global background normalization for light intensity variation
    blur_img = filters.gaussian(img, sigma = 1)
    bg = filters.threshold_local(img, 501)
    img_corr = img / bg
    img_corr = img_corr / np.max(img_corr)

    # threshold of normalized image
    otsu = filters.threshold_otsu(img_corr)
    otsu_mask = img_corr > otsu
    img_masked = img_corr * otsu_mask

    # QC filter to remove very noisy / low dynamic range images
    if np.sum(np.ndarray.flatten(otsu_mask)) > 0.98 * np.size(img):
        print('skipping: ' + name + '\n')
        print('[ reason: poor dynamic range (1) ]')
        return [], []

    # find local extrema
    coords = feature.peak_local_max(img_masked, min_distance=20)
    coords_T = []
    coords_T += coords_T + [[point[1], point[0]] for point in coords]
    coords_T = np.array(coords_T)
    distance = ndi.distance_transform_edt(otsu_mask)
    markers = np.zeros(np.shape(img))
    for i, point in enumerate(coords):
        markers[point[0], point[1]] = i

    # watershed from local extrema and mask
    ws2 = morphology.watershed(-distance, markers, mask=otsu_mask, connectivity=2, watershed_line=True)
    ws3 = ws2 > 0
    ws4 = morphology.remove_small_objects(ws3, small_obj_size)

    # get labeled cell regions
    labeled_cells = measure.label(ws4)
    cell_props = measure.regionprops(labeled_cells, img)

    # save processing steps for review & QC
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,7))
    ax[0, 0].imshow(img, cmap='gray')
    ax[0, 0].set_title('raw')
    ax[0, 1].imshow(img_corr)
    ax[0, 1].set_title('corrected')
    ax[0, 2].imshow(otsu_mask)
    ax[0, 2].set_title('otsu mask')
    ax[1, 0].imshow(distance, cmap='magma')
    ax[1, 0].set_title('distance transform')
    ax[1, 1].imshow(ws2, cmap='magma')
    ax[1, 1].plot(coords[:,1], coords[:,0], c='red', marker = '*', linestyle='', markersize=3)
    ax[1, 1].set_title('watershed + extrema')
    ax[1, 2].imshow(labeled_cells, cmap='nipy_spectral')
    ax[1, 2].set_title('cell regions')
    plt.tight_layout()
    plt.savefig(save_path + '/' + name[:-4] + '_cell_labels.png')
    plt.close()

    return labeled_cells, cell_props



def read_and_process_directory(base_directory, max_cell_area, small_obj_size):
    # process all .tif files in a passed directory and returns results dataframe (.csv)

    # set up paths
    time_stamp = dt.now().strftime('%Y_%m_%d_%H_%M_%S')
    save_path = '_extracted_data_%s' % time_stamp
    save_path = os.path.join(base_directory, save_path)
    print('base: ' + base_directory)
    print('save: ' + save_path)
    os.mkdir(save_path)

    # get paths for images in base directory
    image_list = glob.glob(os.path.join(base_directory, '*.TIF')) # '.TIF' or '.tif'
    image_list = image_list + glob.glob(os.path.join(base_directory, '*.tif'))

    # initialize results dataframe
    results_df = pd.DataFrame()

    # iteratively read in images by filenames
    for i, img_path in enumerate(image_list):
        img = plt.imread(img_path)
        if np.ndim(img) > 2:    # account for duplicated channels in exported cellomics files
            img = img[:,:,0]
        name = os.path.basename(img_path)
        labeled_cells, cell_props = process_image(img, max_cell_area, small_obj_size, name, save_path)

        # save labeled cell images as plots
        # plt.imsave(save_path + '/' + name[:-4] + '_cell_labels.png', labeled_cells, cmap='nipy_spectral')

        # save all cell quant in results dataframe
        for c, cell in enumerate(cell_props):
            results_df = results_df.append({'_image_name': name, '_img_cell_#': c,
                '_clone': name[14:17], '_moi': float(name[18:20]), '_days_post_inf': float(name[22:24]),
                'area': cell_props[c]['area'], 'max': cell_props[c]['max_intensity'],
                'min': cell_props[c]['min_intensity'], 'mean': cell_props[c]['mean_intensity'],
                'extent': cell_props[c]['extent'], 'eccentricity': cell_props[c]['eccentricity'],
                'perimeter': cell_props[c]['perimeter'], 'major_axis': cell_props[c]['major_axis_length'],
                'minor_axis': cell_props[c]['minor_axis_length']}, ignore_index=True)

    results_df.to_csv(save_path + '/' + '_cell_datasheet.csv')
    return results_df, save_path


def cluster_results(results_df, save_path, num_clust):
    # simple unsupervised hierarchical clsutering based on cell properties

    # encode clone number as variable
    clone_key = list(np.unique(results_df['_clone']))
    clone_temp = results_df['_clone'].values
    clone_IDs = [ clone_key.index(item) for item in clone_temp ]

    # dendrogram
    data = results_df.iloc[:, 6:].values
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

    # add columns for tsne labeling
    data_embedded = TSNE(n_components=2).fit_transform(data)
    results_df['tsne-2d-one'] = data_embedded[:,0]
    results_df['tsne-2d-two'] = data_embedded[:,1]
    results_df['cluster_ID'] = ID[:]
    results_df['clone_ID'] = clone_IDs


    # tSNE clusters
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="cluster_ID",
        data=results_df,
        palette=sns.color_palette("Spectral", num_clust),
        legend="full",
        alpha=0.75
    )
    plt.title('cell clustering')
    plt.savefig(save_path + '/' + '_tsne_clusters.png')
    plt.close()


    # tSNE clone IDs
    plt.figure(figsize=(10, 7))
    g = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="clone_ID",
        data=results_df,
        palette=sns.color_palette("hls", len(clone_key)),
        legend="full",
        alpha=0.75
    )
    plt.title('cell clones')
    for t, l in zip(g.legend().texts, clone_key): t.set_text(l)
    plt.savefig(save_path + '/' + '_tsne_clones.png')
    plt.close()


    # tSNE days post-infection
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="_days_post_inf",
        data=results_df,
        palette=sns.color_palette("magma", len(np.unique(results_df['_days_post_inf']))),
        legend="full",
        alpha=0.75
    )
    plt.title('days post infection')
    plt.savefig(save_path + '/' + '_tsne_dpi.png')
    plt.close()


    # tSNE MOI
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="_moi",
        data=results_df,
        palette=sns.color_palette("viridis", len(np.unique(results_df['_moi']))),
        legend="full",
        alpha=0.75
    )
    plt.title('EBV moi')
    plt.savefig(save_path + '/' + '_tsne_moi.png')
    plt.close()


    # tSNE cell areas
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="area",
        data=results_df,
        palette=sns.color_palette("Spectral_r", as_cmap=True),
        legend=None,
        alpha=0.75
    )
    norm = plt.Normalize(results_df['area'].min(), results_df['area'].max())
    sm = plt.cm.ScalarMappable(cmap="Spectral_r", norm=norm)
    sm.set_array([])
    plt.colorbar(sm)
    plt.title('cell area')
    plt.savefig(save_path + '/' + '_tsne_area.png')
    plt.close()


    # tSNE mean intensities
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="mean",
        data=results_df,
        palette=sns.color_palette("Greens", as_cmap=True),
        legend=None,
        alpha=0.75
    )
    norm = plt.Normalize(results_df['mean'].min(), results_df['mean'].max())
    sm = plt.cm.ScalarMappable(cmap="Greens", norm=norm)
    sm.set_array([])
    plt.colorbar(sm)
    plt.title('mean cell intensity (a.u)')
    plt.savefig(save_path + '/' + '_tsne_mean_int.png')
    plt.close()


    # tSNE min intensities
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="min",
        data=results_df,
        palette=sns.color_palette("Blues", as_cmap=True),
        legend=None,
        alpha=0.75
    )
    norm = plt.Normalize(results_df['min'].min(), results_df['min'].max())
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
    sm.set_array([])
    plt.colorbar(sm)
    plt.title('min cell intensity (a.u)')
    plt.savefig(save_path + '/' + '_tsne_min_int.png')
    plt.close()


    # tSNE max intensities
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="max",
        data=results_df,
        palette=sns.color_palette("Reds", as_cmap=True),
        legend=None,
        alpha=0.75
    )
    norm = plt.Normalize(results_df['max'].min(), results_df['max'].max())
    sm = plt.cm.ScalarMappable(cmap="Reds", norm=norm)
    sm.set_array([])
    plt.colorbar(sm)
    plt.title('max cell intensity (a.u)')
    plt.savefig(save_path + '/' + '_tsne_max_int.png')
    plt.close()


    # tSNE eccentricity
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="eccentricity",
        data=results_df,
        palette=sns.color_palette("Purples", as_cmap=True),
        legend=None,
        alpha=0.75
    )
    norm = plt.Normalize(results_df['eccentricity'].min(), results_df['eccentricity'].max())
    sm = plt.cm.ScalarMappable(cmap="Purples", norm=norm)
    sm.set_array([])
    plt.colorbar(sm)
    plt.title('cell eccentricity (a.u)')
    plt.savefig(save_path + '/' + '_tsne_eccentricity.png')
    plt.close()


    # tSNE extent
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="extent",
        data=results_df,
        palette=sns.color_palette("Oranges", as_cmap=True),
        legend=None,
        alpha=0.75
    )
    norm = plt.Normalize(results_df['extent'].min(), results_df['extent'].max())
    sm = plt.cm.ScalarMappable(cmap="Oranges", norm=norm)
    sm.set_array([])
    plt.colorbar(sm)
    plt.title('cell extent (a.u)')
    plt.savefig(save_path + '/' + '_tsne_extent.png')
    plt.close()

    return
