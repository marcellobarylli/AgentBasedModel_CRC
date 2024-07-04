""" 
This file is used the explore the gene dataset used in the paper cited below.
Guinney, J., Dienstmann, R., Wang, X. et al. The consensus molecular subtypes of colorectal cancer. Nat Med 21, 
1350–1356 (2015). https://doi.org/10.1038/nm.3967

Author: Robin van den Berg
Contact: rvdb7345@gmail.com
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, NMF, FastICA
from sklearn.mixture import BayesianGaussianMixture

dir_path = os.path.dirname(os.path.realpath(__file__))


def create_elbow_graph(df, min_clusters, max_clusters):
    """
    Create elbow graph from kmeans clustering
    :param df: The dataframe that you want to cluster
    :param min_clusters: minimum number of clusters
    :param max_clusters: maximum number of clusters
    :return:
    """

    distortions = []
    K = range(min_clusters, max_clusters)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, n_jobs=-1)
        kmeanModel.fit(df[df.columns.difference(['sample', 'dataset', 'CMS_network', 'CMS_RFclassifier',
                                             'CMS_final_network_plus_RFclassifier_in_nonconsensus_samples'])])
        distortions.append(kmeanModel.inertia_)

    plt.figure(figsize=(16, 12))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.grid()
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


def create_cluster_coloured_graph_kmean(df, num_clusters, columns_to_plot):
    """
    show scatter plot coloured according to respective cluster
    :param df: dataframe to cluster
    :param num_clusters: number of clusters to divide the
    :param columns_to_plot:
    :return:
    """
    kmeanModel = KMeans(n_clusters=num_clusters, n_jobs=-1)
    kmeanModel.fit(df[df.columns.difference(['sample', 'dataset', 'CMS_network', 'CMS_RFclassifier',
                                             'fin_dec'])])

    df['k_means'] = kmeanModel.predict(df[df.columns.difference(['sample', 'dataset', 'CMS_network', 'CMS_RFclassifier',
                                             'fin_dec'])])
    plt.scatter(df[columns_to_plot[0]], df[columns_to_plot[1]], c=df['k_means'],
                cmap=plt.cm.Set1)
    plt.title('K_Means', fontsize=18)
    plt.show()


def create_cluster_coloured_graph_meanshift(df, band, columns_to_plot):
    """
    show scatter plot coloured according to respective cluster
    :param df: dataframe to cluster
    :param num_clusters: number of clusters to divide the
    :param columns_to_plot:
    :return:
    """
    meanshiftModel = MeanShift(n_jobs=-1, bandwidth=band)
    meanshiftModel.fit(df[df.columns.difference(['sample', 'dataset', 'CMS_network', 'CMS_RFclassifier',
                                             'fin_dec'])])

    df['meanshift'] = meanshiftModel.predict(df[df.columns.difference(['sample', 'dataset', 'CMS_network',
                                                                  'CMS_RFclassifier',
                                             'fin_dec'])])
    plt.scatter(df[columns_to_plot[0]], df[columns_to_plot[1]], c=df['meanshift'],
                cmap=plt.cm.Set1)
    plt.title('mean_shift', fontsize=18)
    plt.show()


def create_cluster_coloured_graph_dbscan(df, epsilon, columns_to_plot):
    """
    show scatter plot coloured according to respective cluster
    :param df: dataframe to cluster
    :param num_clusters: number of clusters to divide the
    :param columns_to_plot:
    :return:
    """
    dbscanModel = DBSCAN(n_jobs=-1, eps=epsilon)
    # dbscanModel.fit(df[df.columns.difference(['sample', 'dataset', 'CMS_network', 'CMS_RFclassifier',
    #                                          'fin_dec'])])

    df['dbscan'] = dbscanModel.fit_predict(df[df.columns.difference(['sample', 'dataset', 'CMS_network',
                                                                  'CMS_RFclassifier',
                                             'fin_dec'])])
    plt.scatter(df[columns_to_plot[0]], df[columns_to_plot[1]], c=df['dbscan'],
                cmap=plt.cm.Set1)
    plt.title('dbscan', fontsize=18)
    plt.show()

def create_cluster_coloured_graph_gausmix(df, num_components, columns_to_plot):
    """
    show scatter plot coloured according to respective cluster
    :param df: dataframe to cluster
    :param num_clusters: number of clusters to divide the
    :param columns_to_plot:
    :return:
    """
    baygausmixModel = BayesianGaussianMixture(n_components=num_components, verbose=1)
    baygausmixModel.fit(df[df.columns.difference(['sample', 'dataset', 'CMS_network', 'CMS_RFclassifier',
                                             'fin_dec'])])

    df['baygausmix'] = baygausmixModel.predict(df[df.columns.difference(['sample', 'dataset', 'CMS_network',
                                                                  'CMS_RFclassifier',
                                             'fin_dec'])])
    plt.scatter(df[columns_to_plot[0]], df[columns_to_plot[1]], c=df['baygausmix'],
                cmap=plt.cm.Set1)
    plt.title('Bayesian Gaussian Mixture', fontsize=18)
    plt.show()

    print('The covariances of the mixture components: ', baygausmixModel.covariances_)

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    plt.figure(figsize=(20, 5))
    plt.yscale('log')
    dendrogram(linkage_matrix, **kwargs)
    plt.xticks(fontsize=12)
    plt.show()

def create_cluster_coloured_graph_aggl(df, num_components, columns_to_plot):
    """
    show scatter plot coloured according to respective cluster
    :param df: dataframe to cluster
    :param num_clusters: number of clusters to divide the
    :param columns_to_plot:
    :return:
    """
    agglomerativeModel = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='ward',
                                                 distance_threshold=0)
    df['agglomerative'] = agglomerativeModel.fit_predict(df[df.columns.difference(['sample', 'dataset', 'CMS_network',
                                                                                   'CMS_RFclassifier', 'fin_dec'])])

    model = agglomerativeModel.fit(df[df.columns.difference(['sample', 'dataset', 'CMS_network',
                                                                                   'CMS_RFclassifier', 'fin_dec'])])

    plt.figure()
    plt.scatter(df[columns_to_plot[0]], df[columns_to_plot[1]], c=df['agglomerative'],
                cmap=plt.cm.Set1)
    plt.title('agglomerative', fontsize=18)
    plt.show()

    print('The number of clusters found: ', agglomerativeModel.n_clusters_)
    print('The number of leaves: ', agglomerativeModel.n_leaves_)
    print(('The number of connected components: ', agglomerativeModel.n_connected_components_))

    plot_dendrogram(model, truncate_mode='level', p=5)

    print(model.get_params())

    # print('The covariances of the mixture components: ', agglomerativeModel.covariances_)

def create_cluster_dendrogram(df):
    """
    show scatter plot coloured according to respective cluster
    :param df: dataframe to cluster
    :param num_clusters: number of clusters to divide the
    :param columns_to_plot:
    :return:
    """
    linked = linkage(df[df.columns.difference(['sample', 'dataset', 'CMS_network',
                                               'CMS_RFclassifier', 'fin_dec'])], 'single', optimal_ordering=True)

    labelList = range(1, 11)

    plt.figure(figsize=(10, 7))
    dendrogram(linked,
               orientation='top',
               labels=labelList,
               distance_sort='descending',
               show_leaf_counts=True)
    plt.show()

def do_pca(df, num_components):
    pca = PCA(n_components=num_components)

    columns_strings = []

    for i in range(num_components):
        columns_strings.append('principal_component_' + str(i))

    principalComponents = \
        pca.fit_transform(df[df.columns.difference(['sample', 'dataset', 'CMS_network', 'CMS_RFclassifier',
                                             'fin_dec'])].to_numpy())
    principalDf = pd.DataFrame(data=principalComponents,
                               columns=columns_strings)

    df_pcomp = pd.concat([df, principalDf], axis=1)

    # show a graph coloured according to actual type
    plt.figure()
    for idx, group in df_pcomp.groupby('cat_code'):
        plt.plot(group['principal_component_0'], group['principal_component_1'], marker='.',
                 linestyle='', label=group['fin_dec'].values[0], markersize=2)
    plt.legend()
    plt.xlabel('principal component 1', fontsize=14)
    plt.ylabel('principal component 2', fontsize=14)
    plt.title('PCA results', fontsize=18)
    plt.grid()
    plt.show()

    print('the variance we explain is: ', pca.explained_variance_ratio_)


def do_nmf(df, num_components):
    nmf = NMF(n_components=num_components)

    columns_strings = []

    for i in range(num_components):
        columns_strings.append('principal_component_' + str(i))

    principalComponents = \
        nmf.fit_transform(abs(df[df.columns.difference(['sample', 'dataset', 'CMS_network', 'CMS_RFclassifier',
                                             'fin_dec'])].to_numpy()))
    principalDf = pd.DataFrame(data=principalComponents,
                               columns=columns_strings)

    df_pcomp = pd.concat([df, principalDf], axis=1)

    # show a graph coloured according to actual type
    plt.figure()
    for idx, group in df_pcomp.groupby('cat_code'):
        plt.plot(group['principal_component_0'], group['principal_component_1'], marker='.',
                 linestyle='', label=group['fin_dec'].values[0], markersize=2)
    plt.legend()
    plt.xlabel('nmf component 1', fontsize=14)
    plt.ylabel('nmf component 2', fontsize=14)
    plt.title('NMF results', fontsize=18)
    plt.grid()
    plt.show()

    print('The reconstruction error is: ', nmf.reconstruction_err_)

def do_ica(df, num_components):
    nmf = FastICA(n_components=num_components)

    columns_strings = []

    for i in range(num_components):
        columns_strings.append('principal_component_' + str(i))

    principalComponents = \
        nmf.fit_transform(abs(df[df.columns.difference(['sample', 'dataset', 'CMS_network', 'CMS_RFclassifier',
                                             'fin_dec'])].to_numpy()))
    principalDf = pd.DataFrame(data=principalComponents,
                               columns=columns_strings)

    df_pcomp = pd.concat([df, principalDf], axis=1)

    # show a graph coloured according to actual type
    plt.figure()
    for idx, group in df_pcomp.groupby('cat_code'):
        plt.plot(group['principal_component_0'], group['principal_component_1'], marker='.',
                 linestyle='', label=group['fin_dec'].values[0], markersize=2)
    plt.legend()
    plt.xlabel('ica component 1', fontsize=14)
    plt.ylabel('ica component 2', fontsize=14)
    plt.title('ICA results', fontsize=18)
    plt.grid()
    plt.show()



def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

# read in the data
df_crc_data = pd.read_table(os.path.join(dir_path, '../data/formatted_crc_data.txt'))
df_crc_data['sample'] = df_crc_data.index

df_labels = pd.read_table(os.path.join(dir_path, '../data/cms_labels_public_all.txt'))

# print(len(intersection(df_crc_data['sample'].values, df_labels['sample'].values)))

df_merged = pd.merge(df_crc_data, df_labels, on='sample')
df_merged.rename({'CMS_final_network_plus_RFclassifier_in_nonconsensus_samples': 'fin_dec'}, axis='columns',
                 inplace=True)
df_merged.fin_dec = pd.Categorical(df_merged.fin_dec)
df_merged['cat_code'] = df_merged.fin_dec.cat.codes

# df_merged = df_merged.loc[df_merged.fin_dec != 'NOLBL']

# show some graph of the data
# plt.figure()
# plt.scatter(df_GSE23878['GSM588828.CEL'], df_GSE23878['GSM588829.CEL'])
# plt.show()

# # create elbow graph
# min_clusters = 1
# max_clusters = 10
# create_elbow_graph(df_merged, min_clusters, max_clusters)

# plot a scatter plot coloured according to the clusters from kmeans
# num_clusters = 4
# columns_to_plot = ['153396', '1378']
# create_cluster_coloured_graph_kmean(df_merged, num_clusters, columns_to_plot)

# # plot a scatter plot coloured according to the clusters from mean shift
# bandwidth = 2
# columns_to_plot = ['153396', '1378']
# create_cluster_coloured_graph_meanshift(df_merged, bandwidth, columns_to_plot)


# plot a scatter plot coloured according to the clusters from dbscan
# epsilon = 9
# columns_to_plot = ['153396', '1378']
# create_cluster_coloured_graph_dbscan(df_merged, epsilon, columns_to_plot)

# plot a scatter plot coloured according to the clusters from gaus mix
# num_clusters = 8
# columns_to_plot = ['153396', '1378']
# create_cluster_coloured_graph_gausmix(df_merged, num_clusters, columns_to_plot)

# plot a scatter plot coloured according to the clusters from agglomerative clustering
num_clusters = None
columns_to_plot = ['153396', '1378']
create_cluster_coloured_graph_aggl(df_merged, num_clusters, columns_to_plot)

# create a dendrogram
# create_cluster_dendrogram(df_merged)

''' Decomposition analysis '''
# # do a principle component analysis
# num_components = 8
# do_pca(df_merged, num_components)

# do a nmf analysis
# num_components = 100
# do_nmf(df_merged, num_components)

# do a ica analysis
# num_components = 100
# do_ica(df_merged, num_components)


''' show graph of datapoints coloured according to CMS class '''
# show a graph coloured according to actual type
# plt.figure()
# for idx, group in df_merged.groupby('cat_code'):
#     plt.plot(group['57205'], group['1378'], marker='.', linestyle='', label=group['fin_dec'].values[0])
# plt.legend()
# plt.show()



