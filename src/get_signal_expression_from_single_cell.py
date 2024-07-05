import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import time


def get_indices_to_group(data_type, tissue_class='Tumor'):
    annotations_df = pd.read_csv('../data/GSE132465_GEO_processed_CRC_10X_cell_annotation.txt', delimiter='\t')
    annotations_df['bulk_classification'] = annotations_df['Patient'].map(BULK_CLASSIFICATION)
    if tissue_class != 'All':
        indices_to_group = annotations_df.loc[annotations_df.Class == tissue_class].groupby(['Patient', 'Cell_subtype'])['Index'].apply(list)
    elif tissue_class == 'All':
        indices_to_group = annotations_df.groupby(['Patient', 'Cell_subtype'])['Index'].apply(list)
    return indices_to_group, annotations_df

def estimate_chunks(file_path, chunksize):
    file_size = os.path.getsize(file_path)
    first_chunk = next(pd.read_csv(file_path, delimiter='\t', chunksize=chunksize))
    estimated_chunks = file_size / (first_chunk.memory_usage(deep=True).sum())
    return int(estimated_chunks)

def process_chunk(chunk, indices_to_group, n):
    data_dict = {}
    for patient, new_df in indices_to_group.groupby(level=0):
        for subtype, newer_df in new_df.groupby(level=1):
            cols = list(newer_df.values)[0]
            if set(cols).issubset(set(chunk.columns)):
                for col in cols:
                    key = (str(patient), str(subtype), col)
                    data_dict[key] = chunk[col]
    
    index = pd.MultiIndex.from_tuples(data_dict.keys(), names=['patient', 'subtype', 'samples'])
    averaged_chunk = pd.DataFrame(data_dict.values(), index=index, columns=chunk.index)
    
    return averaged_chunk


## HERE THE CHUNKED VERSION FOR MEMORY EFFICIENCY
# def create_selection_sc_gene_expression_df(indices_to_group, scale='raw', n=100, tissue_class='Tumor', chunksize=1000):
#     indices_to_group = indices_to_group.apply(lambda x: x if len(x) <= n else random.sample(x, n))
#     flat_list = [item for sublist in indices_to_group.values for item in sublist]
    
#     print('The number of columns to load would be: ', len(flat_list))
    
#     start = time.time()
#     print('Start to load the dataframe')
    
#     file_path = '../data/GSE132465_GEO_processed_CRC_10X_raw_UMI_count_matrix.txt' if scale == 'raw' else '../data/GSE132465_GEO_processed_CRC_10X_natural_log_TPM_matrix.txt'
#     output_file = '../data/selected_sc_gene_express_{}_{}.csv'.format(tissue_class, scale)
    
#     estimated_chunks = estimate_chunks(file_path, chunksize)
    
#     processed_chunks = 0
    
#     with tqdm(total=estimated_chunks, desc="Processing chunks") as pbar:
#         for chunk in pd.read_csv(file_path, delimiter='\t', usecols=['Index'] + flat_list, index_col=['Index'], chunksize=chunksize):
#             processed_chunk = process_chunk(chunk, indices_to_group, n)
            
#             # Write processed chunk directly to file
#             if processed_chunks == 0:
#                 processed_chunk.to_csv(output_file, mode='w')
#             else:
#                 processed_chunk.to_csv(output_file, mode='a', header=False)
            
#             processed_chunks += 1
            
#             pbar.update(1)
#             pbar.set_postfix({'Memory (MB)': f'{processed_chunk.memory_usage().sum() / 1e6:.2f}'})
            
#             # Clear memory
#             del processed_chunk
            
#     print('Finished processing. Total time elapsed: ', time.time() - start)
#     print('Saved to CSV')

# # AND HERE THE NON-CHUNKED VERSION
# def create_selection_sc_gene_expression_df(indices_to_group, scale='raw', n=100, tissue_class='Tumor'):
#     indices_to_group = indices_to_group.apply(lambda x: x if len(x) <= n else random.sample(x, n))

#     flat_list = [item for sublist in indices_to_group.values for item in sublist]

#     print('The number of columns to load would be: ', len(flat_list))

#     start = time.time()
#     print('Start to load the dataframe')
#     if scale == 'raw':
#         raw_counts_df = pd.read_csv('../data/GSE132465_GEO_processed_CRC_10X_raw_UMI_count_matrix.txt',
#                                     delimiter='\t', usecols=['Index'] + flat_list, index_col=['Index'])
#     elif scale == 'log':
#         raw_counts_df = pd.read_csv('../data/GSE132465_GEO_processed_CRC_10X_natural_log_TPM_matrix.txt',
#                                     delimiter='\t', usecols=['Index'] + flat_list, index_col=['Index'])
#     print('Finished loading the dataframe after: ', time.time() - start)

#     averaged_df = pd.DataFrame(index=raw_counts_df.index)
#     for patient, new_df in indices_to_group.groupby(level=0):
#         for subtype, newer_df in new_df.groupby(level=1):
#             averaged_df[list(zip([str(patient)] * n, [str(subtype)] * n, list(newer_df.values[0])))] = \
#                 raw_counts_df[list(newer_df.values)[0]]
#             averaged_df[list(zip([str(patient)] * n, [str(subtype)] * n, list(newer_df.values[0])))] = \
#                 raw_counts_df[list(newer_df.values)[0]]

#     averaged_df.columns = pd.MultiIndex.from_tuples(list(averaged_df.columns.values),
#                                                     names=['patient', 'subtype', 'samples'])

#     print('This much time has elapsed: ', time.time() - start)

#     averaged_df.to_csv('../data/selected_sc_gene_express_{}_{}.csv'.format(tissue_class, scale))

# NO MEMORY FRAGMENTATION
def create_selection_sc_gene_expression_df(indices_to_group, scale='raw', n=100, tissue_class='Tumor'):
    indices_to_group = indices_to_group.apply(lambda x: x if len(x) <= n else random.sample(x, n))
    flat_list = [item for sublist in indices_to_group.values for item in sublist]
    print('The number of columns to load would be: ', len(flat_list))
    
    start = time.time()
    print('Start to load the dataframe')
    
    file_path = '../data/GSE132465_GEO_processed_CRC_10X_raw_UMI_count_matrix.txt' if scale == 'raw' else '../data/GSE132465_GEO_processed_CRC_10X_natural_log_TPM_matrix.txt'
    raw_counts_df = pd.read_csv(file_path, delimiter='\t', usecols=['Index'] + flat_list, index_col=['Index'])
    
    print('Finished loading the dataframe after: ', time.time() - start)
    
    # Create a list to store the new column names and data
    new_columns = []
    new_data = []
    
    for patient, new_df in indices_to_group.groupby(level=0):
        for subtype, newer_df in new_df.groupby(level=1):
            columns = list(newer_df.values[0])
            new_columns.extend([(str(patient), str(subtype), col) for col in columns])
            new_data.append(raw_counts_df[columns].values)
    
    # Create the averaged_df with pre-allocated columns
    averaged_df = pd.DataFrame(np.hstack(new_data), index=raw_counts_df.index, columns=pd.MultiIndex.from_tuples(new_columns, names=['patient', 'subtype', 'samples']))
    
    print('This much time has elapsed: ', time.time() - start)
    averaged_df.to_csv('../data/selected_sc_gene_express_{}_{}2.csv'.format(tissue_class, scale))

BULK_CLASSIFICATION = {'SMC01': 'CMS3', 'SMC02': 'CMS4', 'SMC03': 'CMS1', 'SMC04': 'CMS4', 'SMC05': 'CMS3',
                       'SMC06': 'CMS1', 'SMC07': 'CMS2', 'SMC08': 'CMS1', 'SMC09': 'CMS2', 'SMC10': 'CMS1',
                       'SMC11': 'CMS2', 'SMC14': 'CMS4', 'SMC15': 'CMS1', 'SMC16': 'CMS3', 'SMC17': 'CMS4',
                       'SMC18': 'CMS2', 'SMC19': 'CMS3', 'SMC20': 'CMS4', 'SMC21': 'CMS2', 'SMC22': 'CMS2',
                       'SMC23': 'CMS2', 'SMC24': 'CMS4', 'SMC25': 'CMS2'}

# Main execution
tissue_class = 'Tumor'
scale = 'raw'
indices_to_group, annotations_df = get_indices_to_group(data_type='GEO', tissue_class=tissue_class)

## CHUNKED
# create_selection_sc_gene_expression_df(indices_to_group, scale=scale, n=100000, tissue_class=tissue_class, chunksize=1000)

## NON-CHUNKED
create_selection_sc_gene_expression_df(indices_to_group, scale=scale, n=100000, tissue_class=tissue_class)

########################################### FULL VERSION BELOW ###########################################################

# import math
# import time
# import random
# import anndata
# import ast
# import json

# # import modin.pandas as pd
# import pandas as pd
# import seaborn as sns
# import numpy as np

# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 7)

# # from dask import dataframe as pdd
# import scanpy as sc

# import matplotlib.pyplot as plt

# from collections import defaultdict, Mapping
# from scipy.sparse import issparse
# from scanpy import _utils


# def correlation_matrix(adata, name_list=None, groupby=None, group=None, n_genes=20, data='Complete', method='pearson',
#                        annotation_key=None):
#     """Calculate correlation matrix.
#         Calculate a correlation matrix for genes strored in sample annotation using rank_genes_groups.py
#         Parameters
#         ----------
#         adata : :class:`~scanpy.api.AnnData`
#             Annotated data matrix.
#         name_list : list, optional (default: None)
#             Takes a list of genes for which to calculate the correlation matrix
#         groupby : `str`, optional (default: None)
#             If no name list is passed, genes are selected from the
#             results of rank_gene_groups. Then this is the key of the sample grouping to consider.
#             Note that in this case also a group index has to be specified.
#         group : `int`, optional (default: None)
#             Group index for which the correlation matrix for top_ranked genes should be calculated.
#             Currently only int is supported, will change very soon
#         n_genes : `int`, optional (default: 20)
#             For how many genes to calculate correlation matrix? If specified, cuts the name list
#             (in whatever order it is passed).
#         data : {'Complete', 'Group', 'Rest'}, optional (default: 'Complete')
#             At the moment, this is only relevant for the case that name_list is drawn from rank_gene_groups results.
#             If specified, collects mask for the called group and then takes only those cells specified.
#             If 'Complete', calculate correlation using full data
#             If 'Group', calculate correlation within the selected group.
#             If 'Rest', calculate corrlation for everything except the group
#         method : {‘pearson’, ‘kendall’, ‘spearman’} optional (default: 'pearson')
#             Which kind of correlation coefficient to use
#             pearson : standard correlation coefficient
#             kendall : Kendall Tau correlation coefficient
#             spearman : Spearman rank correlation
#         annotation_key: String, optional (default: None)
#             Allows to define the name of the anndata entry where results are stored.
#     """

#     # TODO: At the moment, only works for int identifiers

#     ### If no genes are passed, selects ranked genes from sample annotation.
#     ### At the moment, only calculate one table (Think about what comes next)
#     if name_list is None:
#         name_list = list()
#         name_list = adata.uns['rank_genes_groups']['names'][group][:n_genes]
#     else:
#         if len(name_list) > n_genes:
#             name_list = name_list[0:n_genes]

#     # If special method (later) , truncate
#     adata_relevant = adata[:, name_list]
#     # This line just makes group_mask access easier. Nothing else but 'all' will stand here.
#     groups = 'all'
#     if data is 'Complete' or groupby is None:
#         if issparse(adata_relevant.X):
#             Data_array = adata_relevant.X.todense()
#         else:
#             Data_array = adata_relevant.X
#     else:
#         # get group_mask
#         groups_order, groups_masks = _utils.select_groups(
#             adata, groups, groupby)

#         if isinstance(group, str):
#             group_str = group
#             group = np.where(groups_order == group)[0][0]

#         if data is 'Group':
#             if issparse(adata_relevant.X):
#                 Data_array = adata_relevant.X[groups_masks[group], :].todense()
#             else:
#                 Data_array = adata_relevant.X[groups_masks[group], :]
#         elif data is 'Rest':
#             if issparse(adata_relevant.X):
#                 Data_array = adata_relevant.X[~groups_masks[group], :].todense()
#             else:
#                 Data_array = adata_relevant.X[~groups_masks[group], :]
#         else:
#             print('data argument should be either <Complete> or <Group> or <Rest>')

#     # Distinguish between sparse and non-sparse data

#     DF_array = pd.DataFrame(Data_array, columns=name_list)
#     cor_table = DF_array.corr(method=method)
#     if annotation_key is None:
#         if groupby is None:
#             adata.uns['Correlation_matrix'] = cor_table
#         else:
#             adata.uns['Correlation_matrix' + groupby + str(group_str)] = cor_table
#     else:
#         adata.uns[annotation_key] = cor_table


# # define funcitons
# def Tree():
#     return defaultdict(Tree)


# def load_pathways_ranked(side, cutoff_p, libraries, scale, tissue_class):
#     ''' Load the ranked pathway dfs '''

#     cms1_ranked = \
#         pd.read_csv('../results/pathway_comparison/pathway_ranking_cms1_sign_{}_{}_{}_{}_{}.csv'.format(side,
#                                                                                                         cutoff_p,
#                                                                                                         libraries,
#                                                                                                         tissue_class,
#                                                                                                         scale))
#     cms2_ranked = \
#         pd.read_csv('../results/pathway_comparison/pathway_ranking_cms2_sign_{}_{}_{}_{}_{}.csv'.format(side,
#                                                                                                         cutoff_p,
#                                                                                                         libraries,
#                                                                                                         tissue_class,
#                                                                                                         scale))
#     cms3_ranked = \
#         pd.read_csv('../results/pathway_comparison/pathway_ranking_cms3_sign_{}_{}_{}_{}_{}.csv'.format(side,
#                                                                                                         cutoff_p,
#                                                                                                         libraries,
#                                                                                                         tissue_class,
#                                                                                                         scale))
#     cms4_ranked = \
#         pd.read_csv('../results/pathway_comparison/pathway_ranking_cms4_sign_{}_{}_{}_{}_{}.csv'.format(side,
#                                                                                                         cutoff_p,
#                                                                                                         libraries,
#                                                                                                         tissue_class,
#                                                                                                         scale))

#     return cms1_ranked, cms2_ranked, cms3_ranked, cms4_ranked


# def create_cutoff_graph(array_of_genes, accompagnying_pvalues, p_cutoffs, cell_type, tissue_class):
#     lengths_at_pvalue_cms1 = []
#     lengths_at_pvalue_cms2 = []
#     lengths_at_pvalue_cms3 = []
#     lengths_at_pvalue_cms4 = []

#     for p_cutoff in p_cutoffs:
#         # select only the genes that are significantly different expressed
#         current_gene_array = array_of_genes.copy()

#         for i in range(len(current_gene_array)):
#             for j in range(len(current_gene_array[0])):
#                 if accompagnying_pvalues[i][j] > p_cutoff:
#                     current_gene_array[i][j] = ''
#                 else:
#                     continue

#         diff_exp_genes = {}
#         diff_exp_genes['CMS1'] = [tup[0] for tup in current_gene_array]
#         diff_exp_genes['CMS2'] = [tup[1] for tup in current_gene_array]
#         diff_exp_genes['CMS3'] = [tup[2] for tup in current_gene_array]
#         diff_exp_genes['CMS4'] = [tup[3] for tup in current_gene_array]

#         lengths_at_pvalue_cms1.append(len(list(filter(lambda x: x != '', diff_exp_genes['CMS1']))))
#         lengths_at_pvalue_cms2.append(len(list(filter(lambda x: x != '', diff_exp_genes['CMS2']))))
#         lengths_at_pvalue_cms3.append(len(list(filter(lambda x: x != '', diff_exp_genes['CMS3']))))
#         lengths_at_pvalue_cms4.append(len(list(filter(lambda x: x != '', diff_exp_genes['CMS4']))))

#     plt.figure()
#     plt.title('P-value cut-off for {}'.format(cell_type), fontsize=14)
#     plt.plot(p_cutoffs, lengths_at_pvalue_cms1, label='CMS1')
#     plt.plot(p_cutoffs, lengths_at_pvalue_cms2, label='CMS2')
#     plt.plot(p_cutoffs, lengths_at_pvalue_cms3, label='CMS3')
#     plt.plot(p_cutoffs, lengths_at_pvalue_cms4, label='CMS4')
#     plt.ylim(0, max(max(lengths_at_pvalue_cms1), max(lengths_at_pvalue_cms2), max(lengths_at_pvalue_cms3),
#                     max(lengths_at_pvalue_cms4)))
#     plt.xlabel('cut off p-value', fontsize=14)
#     plt.ylabel('# of significant genes', fontsize=14)
#     # plt.xscale('log')
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.legend(fontsize=14)
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig('../results/pvalue_cutoff_{}_{}.png'.format(cell_type, tissue_class), dpi=300)
#     plt.show()


# def grouped_obs_mean(adata, group_key, layer=None, gene_symbols=None):
#     if layer is not None:
#         getX = lambda x: x.layers[layer]
#     else:
#         getX = lambda x: x.X
#     if gene_symbols is not None:
#         new_idx = gene_symbols
#     else:
#         new_idx = adata.var_names

#     grouped = adata.obs.groupby(group_key)
#     out = pd.DataFrame(
#         np.zeros((len(new_idx), len(grouped)), dtype=np.float64),
#         columns=list(grouped.groups.keys()),
#         index=new_idx
#     )

#     out_std = pd.DataFrame(
#         np.zeros((len(new_idx), len(grouped)), dtype=np.float64),
#         columns=list(grouped.groups.keys()),
#         index=new_idx
#     )

#     for group, idx in grouped.indices.items():
#         X = getX(adata[idx, adata.var_names.isin(new_idx)])
#         out[group] = np.ravel(X.mean(axis=0, dtype=np.float64))
#         out_std[group] = np.ravel(X.std(axis=0, dtype=np.float64))

#     return out, out_std


# def plot_correlation_matrix(adata, groupby=None, group=None, corr_matrix=None, annotation_key=None):
#     """Plot correlation matrix.
#             Plot a correlation matrix for genes strored in sample annotation using rank_genes_groups.py
#             Parameters
#             ----------
#             adata : :class:`~scanpy.api.AnnData`
#                 Annotated data matrix.
#             groupby : `str`, optional (default: None)
#                 If specified, searches data_annotation for correlation_matrix+groupby+str(group)
#             group : int
#                 Identifier of the group (necessary if and only if groupby is also specified)
#             corr_matrix : DataFrame, optional (default: None)
#                 Correlation matrix as a DataFrame (annotated axis) that can be transferred manually if wanted
#             annotation_key: `str`, optional (default: None)
#                 If specified, looks in data annotation for this key.
#         """

#     # TODO: At the moment, noly works for int identifiers

#     if corr_matrix is None:
#         # This will produce an error if he annotation doesn't exist, which is okay
#         if annotation_key is None:
#             if groupby is None:
#                 corr_matrix = adata.uns['Correlation_matrix']
#             else:
#                 corr_matrix = adata.uns['Correlation_matrix' + groupby + str(group)]
#             # Throws error if does not exist
#         else:
#             corr_matrix = adata.uns[annotation_key]

#     # Set up mask
#     mask = np.zeros_like(corr_matrix, dtype=np.bool)
#     di = np.diag_indices(len(corr_matrix.axes[0]))
#     mask[di] = True

#     f, ax = plt.subplots(figsize=(11, 9))

#     cmap = sns.diverging_palette(240, 10, as_cmap=True)

#     sns.heatmap(corr_matrix, mask=mask, cmap=cmap,
#                 square=True, linewidths=.5, cbar_kws={"shrink": .5}, center=0)
#     if annotation_key is None:
#         if groupby is None:
#             plt.title('Correlation Matrix')
#         else:
#             plt.title('Correlation Matrix for Group ' + str(group) + "in " + groupby)
#     else:
#         plt.title('Correlation Matrix for' + annotation_key)
#     plt.show()


# def get_indices_to_group(data_type, tissue_class='Tumor'):
#     annotations_df = pd.read_csv('../data/GSE132465_GEO_processed_CRC_10X_cell_annotation.txt', delimiter='\t')
#     patients = annotations_df.sort_values(by=['Patient'])['Patient'].unique()

#     annotations_df['bulk_classification'] = annotations_df['Patient'].map(BULK_CLASSIFICATION)

#     assert tissue_class in ['Tumor', 'Normal', 'All'], 'sample type not recognized'

#     if tissue_class != 'All':
#         indices_to_group = \
#         annotations_df.loc[annotations_df.Class == tissue_class].groupby(['Patient', 'Cell_subtype'])[
#             'Index'].apply(list)
#     elif tissue_class == 'All':
#         indices_to_group = annotations_df.groupby(['Patient', 'Cell_subtype'])[
#             'Index'].apply(list)

#     return indices_to_group, annotations_df


# def plot_sc_type_classification(annotations_df):
#     # check the type distribution
#     nrows = 4
#     ncols = 6
#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
#     idx = 0
#     for tup, patient_df in annotations_df[(annotations_df.Cell_type == 'Epithelial cells') &
#                                           (annotations_df.Class == 'Tumor')].groupby(
#         by=['Patient', 'Class', 'Sample', 'Cell_type']):
#         row = math.floor(idx / ncols)
#         col = math.floor(idx - row * ncols)
#         counts = patient_df.groupby(['Cell_subtype'])['Cell_subtype'].count()
#         colors = []
#         for CMS in ['CMS1', 'CMS2', 'CMS3', 'CMS4']:
#             if CMS not in counts.index:
#                 counts[CMS] = 0

#             colors.append('blue' if not patient_df['bulk_classification'].values[0] == CMS else 'red')
#         counts.sort_index(inplace=True)
#         # print(patient_df.groupby(['Cell_subtype'])['Cell_subtype'].count())
#         print(colors)
#         counts.plot(ax=axes[row, col], kind='bar', color=colors)
#         axes[row, col].axes.get_xaxis().set_visible(False)
#         axes[row, col].set_title(tup[0])
#         idx += 1

#     plt.axis(False)
#     # Set common labels
#     fig.text(0.5, 0.02, 'CMS1, CMS2, CMS3, CMS4', ha='center', va='center')
#     fig.text(0.02, 0.5, 'Counts', ha='center', va='center', rotation='vertical')
#     plt.tight_layout()
#     plt.savefig('../figures/classification_ind_cells.pdf')
#     plt.show()


# def save_different_subtypes_per_patient(patient_df):
#     # check the types of cells that are present in the dataset
#     tree = Tree()
#     for tup, patient_df in annotations_df.groupby(by=['Patient', 'Class', 'Cell_type']):
#         print('In the following, we get: {}'.format(tup))
#         print('The subtypes in this sample are: {}'.format(patient_df['Cell_subtype'].unique()))
#         tree[tup[0]][tup[1]][tup[2]] = patient_df['Cell_subtype'].unique()

#     df_subtypes = pd.DataFrame.from_dict({(i, j): tree[i][j]
#                                           for i in tree.keys()
#                                           for j in tree[i].keys()},
#                                          orient='index')

#     df_subtypes.to_csv('../figures/types_of_cells.csv')


def create_average_sc_gene_expression_df(indices_to_group, scale='raw', n=100):
    indices_to_group = indices_to_group.apply(lambda x: x if len(x) <= n else random.sample(x, n))

    flat_list = [item for sublist in indices_to_group.values for item in sublist]

    print(flat_list)
    print('The number of columns to load would be: ', len(flat_list))

    start = time.time()
    print('Start to load the dataframe')
    if scale == 'raw':
        raw_counts_df = pd.read_csv('../data/GSE132465_GEO_processed_CRC_10X_raw_UMI_count_matrix.txt',
                                    delimiter='\t', usecols=['Index'] + flat_list, index_col=['Index'])
    elif scale == 'log':
        raw_counts_df = pd.read_csv('../data/GSE132465_GEO_processed_CRC_10X_natural_log_TPM_matrix.txt',
                                    delimiter='\t', usecols=['Index'] + flat_list, index_col=['Index'])
    print('Finished loading the dataframe after: ', time.time() - start)

    averaged_df = pd.DataFrame(index=raw_counts_df.index)
    for patient, new_df in indices_to_group.groupby(level=0):
        print(patient, new_df)
        for subtype, newer_df in new_df.groupby(level=1):
            print(subtype, newer_df)
            averaged_df[str(patient) + '-' + str(subtype)] = raw_counts_df[list(newer_df.values)[0]].mean(axis=1)

    print('This much time has elapsed: ', time.time() - start)

    averaged_df.to_csv('../data/average_sc_gene_express.csv')


def create_selection_sc_gene_expression_df(indices_to_group, scale='raw', n=100, tissue_class='Tumor'):
    indices_to_group = indices_to_group.apply(lambda x: x if len(x) <= n else random.sample(x, n))

    flat_list = [item for sublist in indices_to_group.values for item in sublist]

    print('The number of columns to load would be: ', len(flat_list))

    start = time.time()
    print('Start to load the dataframe')
    if scale == 'raw':
        raw_counts_df = pd.read_csv('../data/GSE132465_GEO_processed_CRC_10X_raw_UMI_count_matrix.txt',
                                    delimiter='\t', usecols=['Index'] + flat_list, index_col=['Index'])
    elif scale == 'log':
        raw_counts_df = pd.read_csv('../data/GSE132465_GEO_processed_CRC_10X_natural_log_TPM_matrix.txt',
                                    delimiter='\t', usecols=['Index'] + flat_list, index_col=['Index'])
    print('Finished loading the dataframe after: ', time.time() - start)

    averaged_df = pd.DataFrame(index=raw_counts_df.index)
    for patient, new_df in indices_to_group.groupby(level=0):
        for subtype, newer_df in new_df.groupby(level=1):
            averaged_df[list(zip([str(patient)] * n, [str(subtype)] * n, list(newer_df.values[0])))] = \
                raw_counts_df[list(newer_df.values)[0]]
            averaged_df[list(zip([str(patient)] * n, [str(subtype)] * n, list(newer_df.values[0])))] = \
                raw_counts_df[list(newer_df.values)[0]]

    averaged_df.columns = pd.MultiIndex.from_tuples(list(averaged_df.columns.values),
                                                    names=['patient', 'subtype', 'samples'])

    print('This much time has elapsed: ', time.time() - start)

    averaged_df.to_csv('../data/selected_sc_gene_express_{}_{}.csv'.format(tissue_class, scale))


def save_expression_data_as_format_for_scanpy(cell_type_to_look_at, select_object, tissue_class, scale):
    print('doing calculations for {}'.format(cell_type_to_look_at))
    averaged_expression_df = pd.read_csv('../data/selected_sc_gene_express_{}_{}.csv'.format(tissue_class, scale),
                                         index_col=[0],
                                         header=[0, 1, 2])

    # get the dataframe correctly orientated
    averaged_expression_df = averaged_expression_df.transpose()

    # create columns with the cell type and cms subtype separated
    obs_df = pd.DataFrame(list(averaged_expression_df.index.values), columns=['patient', 'subtype', 'sample'])
    obs_df['CMS_subtype'] = 0
    obs_df['CMS_subtype'] = obs_df['patient'].map(BULK_CLASSIFICATION)

    obs_df = pd.merge(obs_df, annotations_df[['sample', 'Cell_type']], on='sample')

    # put the different genes in a dataframe
    var_df = pd.DataFrame(list(averaged_expression_df.columns.values), columns=['var'])

    # create a dataframe for the selection of certain types
    mod_df = averaged_expression_df.copy()
    mod_df[list(obs_df.columns)] = obs_df.values
    if select_object == 'cell_type':
        selection = (mod_df.Cell_type == cell_type_to_look_at)
    elif select_object == 'subtype':
        selection = (mod_df.subtype == cell_type_to_look_at)

    mod_df = mod_df[selection]
    averaged_expression_df = averaged_expression_df[selection]

    # put the data into a compatible data structure
    ann_data = anndata.AnnData(X=averaged_expression_df.values, obs=mod_df[obs_df.columns],
                               var=var_df)

    ann_data.write_h5ad('../data/for_pyscan_{}_{}_{}.h5ad'.format(tissue_class, cell_type_to_look_at, scale))


# def get_regulated_genes(regulated_pathway, pathway_cms, pathway_dir, scale, tissue_class):
#     cms1_ranked_pos, cms2_ranked_pos, cms3_ranked_pos, cms4_ranked_pos = load_pathways_ranked(side='pos',
#                                                                                               cutoff_p='0025',
#                                                                                               libraries='hallmark',
#                                                                                               tissue_class=tissue_class,
#                                                                                               scale=scale)
#     cms1_ranked_neg, cms2_ranked_neg, cms3_ranked_neg, cms4_ranked_neg = load_pathways_ranked(side='neg',
#                                                                                               cutoff_p='0025',
#                                                                                               libraries='hallmark',
#                                                                                               tissue_class=tissue_class,
#                                                                                               scale=scale)

#     ranked_pos = [cms1_ranked_pos, cms2_ranked_pos, cms3_ranked_pos, cms4_ranked_pos]
#     ranked_neg = [cms1_ranked_neg, cms2_ranked_neg, cms3_ranked_neg, cms4_ranked_neg]
#     if pathway_dir == 'pos':
#         pathway_df = ranked_pos[pathway_cms - 1].copy()
#     elif pathway_dir == 'neg':
#         pathway_df = ranked_neg[pathway_cms - 1].copy()

#     pathways_to_be_up = [regulated_pathway]
#     pathway_row = pathway_df.loc[pathway_df['pathway'].isin(pathways_to_be_up)]

#     for gene_list in pathway_row['gene_union'].values:
#         regulated_genes = ast.literal_eval(gene_list)

#     return regulated_genes


# def load_adata_file(cell_type_to_look_at, tissue_class, scale):
#     adata = sc.read('../data/for_pyscan_{}_{}_{}.h5ad'.format(tissue_class, cell_type_to_look_at, scale))
#     adata.obs.cell_type = adata.obs.subtype.astype('category')
#     adata.var_names = adata.var.values.flatten()

#     return adata


# def load_full_adata_file():
#     adata = sc.read('../data/for_pyscan.h5ad')
#     adata.obs.cell_type = adata.obs.subtype.astype('category')
#     adata.var_names = adata.var.values.flatten()

#     return adata


# def save_diff_exp_genes(adata, p_cutoff, cell_type_to_look_at, tissue_class, scale, cut_off_graph=False):
#     ranked_genes = adata.uns['rank_genes_groups']['names']
#     array_of_genes = np.array(adata.uns['rank_genes_groups']['names'])
#     array_of_genes_pos = np.asarray(array_of_genes).copy()
#     array_of_genes_neg = np.asarray(array_of_genes).copy()

#     accompagnying_pvalues = np.array(adata.uns['rank_genes_groups']['pvals_adj'])
#     accompagnying_pvalues = np.asarray(accompagnying_pvalues).copy()
#     accompagnying_pvalues_pos = np.asarray(accompagnying_pvalues).copy()
#     accompagnying_pvalues_neg = np.asarray(accompagnying_pvalues).copy()

#     accompagnying_scores = np.array(adata.uns['rank_genes_groups']['scores'])
#     accompagnying_scores = np.asarray(accompagnying_scores)

#     # create a graph that shows how many genes are included for certain p-values
#     if cut_off_graph:
#         p_value_cutoffs = [0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.02, 0.03, 0.04,
#                            0.05,
#                            0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21,
#                            0.22, 0.23, 0.24, 0.25]
#         create_cutoff_graph(array_of_genes, accompagnying_pvalues, p_value_cutoffs, cell_type_to_look_at, tissue_class)

#     # select only the genes that are significantly different expressed
#     for i in range(len(array_of_genes_pos)):
#         for j in range(len(array_of_genes_pos[0])):
#             if accompagnying_pvalues[i][j] > p_cutoff or accompagnying_scores[i][j] < 0:
#                 array_of_genes_pos[i][j] = ''
#                 accompagnying_pvalues_pos[i][j] = 123211918239182
#             if accompagnying_pvalues[i][j] > p_cutoff or accompagnying_scores[i][j] >= 0:
#                 array_of_genes_neg[i][j] = ''
#                 accompagnying_pvalues_neg[i][j] = 123211918239182

#     diff_exp_genes_pos = {}
#     diff_exp_genes_pos['CMS1'] = [tup[0] for tup in array_of_genes_pos]
#     diff_exp_genes_pos['CMS2'] = [tup[1] for tup in array_of_genes_pos]
#     diff_exp_genes_pos['CMS3'] = [tup[2] for tup in array_of_genes_pos]
#     diff_exp_genes_pos['CMS4'] = [tup[3] for tup in array_of_genes_pos]

#     diff_exp_genes_pos_pvalue = {}
#     diff_exp_genes_pos_pvalue['CMS1'] = [tup[0] for tup in accompagnying_pvalues_pos]
#     diff_exp_genes_pos_pvalue['CMS2'] = [tup[1] for tup in accompagnying_pvalues_pos]
#     diff_exp_genes_pos_pvalue['CMS3'] = [tup[2] for tup in accompagnying_pvalues_pos]
#     diff_exp_genes_pos_pvalue['CMS4'] = [tup[3] for tup in accompagnying_pvalues_pos]

#     diff_exp_genes_pos['CMS1'] = list(filter(lambda x: x != '', diff_exp_genes_pos['CMS1']))
#     diff_exp_genes_pos['CMS2'] = list(filter(lambda x: x != '', diff_exp_genes_pos['CMS2']))
#     diff_exp_genes_pos['CMS3'] = list(filter(lambda x: x != '', diff_exp_genes_pos['CMS3']))
#     diff_exp_genes_pos['CMS4'] = list(filter(lambda x: x != '', diff_exp_genes_pos['CMS4']))

#     diff_exp_genes_pos_pvalue['CMS1'] = list(filter(lambda x: x != 123211918239182, diff_exp_genes_pos_pvalue['CMS1']))
#     diff_exp_genes_pos_pvalue['CMS2'] = list(filter(lambda x: x != 123211918239182, diff_exp_genes_pos_pvalue['CMS2']))
#     diff_exp_genes_pos_pvalue['CMS3'] = list(filter(lambda x: x != 123211918239182, diff_exp_genes_pos_pvalue['CMS3']))
#     diff_exp_genes_pos_pvalue['CMS4'] = list(filter(lambda x: x != 123211918239182, diff_exp_genes_pos_pvalue['CMS4']))

#     with open('../results/diff_exp_genes_sign_pos_{}_{}_{}_{}.txt'.format(str(p_cutoff).replace('.', ''),
#                                                                           cell_type_to_look_at, tissue_class,
#                                                                           scale),
#               'w') as file:
#         file.write(json.dumps(diff_exp_genes_pos))

#     with open('../results/diff_exp_genes_sign_pos_{}_{}_{}_{}_pvalues.txt'.format(str(p_cutoff).replace('.', ''),
#                                                                                   cell_type_to_look_at, tissue_class,
#                                                                                   scale),
#               'w') as file:
#         file.write(json.dumps(diff_exp_genes_pos_pvalue))

#     diff_exp_genes_neg = {}
#     diff_exp_genes_neg['CMS1'] = [tup[0] for tup in array_of_genes_neg]
#     diff_exp_genes_neg['CMS2'] = [tup[1] for tup in array_of_genes_neg]
#     diff_exp_genes_neg['CMS3'] = [tup[2] for tup in array_of_genes_neg]
#     diff_exp_genes_neg['CMS4'] = [tup[3] for tup in array_of_genes_neg]

#     diff_exp_genes_neg_pvalue = {}
#     diff_exp_genes_neg_pvalue['CMS1'] = [tup[0] for tup in accompagnying_pvalues_neg]
#     diff_exp_genes_neg_pvalue['CMS2'] = [tup[1] for tup in accompagnying_pvalues_neg]
#     diff_exp_genes_neg_pvalue['CMS3'] = [tup[2] for tup in accompagnying_pvalues_neg]
#     diff_exp_genes_neg_pvalue['CMS4'] = [tup[3] for tup in accompagnying_pvalues_neg]

#     diff_exp_genes_neg['CMS1'] = list(filter(lambda x: x != '', diff_exp_genes_neg['CMS1']))
#     diff_exp_genes_neg['CMS2'] = list(filter(lambda x: x != '', diff_exp_genes_neg['CMS2']))
#     diff_exp_genes_neg['CMS3'] = list(filter(lambda x: x != '', diff_exp_genes_neg['CMS3']))
#     diff_exp_genes_neg['CMS4'] = list(filter(lambda x: x != '', diff_exp_genes_neg['CMS4']))

#     diff_exp_genes_neg_pvalue['CMS1'] = list(filter(lambda x: x != 123211918239182, diff_exp_genes_neg_pvalue['CMS1']))
#     diff_exp_genes_neg_pvalue['CMS2'] = list(filter(lambda x: x != 123211918239182, diff_exp_genes_neg_pvalue['CMS2']))
#     diff_exp_genes_neg_pvalue['CMS3'] = list(filter(lambda x: x != 123211918239182, diff_exp_genes_neg_pvalue['CMS3']))
#     diff_exp_genes_neg_pvalue['CMS4'] = list(filter(lambda x: x != 123211918239182, diff_exp_genes_neg_pvalue['CMS4']))

#     with open('../results/diff_exp_genes_sign_neg_{}_{}_{}_{}.txt'.format(str(p_cutoff).replace('.', ''),
#                                                                           cell_type_to_look_at, tissue_class,
#                                                                           scale), 'w') as file:
#         file.write(json.dumps(diff_exp_genes_neg))

#     with open('../results/diff_exp_genes_sign_neg_{}_{}_{}_{}_pvalues.txt'.format(str(p_cutoff).replace('.', ''),
#                                                                                   cell_type_to_look_at, tissue_class,
#                                                                                   scale),
#               'w') as file:
#         file.write(json.dumps(diff_exp_genes_neg_pvalue))


# def get_genes_to_regulate(pathways_to_be_up, upregulated_pathways_sc, downregulated_pathways_sc):
#     """ Extracts the genes from a pathway df """

#     regulated_genes_cummulative = []

#     for direction, pathway_dfs in zip(['pos', 'neg'], [upregulated_pathways_sc, downregulated_pathways_sc]):
#         for idx, type in enumerate(['CMS1', 'CMS2', 'CMS3', 'CMS4']):
#             pathway_df = pathway_dfs[idx].copy()
#             pathway_row = pathway_df.loc[pathway_df['pathway'].isin(pathways_to_be_up[direction][type])]

#             for gene_list in pathway_row['gene_union'].values:
#                 regulated_genes = gene_list.replace(' ', ',')
#                 regulated_genes = ast.literal_eval(gene_list)
#                 regulated_genes_cummulative += [str(gene) for gene in regulated_genes]

#     return list(set(regulated_genes_cummulative))


# CMS_TYPES = ['CMS1', 'CMS2', 'CMS3', 'CMS4']
# CELL_TYPES = ['T cells', 'B cells', 'Epithelial cells', 'Myeloids', 'Stromal cells']
# BULK_CLASSIFICATION = {'SMC01': 'CMS3', 'SMC02': 'CMS4', 'SMC03': 'CMS1', 'SMC04': 'CMS4', 'SMC05': 'CMS3',
#                        'SMC06': 'CMS1', 'SMC07': 'CMS2', 'SMC08': 'CMS1', 'SMC09': 'CMS2', 'SMC10': 'CMS1',
#                        'SMC11': 'CMS2', 'SMC14': 'CMS4', 'SMC15': 'CMS1', 'SMC16': 'CMS3', 'SMC17': 'CMS4',
#                        'SMC18': 'CMS2', 'SMC19': 'CMS3', 'SMC20': 'CMS4', 'SMC21': 'CMS2', 'SMC22': 'CMS2',
#                        'SMC23': 'CMS2', 'SMC24': 'CMS4', 'SMC25': 'CMS2'}

# # ''' ------------------------------------------------ annotation data ----------------------------------------------- '''
# tissue_class = 'Tumor'
# scale = 'raw'
# indices_to_group, annotations_df = get_indices_to_group(data_type='GEO', tissue_class=tissue_class)

# # plot_sc_type_classification(annotations_df)
# # save_different_subtypes_per_patient(patient_df)


# ''' ------------------------------------------ expression data averaging ------------------------------------------ '''
# # create_average_sc_gene_expression_df(indices_to_group, scale='log', n=100)


# ''' ------------------------------------------ expression data selecting ------------------------------------------ '''
# # create_selection_sc_gene_expression_df(indices_to_group, scale=scale, n=100000, tissue_class=tissue_class)

# ''' -------------------------------------------- expression comparisons ------------------------------------------ '''
# p_cutoff = 0.025
# annotations_df.drop('Sample', inplace=True, axis=1)
# annotations_df.rename({'Index': 'sample'}, inplace=True, axis=1)

# means_total = None
# means_total_normal = None

# regulated_pathway = 'TNF-alpha Signaling via NF-kB'
# pathway_cms = 2
# pathway_dir = 'neg'

# regulated_genes_dict = {'pos': {'CMS1': [],
#                                 'CMS2': [],
#                                 'CMS3': [],
#                                 'CMS4': []},
#                         'neg': {'CMS1': [regulated_pathway],
#                                 'CMS2': [],
#                                 'CMS3': [],
#                                 'CMS4': []}
#                         }  # which pathways we want to downregulation when moving to the right of the graph

# # cms1_ranked_pos, cms2_ranked_pos, cms3_ranked_pos, cms4_ranked_pos = load_pathways_ranked(side='pos',
# #                                                                                           cutoff_p='0025',
# #                                                                                           libraries='hallmark',
# #                                                                                           scale=scale,
# #                                                                                           tissue_class=tissue_class)
# # cms1_ranked_neg, cms2_ranked_neg, cms3_ranked_neg, cms4_ranked_neg = load_pathways_ranked(side='neg',
# #                                                                                           cutoff_p='0025',
# #                                                                                           libraries='hallmark',
# #                                                                                           scale=scale,
# #                                                                                           tissue_class=tissue_class)
# # upregulated_pathways_sc = [cms1_ranked_pos, cms2_ranked_pos, cms3_ranked_pos, cms4_ranked_pos]
# # downregulated_pathways_sc = [cms1_ranked_neg, cms2_ranked_neg, cms3_ranked_neg, cms4_ranked_neg]
# # regulated_genes = get_genes_to_regulate(regulated_genes_dict, upregulated_pathways_sc,
# #                                         downregulated_pathways_sc)

# CELL_TYPES = ['T cells', 'B cells', 'Myeloids', 'Stromal cells', 'Epithelial cells']

# list_adatas = []
# for cell_type_to_look_at in CELL_TYPES:
#     # adata_full = load_full_adata_file()
#     adata = load_adata_file(cell_type_to_look_at, tissue_class, scale)
#     list_adatas.append(adata)

# full_data = list_adatas[0].concatenate(*list_adatas[1:])

# # print(adata_full)

# print(full_data[:, adata.var_names == 'A1BG'])

# marker_genes_dict = {
#     'B cell': ['IL2'],
#     'Myeloids': ['IL2'],
#     'Stromal cells': ['IL2'],
#     'Epithelial cells': ['IL2'],
#     'T cells': ['IL2']
# }

# sc.pp.normalize_total(full_data, target_sum=1e4)
# # sc.pp.log1p(full_data)

# # ax = sc.pl.heatmap(adata, marker_genes_dict, groupby='CMS_subtype', cmap='viridis', dendrogram=True)
# # ax1 = sc.pl.tracksplot(full_data, ['IL2'], groupby='batch', figsize=(5,4))

# markers = ['IL2',
#            'IL6',
#            'STAT3',
#            'IFNG',
#            'MYC',
#            'TNF',
#            'WNT1', 'WNT2', 'WNT3', 'WNT4', 'WNT5A', 'WNT6', 'WNT7A', 'WNT7B', 'WNT8A', 'WNT8B', 'WNT10B', 'WNT11',
#            'WNT2B', 'WNT9A', 'WNT9B', 'WNT10A', 'WNT16', 'WNT5B', 'WNT3A',
#            'FGF1', 'FGF2', 'FGF3', 'FGF4', 'FGF5',
#            'EGF',
#            'HGF',
#            'NOTCH1', 'NOTCH2', 'NOTCH3', 'NOTCH4',
#            'TGFB1', 'TGFB2', 'TGFB3',
#            'NFKB1', 'NFKB2','SPP1',
#            'TGFB1',
#            'MZF1',
#            'ITGB1',
#            'ITGB3',
#            'ITGB5',
#            'PRKACA',
#            'CD44']

# # creates a plot of the expression of certain genes per grouped group
# dp = sc.pl.dotplot(full_data, markers, groupby=['Cell_type'], figsize=(20, 5),
#                    return_fig=True,
#                    var_group_positions=[(6, 23), (24, 29), (32, 35), (36, 38), (39, 40)],
#                    var_group_labels=['WNT', 'FGF', 'NOTCH', 'TGFB', 'NFKB'],
#                    )
# dp.DEFAULT_PLOT_Y_PADDING = 2
# dp.style(cmap='Greens', y_padding=2)
# # dp.add_totals().style(dot_edge_color='black', dot_edge_lw=0.5).show()
# dp.swap_axes()
# axes_dict = dp.get_axes()
# axes_dict['mainplot_ax'].tick_params(labelsize=22)
# plt.rcParams.update({'axes.labelsize': 'large'})
# dp.savefig('dotplot_cell_types_comp_noscaling.pdf')
# plt.show()

# # # creates a correlation matrix of various genes
# # sc.tl.rank_genes_groups(full_data, groupby='CMS_subtype',
# #                                 method='t-test_overestim_var', rankby_abs=True)  # compute differential
# #
# # print(full_data)
# # print(full_data.uns['rank_genes_groups'])
# # correlation_matrix(full_data,
# #                    # name_list=['SPP1', 'TGFB1', 'TGFB2', 'TGFB3', 'MZF1', 'ITGB3', 'PRKACA', 'CD44'],
# #                    n_genes=20,
# #                    annotation_key=None,
# #                    method='pearson',
# #                    group='CMS4',
# #                    data='Group')
# # plot_correlation_matrix(full_data)

# # # creates a correlation matrix of various genes from a specific cell type
# cell_type = 'Myeloids'
# correlation_matrix(full_data, groupby='Cell_type', group=cell_type,
#                    name_list=['SPP1', 'TGFB1', 'TGFB2', 'TGFB3', 'MZF1'],
#                    annotation_key=None,
#                    method='pearson',
#                    data='Group')
# plot_correlation_matrix(full_data, groupby='Cell_type', group=cell_type)


# # # creates a secretion plot of a specific cell type
# # cell_type='Myeloids'
# # dp = sc.pl.dotplot(full_data[full_data.obs.Cell_type == cell_type], markers, groupby='CMS_subtype', figsize=(10, 3),
# #                    expression_cutoff=1, return_fig=True,
# #                    var_group_positions=[(6,23),(24, 29),(32,35),(36,38),(39,40)],
# #                    var_group_labels=['WNT', 'FGF', 'NOTCH', 'TGFB', 'NFKB']
# #                    )
# # dp.DEFAULT_PLOT_Y_PADDING = 2
# # dp.style(cmap='Greens', size_exponent=0.8)
# # dp.add_totals().style(dot_edge_color='black', dot_edge_lw=0.5).show()
# # dp.savefig('dotplot_CMS_{}_noscaling.pdf'.format(cell_type))
# # axes_dict = dp.get_axes()
# # # plt.tight_layout(pad=2)
# # plt.show()


# # cell_type='Stromal cells'
# # dp = sc.pl.matrixplot(full_data[full_data.obs.Cell_type == cell_type], markers, groupby='CMS_subtype', figsize=(10, 3),
# #                    return_fig=True,
# #                    var_group_positions=[(5,23),(24, 29),(32,35),(36,38),(39,40)],
# #                    var_group_labels=['WNT', 'FGF', 'NOTCH', 'TGFB', 'NFKB']
# #                    )
# # dp.DEFAULT_PLOT_Y_PADDING = 2
# # # dp.style(cmap='Greens', size_exponent=0.8)
# # # dp.add_totals().style(dot_edge_color='black', dot_edge_lw=0.5).show()
# # # dp.savefig('dotplot_CMS_{}_noscaling.pdf'.format(cell_type))
# # axes_dict = dp.get_axes()
# # # plt.tight_layout(pad=2)
# # plt.show()

# # means_normal, std_normal = grouped_obs_mean(adata, 'CMS_subtype', layer=None, gene_symbols=regulated_genes)
