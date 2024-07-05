# import the libraries

import math
import time
import random
import anndata
import ast
import json

# import modin.pandas as pd
import pandas as pd
import seaborn as sns
import numpy as np
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 7)


# from dask import dataframe as pdd
import scanpy as sc

import matplotlib.pyplot as plt

from collections import defaultdict, Mapping
from collections.abc import Mapping


# define funcitons
def Tree():
    return defaultdict(Tree)

def load_pathways_ranked(side, cutoff_p, libraries, scale, tissue_class):
    ''' Load the ranked pathway dfs '''

    cms1_ranked = \
        pd.read_csv('../results/pathway_comparison/pathway_ranking_cms1_sign_{}_{}_{}_{}_{}.csv'.format(side,
                                                                                                         cutoff_p,
                                                                                                         libraries,
                                        tissue_class, scale))
    cms2_ranked = \
        pd.read_csv('../results/pathway_comparison/pathway_ranking_cms2_sign_{}_{}_{}_{}_{}.csv'.format(side,
                                                                                                         cutoff_p,
                                                                                                         libraries,
                                        tissue_class, scale))
    cms3_ranked = \
        pd.read_csv('../results/pathway_comparison/pathway_ranking_cms3_sign_{}_{}_{}_{}_{}.csv'.format(side,
                                                                                                         cutoff_p,
                                                                                                         libraries,
                                        tissue_class, scale))
    cms4_ranked = \
        pd.read_csv('../results/pathway_comparison/pathway_ranking_cms4_sign_{}_{}_{}_{}_{}.csv'.format(side,
                                                                                                         cutoff_p,
                                                                                                         libraries,
                                        tissue_class, scale))

    return cms1_ranked, cms2_ranked, cms3_ranked, cms4_ranked


def create_cutoff_graph(array_of_genes, accompagnying_pvalues, p_cutoffs, cell_type, tissue_class):

    lengths_at_pvalue_cms1 = []
    lengths_at_pvalue_cms2 = []
    lengths_at_pvalue_cms3 = []
    lengths_at_pvalue_cms4 = []

    for p_cutoff in p_cutoffs:
        # select only the genes that are significantly different expressed
        current_gene_array = array_of_genes.copy()

        for i in range(len(current_gene_array)):
            for j in range(len(current_gene_array[0])):
                if accompagnying_pvalues[i][j] > p_cutoff:
                    current_gene_array[i][j] = ''
                else:
                    continue

        diff_exp_genes = {}
        diff_exp_genes['CMS1'] = [tup[0] for tup in current_gene_array]
        diff_exp_genes['CMS2'] = [tup[1] for tup in current_gene_array]
        diff_exp_genes['CMS3'] = [tup[2] for tup in current_gene_array]
        diff_exp_genes['CMS4'] = [tup[3] for tup in current_gene_array]

        lengths_at_pvalue_cms1.append(len(list(filter(lambda x: x != '', diff_exp_genes['CMS1']))))
        lengths_at_pvalue_cms2.append(len(list(filter(lambda x: x != '', diff_exp_genes['CMS2']))))
        lengths_at_pvalue_cms3.append(len(list(filter(lambda x: x != '', diff_exp_genes['CMS3']))))
        lengths_at_pvalue_cms4.append(len(list(filter(lambda x: x != '', diff_exp_genes['CMS4']))))

    plt.figure()
    plt.plot(p_cutoffs, lengths_at_pvalue_cms1, label='CMS1')
    plt.plot(p_cutoffs, lengths_at_pvalue_cms2, label='CMS2')
    plt.plot(p_cutoffs, lengths_at_pvalue_cms3, label='CMS3')
    plt.plot(p_cutoffs, lengths_at_pvalue_cms4, label='CMS4')
    plt.ylim(0, max(max(lengths_at_pvalue_cms1), max(lengths_at_pvalue_cms2), max(lengths_at_pvalue_cms3),
                    max(lengths_at_pvalue_cms4)))
    plt.xlabel('cut-off p-value', fontsize=14)
    plt.ylabel('# of significant genes', fontsize=14)
    # plt.xscale('log')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('../results/pvalue_cutoff_{}_{}.pdf'.format(cell_type, tissue_class))
    plt.show()

def grouped_obs_mean(adata, group_key, layer=None, gene_symbols=None):
    if layer is not None:
        getX = lambda x: x.layers[layer]
    else:
        getX = lambda x: x.X
    if gene_symbols is not None:
        new_idx = gene_symbols
    else:
        new_idx = adata.var_names

    grouped = adata.obs.groupby(group_key)
    out = pd.DataFrame(
        np.zeros((len(new_idx), len(grouped)), dtype=np.float64),
        columns=list(grouped.groups.keys()),
        index=new_idx
    )

    out_std = pd.DataFrame(
        np.zeros((len(new_idx), len(grouped)), dtype=np.float64),
        columns=list(grouped.groups.keys()),
        index=new_idx
    )

    for group, idx in grouped.indices.items():
        X = getX(adata[idx, adata.var_names.isin(new_idx)])
        out[group] = np.ravel(X.mean(axis=0, dtype=np.float64))
        out_std[group] = np.ravel(X.std(axis=0, dtype=np.float64))

    return out, out_std


def get_indices_to_group(data_type, tissue_class='Tumor'):
    annotations_df = pd.read_csv('../data/GSE132465_GEO_processed_CRC_10X_cell_annotation.txt', delimiter='\t')
    patients = annotations_df.sort_values(by=['Patient'])['Patient'].unique()


    annotations_df['bulk_classification'] = annotations_df['Patient'].map(BULK_CLASSIFICATION)

    assert tissue_class in ['Tumor', 'Normal', 'All'], 'sample type not recognized'

    if tissue_class != 'All':
        indices_to_group = annotations_df.loc[annotations_df.Class == tissue_class].groupby(['Patient', 'Cell_subtype'])[
            'Index'].apply(list)
    elif tissue_class == 'All':
        indices_to_group = annotations_df.groupby(['Patient', 'Cell_subtype'])[
            'Index'].apply(list)

    return indices_to_group, annotations_df


def plot_sc_type_classification(annotations_df):
    # check the type distribution
    nrows = 4
    ncols = 6
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    idx = 0
    for tup, patient_df in annotations_df[(annotations_df.Cell_type == 'Epithelial cells') &
                                          (annotations_df.Class == 'Tumor')].groupby(
        by=['Patient', 'Class', 'Sample', 'Cell_type']):
        row = math.floor(idx / ncols)
        col = math.floor(idx - row * ncols)
        counts = patient_df.groupby(['Cell_subtype'])['Cell_subtype'].count()
        colors = []
        for CMS in ['CMS1', 'CMS2', 'CMS3', 'CMS4']:
            if CMS not in counts.index:
                counts[CMS] = 0

            colors.append('blue' if not patient_df['bulk_classification'].values[0] == CMS else 'red')

        print(counts, patient_df['bulk_classification'].values[0])
        counts.sort_index(inplace=True)
        # print(patient_df.groupby(['Cell_subtype'])['Cell_subtype'].count())
        print(colors)
        counts.plot(ax=axes[row, col], kind='bar', color=colors)
        axes[row, col].axes.get_xaxis().set_visible(False)
        axes[row, col].set_title(tup[0])
        idx += 1

    plt.axis(False)
    # Set common labels
    fig.text(0.5, 0.02, 'CMS1, CMS2, CMS3, CMS4', ha='center', va='center')
    fig.text(0.02, 0.5, 'Counts', ha='center', va='center', rotation='vertical')
    plt.tight_layout()
    plt.savefig('../figures/classification_ind_cells.pdf')
    plt.show()


def save_different_subtypes_per_patient(patient_df):

    # check the types of cells that are present in the dataset
    tree = Tree()
    for tup, patient_df in annotations_df.groupby(by=['Patient', 'Class', 'Cell_type']):
        print('In the following, we get: {}'.format(tup))
        print('The subtypes in this sample are: {}'.format(patient_df['Cell_subtype'].unique()))
        tree[tup[0]][tup[1]][tup[2]] = patient_df['Cell_subtype'].unique()

    df_subtypes = pd.DataFrame.from_dict({(i,j): tree[i][j]
                               for i in tree.keys()
                               for j in tree[i].keys()},
                           orient='index')

    df_subtypes.to_csv('../figures/types_of_cells.csv')


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
            averaged_df[list(zip([str(patient)]*n, [str(subtype)]*n, list(newer_df.values[0])))] = \
                raw_counts_df[list(newer_df.values)[0]]
            averaged_df[list(zip([str(patient)]*n, [str(subtype)]*n, list(newer_df.values[0])))] = \
                raw_counts_df[list(newer_df.values)[0]]

    averaged_df.columns = pd.MultiIndex.from_tuples(list(averaged_df.columns.values),
                                                    names=['patient', 'subtype', 'samples'])

    print('This much time has elapsed: ', time.time() - start)

    averaged_df.to_csv('../data/selected_sc_gene_express_{}_{}.csv'.format(tissue_class, scale))



## NEW 2, works to write file
def save_expression_data_as_format_for_scanpy(cell_type_to_look_at, select_object, tissue_class, scale, sample_size=1000):
    print('doing calculations for {}'.format(cell_type_to_look_at))
    
    # Read the CSV file in chunks
    chunks = pd.read_csv(
        '../data/selected_sc_gene_express_{}_{}.csv'.format(tissue_class, scale),
        index_col=0,
        header=[0, 1, 2],
        chunksize=1100  # Adjust this value based on your memory constraints
    )
    
    # Process the first chunk to get the desired columns
    first_chunk = next(chunks)
    if sample_size < first_chunk.shape[1]:
        averaged_expression_df = first_chunk.iloc[:, :sample_size]
    else:
        averaged_expression_df = first_chunk
    
    print("Initial shape:", averaged_expression_df.shape)
    
    # If we need more columns, keep reading chunks
    while averaged_expression_df.shape[1] < sample_size:
        chunk = next(chunks)
        columns_needed = sample_size - averaged_expression_df.shape[1]
        averaged_expression_df = pd.concat([averaged_expression_df, chunk.iloc[:, :columns_needed]], axis=1)
    
    print("After reading chunks:", averaged_expression_df.shape)
    
    # Create obs_df
    obs_df = pd.DataFrame(averaged_expression_df.columns.to_frame().reset_index(drop=True))
    obs_df.columns = ['patient', 'subtype', 'sample']
    obs_df['CMS_subtype'] = obs_df['patient'].map(BULK_CLASSIFICATION)
    obs_df = pd.merge(obs_df, annotations_df[['sample', 'Cell_type']], on='sample')
    
    print("obs_df shape:", obs_df.shape)
    
    # Create var_df
    var_df = pd.DataFrame(index=averaged_expression_df.index).reset_index()
    var_df.columns = ['var']
    
    # Transpose averaged_expression_df and set index to match obs_df
    averaged_expression_df = averaged_expression_df.T
    averaged_expression_df.index = obs_df.index
    
    print("Transposed averaged_expression_df shape:", averaged_expression_df.shape)
    
    # Select specific cell type or subtype if required
    if select_object == 'cell_type':
        selection = (obs_df.Cell_type == cell_type_to_look_at)
    elif select_object == 'subtype':
        selection = (obs_df.subtype == cell_type_to_look_at)
    
    if select_object in ['cell_type', 'subtype']:
        averaged_expression_df = averaged_expression_df.loc[selection]
        obs_df = obs_df.loc[selection]
    
    print("After selection - averaged_expression_df shape:", averaged_expression_df.shape)
    print("After selection - obs_df shape:", obs_df.shape)
    
    # Create AnnData object
    ann_data = anndata.AnnData(X=averaged_expression_df.values, 
                               obs=obs_df,
                               var=var_df)
    
    # Write AnnData to file
    ann_data.write_h5ad('../data/for_pyscan_{}_{}_{}.h5ad'.format(tissue_class, cell_type_to_look_at, scale))

    print('wrote the file')


# ### NEWEST, FOR HPC ENVIRONMENT
# def save_expression_data_as_format_for_scanpy(cell_type_to_look_at, select_object, tissue_class, scale):
#     print('doing calculations for {}'.format(cell_type_to_look_at))
    
#     # Read the entire CSV file
#     averaged_expression_df = pd.read_csv('../data/selected_sc_gene_express_{}_{}.csv'.format(tissue_class, scale),
#                                          index_col=0,
#                                          header=[0, 1, 2])
    
#     print("Initial shape:", averaged_expression_df.shape)
    
#     # Create obs_df
#     obs_df = pd.DataFrame(averaged_expression_df.columns.to_frame().reset_index(drop=True))
#     obs_df.columns = ['patient', 'subtype', 'sample']
#     obs_df['CMS_subtype'] = obs_df['patient'].map(BULK_CLASSIFICATION)
#     obs_df = pd.merge(obs_df, annotations_df[['sample', 'Cell_type']], on='sample')
    
#     print("obs_df shape:", obs_df.shape)
    
#     # Create var_df
#     var_df = pd.DataFrame(index=averaged_expression_df.index).reset_index()
#     var_df.columns = ['var']

#     # Transpose averaged_expression_df and set index to match obs_df
#     averaged_expression_df = averaged_expression_df.T
#     averaged_expression_df.index = obs_df.index

#     print("Transposed averaged_expression_df shape:", averaged_expression_df.shape)
    
#     # Select specific cell type or subtype if required
#     if select_object == 'cell_type':
#         selection = (obs_df.Cell_type == cell_type_to_look_at)
#     elif select_object == 'subtype':
#         selection = (obs_df.subtype == cell_type_to_look_at)
    
#     if select_object in ['cell_type', 'subtype']:
#         averaged_expression_df = averaged_expression_df.loc[selection]
#         obs_df = obs_df.loc[selection]
    
#     print("After selection - averaged_expression_df shape:", averaged_expression_df.shape)
#     print("After selection - obs_df shape:", obs_df.shape)
    
#     # Create AnnData object
#     ann_data = anndata.AnnData(X=averaged_expression_df.values, 
#                                obs=obs_df,
#                                var=var_df)
    
#     # Write AnnData to file
#     ann_data.write_h5ad('../data/for_pyscan_{}_{}_{}.h5ad'.format(tissue_class, cell_type_to_look_at, scale))

#     print('wrote the file')


def get_regulated_genes(regulated_pathway, pathway_cms, pathway_dir, scale, tissue_class):
    cms1_ranked_pos, cms2_ranked_pos, cms3_ranked_pos, cms4_ranked_pos = load_pathways_ranked(side='pos',
                                                                                              cutoff_p='0025',
                                                                                              libraries='hallmark',
                                        tissue_class=tissue_class, scale=scale)
    cms1_ranked_neg, cms2_ranked_neg, cms3_ranked_neg, cms4_ranked_neg = load_pathways_ranked(side='neg',
                                                                                             cutoff_p='0025',
                                                                                             libraries='hallmark',
                                        tissue_class=tissue_class, scale=scale)

    ranked_pos = [cms1_ranked_pos, cms2_ranked_pos, cms3_ranked_pos, cms4_ranked_pos]
    ranked_neg = [cms1_ranked_neg, cms2_ranked_neg, cms3_ranked_neg, cms4_ranked_neg]
    if pathway_dir == 'pos':
        pathway_df = ranked_pos[pathway_cms - 1].copy()
    elif pathway_dir == 'neg':
        pathway_df = ranked_neg[pathway_cms - 1].copy()

    pathways_to_be_up = [regulated_pathway]
    pathway_row = pathway_df.loc[pathway_df['pathway'].isin(pathways_to_be_up)]

    for gene_list in pathway_row['gene_union'].values:
        regulated_genes = ast.literal_eval(gene_list)

    return regulated_genes

def load_adata_file(cell_type_to_look_at, tissue_class, scale):
    adata = sc.read('../data/for_pyscan_{}_{}_{}.h5ad'.format(tissue_class, cell_type_to_look_at, scale))
    adata.obs.cell_type = adata.obs.subtype.astype('category')
    adata.var_names = adata.var.values.flatten()

    return adata


def save_diff_exp_genes(adata, p_cutoff, cell_type_to_look_at, tissue_class, scale, cut_off_graph=False):

    ranked_genes = adata.uns['rank_genes_groups']['names']
    array_of_genes = np.array(adata.uns['rank_genes_groups']['names'])
    array_of_genes_pos = np.asarray(array_of_genes).copy()
    array_of_genes_neg = np.asarray(array_of_genes).copy()

    accompagnying_pvalues = np.array(adata.uns['rank_genes_groups']['pvals_adj'])
    accompagnying_pvalues = np.asarray(accompagnying_pvalues).copy()
    accompagnying_pvalues_pos = np.asarray(accompagnying_pvalues).copy()
    accompagnying_pvalues_neg = np.asarray(accompagnying_pvalues).copy()

    accompagnying_scores = np.array(adata.uns['rank_genes_groups']['scores'])
    accompagnying_scores = np.asarray(accompagnying_scores)


    # create a graph that shows how many genes are included for certain p-values
    if cut_off_graph:
        p_value_cutoffs = [0.000000001,0.00000001,0.0000001,0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.02, 0.03, 0.04, 0.05,
                           0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21,
                           0.22, 0.23, 0.24, 0.25]
        create_cutoff_graph(array_of_genes, accompagnying_pvalues, p_value_cutoffs, cell_type_to_look_at, tissue_class)


    # select only the genes that are significantly different expressed
    for i in range(len(array_of_genes_pos)):
        for j in range(len(array_of_genes_pos[0])):
            if accompagnying_pvalues[i][j] > p_cutoff or accompagnying_scores[i][j] < 0:
                array_of_genes_pos[i][j] = ''
                accompagnying_pvalues_pos[i][j] = 123211918239182
            if accompagnying_pvalues[i][j] > p_cutoff or accompagnying_scores[i][j] >= 0:
                array_of_genes_neg[i][j] = ''
                accompagnying_pvalues_neg[i][j] = 123211918239182

    diff_exp_genes_pos = {}
    diff_exp_genes_pos['CMS1'] = [tup[0] for tup in array_of_genes_pos]
    diff_exp_genes_pos['CMS2'] = [tup[1] for tup in array_of_genes_pos]
    diff_exp_genes_pos['CMS3'] = [tup[2] for tup in array_of_genes_pos]
    diff_exp_genes_pos['CMS4'] = [tup[3] for tup in array_of_genes_pos]

    diff_exp_genes_pos_pvalue = {}
    diff_exp_genes_pos_pvalue['CMS1'] = [tup[0] for tup in accompagnying_pvalues_pos]
    diff_exp_genes_pos_pvalue['CMS2'] = [tup[1] for tup in accompagnying_pvalues_pos]
    diff_exp_genes_pos_pvalue['CMS3'] = [tup[2] for tup in accompagnying_pvalues_pos]
    diff_exp_genes_pos_pvalue['CMS4'] = [tup[3] for tup in accompagnying_pvalues_pos]

    diff_exp_genes_pos['CMS1'] = list(filter(lambda x: x != '', diff_exp_genes_pos['CMS1']))
    diff_exp_genes_pos['CMS2'] = list(filter(lambda x: x != '', diff_exp_genes_pos['CMS2']))
    diff_exp_genes_pos['CMS3'] = list(filter(lambda x: x != '', diff_exp_genes_pos['CMS3']))
    diff_exp_genes_pos['CMS4'] = list(filter(lambda x: x != '', diff_exp_genes_pos['CMS4']))

    diff_exp_genes_pos_pvalue['CMS1'] = list(filter(lambda x: x != 123211918239182, diff_exp_genes_pos_pvalue['CMS1']))
    diff_exp_genes_pos_pvalue['CMS2'] = list(filter(lambda x: x != 123211918239182, diff_exp_genes_pos_pvalue['CMS2']))
    diff_exp_genes_pos_pvalue['CMS3'] = list(filter(lambda x: x != 123211918239182, diff_exp_genes_pos_pvalue['CMS3']))
    diff_exp_genes_pos_pvalue['CMS4'] = list(filter(lambda x: x != 123211918239182, diff_exp_genes_pos_pvalue['CMS4']))

    with open('../results/diff_exp_genes_sign_pos_{}_{}_{}_{}.txt'.format(str(p_cutoff).replace('.', ''),
                                                                     cell_type_to_look_at, tissue_class,
                                                                          scale),
              'w') as file:
         file.write(json.dumps(diff_exp_genes_pos))

    with open('../results/diff_exp_genes_sign_pos_{}_{}_{}_{}_pvalues.txt'.format(str(p_cutoff).replace('.', ''),
                                                                     cell_type_to_look_at, tissue_class,
                                                                          scale),
              'w') as file:
         file.write(json.dumps(diff_exp_genes_pos_pvalue))


    diff_exp_genes_neg = {}
    diff_exp_genes_neg['CMS1'] = [tup[0] for tup in array_of_genes_neg]
    diff_exp_genes_neg['CMS2'] = [tup[1] for tup in array_of_genes_neg]
    diff_exp_genes_neg['CMS3'] = [tup[2] for tup in array_of_genes_neg]
    diff_exp_genes_neg['CMS4'] = [tup[3] for tup in array_of_genes_neg]

    diff_exp_genes_neg_pvalue = {}
    diff_exp_genes_neg_pvalue['CMS1'] = [tup[0] for tup in accompagnying_pvalues_neg]
    diff_exp_genes_neg_pvalue['CMS2'] = [tup[1] for tup in accompagnying_pvalues_neg]
    diff_exp_genes_neg_pvalue['CMS3'] = [tup[2] for tup in accompagnying_pvalues_neg]
    diff_exp_genes_neg_pvalue['CMS4'] = [tup[3] for tup in accompagnying_pvalues_neg]

    diff_exp_genes_neg['CMS1'] = list(filter(lambda x: x != '', diff_exp_genes_neg['CMS1']))
    diff_exp_genes_neg['CMS2'] = list(filter(lambda x: x != '', diff_exp_genes_neg['CMS2']))
    diff_exp_genes_neg['CMS3'] = list(filter(lambda x: x != '', diff_exp_genes_neg['CMS3']))
    diff_exp_genes_neg['CMS4'] = list(filter(lambda x: x != '', diff_exp_genes_neg['CMS4']))

    diff_exp_genes_neg_pvalue['CMS1'] = list(filter(lambda x: x != 123211918239182, diff_exp_genes_neg_pvalue['CMS1']))
    diff_exp_genes_neg_pvalue['CMS2'] = list(filter(lambda x: x != 123211918239182, diff_exp_genes_neg_pvalue['CMS2']))
    diff_exp_genes_neg_pvalue['CMS3'] = list(filter(lambda x: x != 123211918239182, diff_exp_genes_neg_pvalue['CMS3']))
    diff_exp_genes_neg_pvalue['CMS4'] = list(filter(lambda x: x != 123211918239182, diff_exp_genes_neg_pvalue['CMS4']))

    with open('../results/diff_exp_genes_sign_neg_{}_{}_{}_{}.txt'.format(str(p_cutoff).replace('.', ''),
                                                                     cell_type_to_look_at, tissue_class,
                                                                          scale), 'w') as file:
         file.write(json.dumps(diff_exp_genes_neg))

    with open('../results/diff_exp_genes_sign_neg_{}_{}_{}_{}_pvalues.txt'.format(str(p_cutoff).replace('.', ''),
                                                                     cell_type_to_look_at, tissue_class,
                                                                          scale),
              'w') as file:
         file.write(json.dumps(diff_exp_genes_neg_pvalue))

def get_genes_to_regulate(pathways_to_be_up, upregulated_pathways_sc, downregulated_pathways_sc):
    """ Extracts the genes from a pathway df """

    regulated_genes_cummulative = []

    for direction, pathway_dfs in zip(['pos', 'neg'], [upregulated_pathways_sc, downregulated_pathways_sc]):
        for idx, type in enumerate(['CMS1', 'CMS2', 'CMS3', 'CMS4']):
            pathway_df = pathway_dfs[idx].copy()
            pathway_row = pathway_df.loc[pathway_df['pathway'].isin(pathways_to_be_up[direction][type])]

            for gene_list in pathway_row['gene_union'].values:
                regulated_genes = gene_list.replace(' ', ',')
                regulated_genes = ast.literal_eval(gene_list)
                regulated_genes_cummulative += [str(gene) for gene in regulated_genes]

    return list(set(regulated_genes_cummulative))

CMS_TYPES = ['CMS1', 'CMS2', 'CMS3', 'CMS4']
CELL_TYPES = ['T cells', 'B cells', 'Epithelial cells', 'Myeloids', 'Stromal cells']
BULK_CLASSIFICATION = {'SMC01': 'CMS3', 'SMC02': 'CMS4', 'SMC03': 'CMS1', 'SMC04': 'CMS4', 'SMC05': 'CMS3',
                       'SMC06': 'CMS1', 'SMC07': 'CMS2', 'SMC08': 'CMS1', 'SMC09': 'CMS2', 'SMC10': 'CMS1',
                       'SMC11': 'CMS2', 'SMC14': 'CMS4', 'SMC15': 'CMS1', 'SMC16': 'CMS3', 'SMC17': 'CMS4',
                       'SMC18': 'CMS2', 'SMC19': 'CMS3', 'SMC20': 'CMS4', 'SMC21': 'CMS2', 'SMC22': 'CMS2',
                       'SMC23': 'CMS2', 'SMC24': 'CMS4', 'SMC25': 'CMS2'}

# ''' ------------------------------------------------ annotation data ----------------------------------------------- '''
tissue_class = 'Tumor'
scale = 'raw'
indices_to_group, annotations_df = get_indices_to_group(data_type='GEO', tissue_class=tissue_class)

# ADDITION
indices_to_group_tumor, annotations_df_tumor = get_indices_to_group(data_type='GEO', tissue_class='Tumor')
indices_to_group_normal, annotations_df_normal = get_indices_to_group(data_type='GEO', tissue_class='Normal')

annotations_df = annotations_df_tumor


## UNCOMMENT TO GET THE PLOTS BACK
# plot_sc_type_classification(annotations_df)
# print(annotations_df)
# save_different_subtypes_per_patient(patient_df)


''' ------------------------------------------ expression data averaging ------------------------------------------ '''
# create_average_sc_gene_expression_df(indices_to_group, scale='log', n=100)


''' ------------------------------------------ expression data selecting ------------------------------------------ '''
# create_selection_sc_gene_expression_df(indices_to_group, scale=scale, n=100000, tissue_class=tissue_class)

# ADDITION
create_selection_sc_gene_expression_df(indices_to_group_tumor, scale=scale, n=100000, tissue_class='Tumor')
create_selection_sc_gene_expression_df(indices_to_group_normal, scale=scale, n=100000, tissue_class='Normal')

''' -------------------------------------------- expression comparisons ------------------------------------------ '''
p_cutoff = 0.025
annotations_df.drop('Sample', inplace=True, axis=1)
annotations_df.rename({'Index': 'sample'}, inplace=True, axis=1)

means_total = None
means_total_normal = None

regulated_pathway = 'TNF-alpha Signaling via NF-kB'
pathway_cms = 2
pathway_dir = 'neg'

regulated_genes_dict = {'pos': {'CMS1': [],
                           'CMS2': [],
                           'CMS3': [],
                           'CMS4': []},
                   'neg': {'CMS1': [regulated_pathway],
                           'CMS2': [],
                           'CMS3': [],
                           'CMS4': []}
                   }  # which pathways we want to downregulation when moving to the right of the graph

# cms1_ranked_pos, cms2_ranked_pos, cms3_ranked_pos, cms4_ranked_pos = load_pathways_ranked(side='pos',
#                                                                                           cutoff_p='0025',
#                                                                                           libraries='hallmark',
#                                                                                           scale=scale,
#                                                                                           tissue_class=tissue_class)
# cms1_ranked_neg, cms2_ranked_neg, cms3_ranked_neg, cms4_ranked_neg = load_pathways_ranked(side='neg',
#                                                                                           cutoff_p='0025',
#                                                                                           libraries='hallmark',
#                                                                                           scale=scale,
#                                                                                           tissue_class=tissue_class)
# upregulated_pathways_sc = [cms1_ranked_pos, cms2_ranked_pos, cms3_ranked_pos, cms4_ranked_pos]
# downregulated_pathways_sc = [cms1_ranked_neg, cms2_ranked_neg, cms3_ranked_neg, cms4_ranked_neg]
# regulated_genes = get_genes_to_regulate(regulated_genes_dict, upregulated_pathways_sc,
#                                                         downregulated_pathways_sc)

CELL_TYPES = ['T cells', 'B cells', 'Myeloids', 'Stromal cells', 'Epithelial cells']

for cell_type_to_look_at in CELL_TYPES:
    print('Currently looking at ', cell_type_to_look_at)
    select_object = 'cell_type'  # two options: 'cell_type' or 'subtype'
    save_expression_data_as_format_for_scanpy(cell_type_to_look_at, select_object, 'Tumor', scale)
    save_expression_data_as_format_for_scanpy(cell_type_to_look_at, select_object, 'Normal', scale)
    #
    # regulated_genes = get_regulated_genes(regulated_pathway=regulated_pathway, pathway_cms=pathway_cms, pathway_dir=pathway_dir,
    #                                       scale=scale, tissue_class=tissue_class)

    # load the h5ad file
    adata = load_adata_file(cell_type_to_look_at, tissue_class, scale)
    print(adata)
    adata_normal = load_adata_file(cell_type_to_look_at, tissue_class='Normal', scale=scale)


    print("patients normal data: ", adata_normal.obs.patient.unique())

    adata = adata[adata.obs.patient.isin(list(adata_normal.obs.patient.unique())), :]
    print("patients cancerous data: ", adata.obs.patient.unique())


    # get the differentially expressed genes between the CMS subtypes
    try:
        sc.tl.rank_genes_groups(adata, groupby='CMS_subtype', use_raw=True,
                                method='t-test_overestim_var', rankby_abs=True)  # compute differential
        save_diff_exp_genes(adata, p_cutoff, cell_type_to_look_at, tissue_class, scale, cut_off_graph=True)

    except Exception as e:
        print('was not able to rank the gene because of following error \n', e)

    # sc.pl.dotplot(adata, regulated_genes, groupby='CMS_subtype', swap_axes=True, figsize=(10,15))
    # sc.pl.heatmap(adata, regulated_genes, groupby='CMS_subtype', swap_axes=True, figsize=(10,15))
    # sc.pl.tracksplot(adata, regulated_genes, groupby='CMS_subtype', swap_axes=True, figsize=(10, 15))

    # sc.pl.rank_genes_groups_tracksplot(adata, groupby='CMS_subtype', n_genes=20,
    #                                    save='diff_expr_{}.pdf'.format(cell_type_to_look_at.replace(" ", "") + str(
    #                                        time.strftime("%Y%m%d-%H%M%S"))),
    #                                    title='{}'.format(cell_type_to_look_at))  # plot the result

    # plt.figure(figsize=(10,10))
    # sc.tl.dendrogram(adata, groupby='CMS_subtype')
    #
    # sc.tl.correlation_matrix(adata, ['SPP1', 'TGFB1'], n_genes=20, annotation_key=None, method='pearson')
    sc.pl.correlation_matrix(adata, groupby='CMS_subtype', figsize=(10,10))
    markers = ['SPP1', 'TGFB1', 'TGFB1I1', 'TGFB2', 'TGFB2-AS1', 'TGFB3', 'TGFBI', 'TGFBR1',
               'TGFBR2', 'TGFBR3', 'TGFBR3L', 'TGFBRAP1']
    sc.pl.dotplot(adata, markers, groupby='CMS_subtype', dendrogram=True)
    sc.pl.clustermap(adata, obs_keys='CMS_subtype')

    # sc.pl.violin(adata, regulated_genes, groupby='CMS_subtype', multi_panel=True)

    # sc.pp.neighbors(adata)
    # sc.tl.leiden(adata)
    # sc.tl.paga(adata, neighbors_key ='neighbors')
    # sc.pl.paga(adata)
    #
    # print(adata)
    # sc.pl.correlation_matrix(adata, 'CMS_subtype', figsize=(5, 3.5))


    # selected_genes_df = adata[:, adata.var_names.isin(regulated_genes)]
    # print('selected genes df: ', selected_genes_df)
    # averaged_gene_expression = selected_genes_df.groupby('CMS_subtype')
    #
    # for group_name, group in averaged_gene_expression:
    #     print(group_name, group.keys())
    #     print('this is the expression per gene then: \n', group.X)
    #
    # print(averaged_gene_expression)

    # # create means for making heat maps to show differential expression
    # means_, std_ = grouped_obs_mean(adata, 'CMS_subtype', layer=None, gene_symbols=regulated_genes)
    #
    # if means_total is None:
    #     means_total = means_
    # else:
    #     means_total = means_total + means_
    #
    # means_normal, std_normal = grouped_obs_mean(adata_normal, 'CMS_subtype', layer=None, gene_symbols=regulated_genes)
    #
    # if means_total_normal is None:
    #     means_total_normal = means_normal
    # else:
    #     means_total_normal = means_total_normal + means_normal



# means_total_TN_percent = means_total.sub(means_total_normal.mean(axis=1), axis=0) / means_total_normal * 100
# plt.figure(figsize=(9,16))
# plt.title('Percentage diff expression relative to normal tissue \n of {} genes'.format(regulated_pathway))
# sns.heatmap(means_total_TN_percent, annot=True, cmap='seismic', center=0, vmin=-150, vmax=150)
# plt.show()
#
# means_total_percent = means_total.sub(means_total.mean(axis=1), axis=0) / means_total * 100
# plt.figure(figsize=(9,16))
# plt.title('Percentage diff expression relative to tumor tissue \n of {} genes'.format(regulated_pathway))
# sns.heatmap(means_total_percent, annot=True, cmap='seismic', center=0, vmin=-150, vmax=150)
# plt.show()
#
#
# means_total_leveled = means_total.sub(means_total.mean(axis=1), axis=0)
# plt.figure(figsize=(9,16))
# plt.title('Expression of {} genes - Tumor'.format(regulated_pathway))
# sns.heatmap(means_total_leveled, annot=True, cmap='seismic', center=0, vmin=-50, vmax=50)
# plt.show()
#
#
# means_total_normal_percent = means_total_normal.sub(means_total_normal.mean(axis=1), axis=0) / means_total_normal * 100
# plt.figure(figsize=(9,16))
# plt.title('Percentage diff expression of {} genes - Normal'.format(regulated_pathway))
# sns.heatmap(means_total_normal_percent, annot=True, cmap='seismic', center=0, vmin=-150, vmax=150)
# plt.show()
#
#
#
# means_total_normal_leveled = means_total_normal.sub(means_total_normal.mean(axis=1), axis=0)
# plt.figure(figsize=(9,16))
# plt.title('Expression of {} genes - Normal'.format(regulated_pathway))
# sns.heatmap(means_total_normal_leveled, annot=True, cmap='seismic', center=0, vmin=-50, vmax=50)
# plt.show()
#
#
#
#
# # plotting the normal and the tumour tissue side to side:
#
# for p in range(1, 5):
#     tissue_comparison = pd.DataFrame(columns=['Normal', 'Tumour'])
#     tissue_comparison['Normal'] = means_total_normal['CMS{}'.format(p)]
#     tissue_comparison['Tumour'] = means_total['CMS{}'.format(p)]
#
#     plt.figure(figsize=(9,16))
#     plt.title('Expression of {} genes from cms{} - Normal vs Tumour'.format(regulated_pathway, p))
#     sns.heatmap(tissue_comparison, annot=True, cmap='seismic', center=0, vmin=0, vmax=30)
#     plt.show()
#
#


# # expression
# # print("These are the names returned by the sc.tl function: \n", return_object)
# # print("These are the scores returned by the sc.tl function: \n", scores)
#
#
# sc.pl.rank_genes_groups_tracksplot(adata, groupby='CMS_subtype', n_genes=20,
#                                    save='diff_expr_{}.pdf'.format(cell_type_to_look_at.replace(" ", "") + str(
#                                        time.strftime("%Y%m%d-%H%M%S"))),
#                                    title='{}'.format(cell_type_to_look_at))  # plot the result
#
# # # apply tsne to the data
# # sc.tl.tsne(adata, n_jobs=24)
# # sc.pl.tsne(adata, color=['subtype'], save='tsne_{}.pdf'.format(cell_type_to_look_at))
#
# # # apply pca to the data
# # sc.tl.pca(adata, svd_solver='arpack')
# # sc.pl.pca(adata, color=['CMS1', 'CMS2', 'CMS3','CMS4'])
# # sc.pl.pca_variance_ratio(adata, log=True)
# # sc.pp.pca(adata)
# # sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
# # sc.tl.umap(adata)
# # sc.pl.umap(adata, color='cell_type')
#
# # sc.pl.umap(adata, color=['CMS_subtype'])
#
#
#
# # sc.pl.umap(adata, color='cell type', legend_loc='on data',
# #            frameon=False, legend_fontsize=10)
#
# ''' -------------------------------------------- Calculate counts ------------------------------------------ '''
count_df = pd.DataFrame(columns=CMS_TYPES)
std_df = pd.DataFrame(columns=CMS_TYPES)

for subtype in CMS_TYPES:
    print((annotations_df[(annotations_df.bulk_classification == subtype) &
                                 (~annotations_df.Cell_subtype.isin(CMS_TYPES))].groupby(by='Patient')[
        'Cell_type'].value_counts() / annotations_df[(annotations_df.bulk_classification == subtype) &
                                 (~annotations_df.Cell_subtype.isin(CMS_TYPES))].groupby(by='Patient')[
        'Cell_type'].value_counts().sum(level='Patient')).mean(level='Cell_type'))

    cell_counts = (annotations_df[(annotations_df.bulk_classification == subtype) &
                                 (~annotations_df.Cell_subtype.isin(CMS_TYPES))].groupby(by='Patient')[
        'Cell_type'].value_counts() / annotations_df[(annotations_df.bulk_classification == subtype) &
                                 (~annotations_df.Cell_subtype.isin(CMS_TYPES))].groupby(by='Patient')[
        'Cell_type'].value_counts().sum(level='Patient')).mean(level='Cell_type')
    cell_counts_std = (annotations_df[(annotations_df.bulk_classification == subtype) &
                                 (~annotations_df.Cell_subtype.isin(CMS_TYPES))].groupby(by='Patient')[
        'Cell_type'].value_counts() / annotations_df[(annotations_df.bulk_classification == subtype) &
                                 (~annotations_df.Cell_subtype.isin(CMS_TYPES))].groupby(by='Patient')[
        'Cell_type'].value_counts().sum(level='Patient')).sem(level='Cell_type')
    count_df[subtype] = cell_counts
    std_df[subtype] = cell_counts_std
#
print(count_df)

chunk = 6
for idx in np.arange(0, 6, chunk):
    count_df[CMS_TYPES].iloc[idx:idx+chunk].plot(kind='bar', sharey=True, layout=(1, chunk), figsize=(12, 6), yerr=std_df)
    # count_df[CMS_TYPES].iloc[idx:idx+chunk].plot(kind='bar', sharey=True, layout=(1, chunk), figsize=(12, 6))

    plt.show()

print('These are how much the cells occur in the different subtypes: \n', count_df)
