import os
import ast
import mygene
import random
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm


def map_pathway_genes(df_to_map_genes_of, side, cutoff_p, celltype, scale, tissue_class, date):
    # map the gene ids to entrez genes
    mg = mygene.MyGeneInfo()

    idx_cms = 0
    with tqdm(total=df_to_map_genes_of.shape[0]) as pbar:
        for idx, row in tqdm(df_to_map_genes_of.iterrows(), position=1, colour='green'):
            genes_to_map = row.Genes
            genes_to_map = genes_to_map.split(';')
            genes_to_map = pd.Series(genes_to_map)

            already_mapped_ids = pd.read_csv('../data/hgnc-symbol-check.csv')

            already_mapped_ids['HGNC ID'] = already_mapped_ids['HGNC ID'].str.lower()
            already_mapped_ids['HGNC ID'] = already_mapped_ids['HGNC ID'].apply(
                lambda x: str(x).replace('hgnc:', ''))
            already_mapped_ids.dropna(axis=0, inplace=True)
            mapping_df = already_mapped_ids[['Input', 'HGNC ID']].set_index('Input')
            genes_to_map = genes_to_map.map(mapping_df.to_dict()['HGNC ID'])
            genes_to_map = genes_to_map.apply(lambda x: 'hgnc:' + str(x))

            genes_to_map = genes_to_map.to_list()

            mapping_df = pd.DataFrame(columns=['hgncid', 'entrez'])

            for i in range(len(genes_to_map)):
                genes_to_map[i] = str(genes_to_map[i])

                query_res = mg.query(genes_to_map[i], scopes=['hgnc'], species=['human'],
                                     as_dataframe=True)
                mapping_df = mapping_df.append(
                    {'hgncid': genes_to_map[i], 'entrez': query_res['entrezgene'].values[0]},
                    ignore_index=True)

            mapping_df = mapping_df.set_index('hgncid')
            genes_to_map = pd.Series(genes_to_map).map(mapping_df.to_dict()['entrez'])

            df_to_map_genes_of.loc[idx, 'Genes'] = str(genes_to_map.values)
            pbar.update(1)

    df_to_map_genes_of.to_csv('../results/pathway_analysis_{}_{}_{}_{}_{}_{}_mapped.csv'.format(side,
                                                                                                cutoff_p,
                                                                                                celltype,
                                                                                                tissue_class,
                                                                                                scale,
                                                                                                date))


def load_pathways_ranked(side, cutoff_p, celltype, scale, tissue_class, date):
    ''' Load the ranked pathway dfs '''

    df_ = \
        pd.read_csv('../results/pathway_analysis_{}_{}_{}_{}_{}_{}.csv'.format(side,
                                                                               cutoff_p,
                                                                               celltype,
                                                                               tissue_class,
                                                                               scale,
                                                                               date))
    return df_


for side in ['neg']:
    for celltype in ['T cells', 'B cells', 'Myeloids', 'Epithelial cells']:
        df_ = load_pathways_ranked(side=side,
                                   cutoff_p='0025',
                                   celltype=celltype,
                                   scale='raw',
                                   tissue_class='Tumor',
                                   date='20210401')

        map_pathway_genes(df_[df_.Gene_set == 'MSigDB_Hallmark_2020'], side=side,
                          cutoff_p='0025',
                          celltype=celltype,
                          scale='raw',
                          tissue_class='Tumor',
                          date='20210401')

side = 'pos'
cutoff_p = '0025'
library = 'hallmark'
tissue_class = 'Tumor'
scale = 'raw'
# map_pathway_genes(upregulated_pathways_sc, side, cutoff_p, library,
#                   tissue_class, scale)
# map_pathway_genes(downregulated_pathways_sc, side, cutoff_p, library,
#                   tissue_class, scale)
