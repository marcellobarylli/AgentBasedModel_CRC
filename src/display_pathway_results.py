import pandas as pd
import numpy as np
import json
from tabulate import tabulate
from scipy.stats import trim_mean

def match_up_genes_pvalues(side, p_cutoff, cell_type_to_look_at, tissue_class, scale):
    with open('../results/diff_exp_genes_sign_{}_{}_{}_{}_{}.txt'.format(side, p_cutoff, cell_type_to_look_at,
                                                                      tissue_class, scale)) as \
            json_file:
        genes = json.load(json_file)

    with open('../results/diff_exp_genes_sign_{}_{}_{}_{}_{}_pvalues.txt'.format(side, p_cutoff, cell_type_to_look_at,
                                                                      tissue_class, scale)) as \
            json_file:
        genes_pvalues = json.load(json_file)

    match_dataframe = pd.DataFrame(columns=['CMS', 'genes', 'pvalues'])

    for cms in ['CMS1', 'CMS2', 'CMS3', 'CMS4']:
        genes_cms_specific = genes[cms]
        pvalues_cms_specific = list(map(float, genes_pvalues[cms]))

        match_dataframe = match_dataframe.append(pd.DataFrame(np.array([[cms for i in range(len(genes_cms_specific))],
                                                                genes_cms_specific,
                                             pvalues_cms_specific]).T, columns=['CMS', 'genes', 'pvalues']),
                                                 ignore_index=True)


    return match_dataframe


def show_per_cell_type(side, p_cutoff, tissue_class, scale, cell_type_to_look_at, geneset):
    result_df = pd.read_csv('../results/pathway_analysis_{}_{}_{}_{}_{}_20210401.csv'.format(side, p_cutoff,
                                                                                    cell_type_to_look_at,
                                                                                    tissue_class, scale))
    result_df.drop(columns='Unnamed: 0', inplace=True)

    diff_exp_pvalues_dataframe = match_up_genes_pvalues(side, p_cutoff, cell_type_to_look_at, tissue_class, scale)

    diff_exp_pvalues_dataframe['pvalues'] = diff_exp_pvalues_dataframe['pvalues'].astype(float)

    # print("This is the pathways found in CMS1: \n", tabulate(result_df.loc[(result_df.CMS == 'CMS1') &
    #                                                                        (result_df.Gene_set == geneset) &
    #                                                                        (result_df['Adjusted P-value'] < 0.05)],
    #                                                          headers='keys',
    #                                                          tablefmt='csv'))
    # print("This is the pathways found in CMS2: \n", tabulate(result_df.loc[(result_df.CMS == 'CMS2') & (
    #         result_df.Gene_set == geneset) & (result_df['Adjusted P-value'] < 0.05)], headers='keys', tablefmt='csv'))
    # print("This is the pathways found in CMS3: \n", tabulate(result_df.loc[(result_df.CMS == 'CMS3') & (
    #         result_df.Gene_set == geneset)& (result_df['Adjusted P-value'] < 0.05)], headers='keys', tablefmt='csv'))
    # print("This is the pathways found in CMS4: \n", tabulate(result_df.loc[(result_df.CMS == 'CMS4') & (
    #         result_df.Gene_set == geneset)& (result_df['Adjusted P-value'] < 0.05)], headers='keys', tablefmt='csv'))

    results_for_pp = pd.concat([result_df.loc[(result_df.CMS == 'CMS1') & (
            result_df.Gene_set == geneset) & (result_df['Adjusted P-value'] < 0.05)], result_df.loc[(result_df.CMS == 'CMS2') & (
            result_df.Gene_set == geneset) & (result_df['Adjusted P-value'] < 0.05)], result_df.loc[(result_df.CMS == 'CMS3') & (
            result_df.Gene_set == geneset) & (result_df['Adjusted P-value'] < 0.05)], result_df.loc[(result_df.CMS == 'CMS4') & (
            result_df.Gene_set == geneset) & (result_df['Adjusted P-value'] < 0.05)]])

    results_for_pp['mean_pvalue_genes'] = ''
    results_for_pp['trunc_mean_pvalue_genes'] = ''

    # for group_name, group in results_for_pp.groupby(by='CMS'):
    #     for idx, row in group.iterrows():
    #         p_values = []
    #         for gene in row.Genes.split(';'):
    #             pvalue = diff_exp_pvalues_dataframe.loc[(diff_exp_pvalues_dataframe.CMS == group_name) &
    #                                                     (diff_exp_pvalues_dataframe.genes == gene)]['pvalues'].values[0]
    #             p_values.append(pvalue)
    #         results_for_pp.loc[idx, 'mean_pvalue_genes'] = np.mean(p_values)
    #         results_for_pp.loc[idx, 'trunc_mean_pvalue_genes'] = trim_mean(p_values, 0.05)


    print(len(results_for_pp.loc[(results_for_pp.CMS == 'CMS1')]))

    print("This is the pathways found in pp: \n", tabulate(results_for_pp[['CMS', 'Term', 'Overlap',
                                                                           'Adjusted P-value']],
                                                           headers='keys', tablefmt='latex_raw', showindex=False,
                                                           floatfmt='.2e'))


def show_summarised_results(side, cutoff_p, tissue_class, scale, libraries):
    pathway_ranking_cms1 = pd.read_csv(
        'pathway_ranking_cms1_sign_{}_{}_{}_{}_{}.csv'.format(side,
                                                              cutoff_p,
                                                              libraries, tissue_class,
                                                              scale))
    pathway_ranking_cms2 = pd.read_csv(
        'pathway_ranking_cms2_sign_{}_{}_{}_{}_{}.csv'.format(side,
                                                              cutoff_p,
                                                              libraries, tissue_class,
                                                              scale))
    pathway_ranking_cms3 = pd.read_csv(
        'pathway_ranking_cms3_sign_{}_{}_{}_{}_{}.csv'.format(side,
                                                              cutoff_p,
                                                              libraries, tissue_class,
                                                              scale))
    pathway_ranking_cms4 = pd.read_csv(
        'pathway_ranking_cms4_sign_{}_{}_{}_{}_{}.csv'.format(side,
                                                              cutoff_p,
                                                              libraries, tissue_class,
                                                              scale))

    print("This is the pathways found in CMS1: \n", tabulate(pathway_ranking_cms1, headers='keys', tablefmt='latex_raw'))
    print("This is the pathways found in CMS2: \n", tabulate(pathway_ranking_cms2, headers='keys', tablefmt='latex_raw'))
    print("This is the pathways found in CMS3: \n", tabulate(pathway_ranking_cms3, headers='keys', tablefmt='latex_raw'))
    print("This is the pathways found in CMS4: \n", tabulate(pathway_ranking_cms4, headers='keys', tablefmt='latex_raw'))


ALL_CELL_TYPES = ['T cells', 'B cells', 'Myeloids', 'Stromal cells', 'Epithelial cells']
ALL_PATHWAY_LIBRARIES = ['MSigDB_Hallmark_2020', 'KEGG_2019_Human', 'GO_Biological_Process_2018',
                         'GO_Cellular_Component_2018', 'Human_Phenotype_Ontology', 'MSigDB_Oncogenic_Signatures']
side = 'pos'
p_cutoff = '0025'
tissue_class = 'Tumor'
scale = 'raw'
pathway_library = 'MSigDB_Hallmark_2020'
cell_type_to_look_at = 'B cells'
show_per_cell_type(side, p_cutoff, tissue_class, scale, cell_type_to_look_at, pathway_library)

pathway_library_sum = 'hallmark'  # options are 'all' or 'hallmark'
# show_summarised_results(side, p_cutoff, tissue_class, scale, pathway_library_sum)
