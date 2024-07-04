import pandas as pd
import numpy as np
from functools import reduce
import itertools
import tqdm

from fractions import Fraction

pd.set_option("display.max_columns", 20)
pd.set_option('display.width', 1700)
pd.set_option('display.max_rows', 500)


def getIndexes(dfObj, value, cms_celltype_comparison_odds, cms_celltype_comparison_pvalue,
               cms_celltype_comparison_overlap, cms_celltype_comparison_genes):
    ''' Get index positions of value in dataframe i.e. dfObj.'''
    listOfPos = list()
    listOfOdds = list()
    listOfPvalues = list()
    listOfOverlaps = list()
    listOfGenes = list()

    # Get bool dataframe with True at positions where the given value exists
    result = dfObj.isin([value])
    # Get list of columns that contains the value
    seriesObj = result.any()
    columnNames = list(seriesObj[seriesObj == True].index)
    # Iterate over list of columns and fetch the rows indexes where value exists
    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
        odds = list(cms_celltype_comparison_odds[col][result[col] == True].values)
        pvalues = list(cms_celltype_comparison_pvalue[col][result[col] == True].values)
        overlaps = list(cms_celltype_comparison_overlap[col][result[col] == True].values)
        genes = list(cms_celltype_comparison_genes[col][result[col] == True].values)

        for row in rows:
            listOfPos.append((row, col))

        for odd in odds:
            listOfOdds.append((odd, col))

        for pvalue in pvalues:
            listOfPvalues.append((pvalue, col))

        for overlap in overlaps:
            listOfOverlaps.append((overlap, col))

        for gene in genes:
            listOfGenes.append((gene, col))

    # Return a list of tuples indicating the positions of value in the dataframe
    return listOfPos, listOfOdds, listOfPvalues, listOfOverlaps, listOfGenes


def analyse_pathways_diff_celltypes(CMS_type):
    # get a dataframe with the pathways in mentioned order, for index extraction par example
    cms_celltype_comparison = pd.DataFrame()
    cms_celltype_comparison['B cells'] = pd.Series(pw_bcells.loc[pw_bcells['CMS'] == CMS_type]['Term'].values)
    cms_celltype_comparison['T cells'] = pd.Series(pw_tcells.loc[pw_tcells['CMS'] == CMS_type]['Term'].values)
    # cms_celltype_comparison['Epithelial cells'] = pd.Series(pw_epith.loc[pw_epith['CMS'] == CMS_type]['Term'].values)
    # cms_celltype_comparison['Mast cells'] = pd.Series(pw_mast.loc[pw_mast['CMS'] == CMS_type]['Term'].values)
    cms_celltype_comparison['Myeloids'] = pd.Series(pw_myel.loc[pw_myel['CMS'] == CMS_type]['Term'].values)
    cms_celltype_comparison['Stromal cells'] = pd.Series(pw_stromal.loc[pw_stromal['CMS'] == CMS_type]['Term'].values)

    # get a dataframe full off the odds ratios mentioned in the pathway analysis
    cms_celltype_comparison_odds = pd.DataFrame()
    cms_celltype_comparison_odds['B cells'] = \
        pd.Series(pw_bcells.loc[pw_bcells['CMS'] == CMS_type]['Odds Ratio'].values)
    cms_celltype_comparison_odds['T cells'] = \
        pd.Series(pw_tcells.loc[pw_tcells['CMS'] == CMS_type]['Odds Ratio'].values)
    # cms_celltype_comparison_odds['Epithelial cells'] = \
    #     pd.Series(pw_epith.loc[pw_epith['CMS'] == CMS_type]['Odds Ratio'].values)
    # cms_celltype_comparison_odds['Mast cells'] = \
    #     pd.Series(pw_mast.loc[pw_mast['CMS'] == CMS_type]['Odds Ratio'].values)
    cms_celltype_comparison_odds['Myeloids'] = \
        pd.Series(pw_myel.loc[pw_myel['CMS'] == CMS_type]['Odds Ratio'].values)
    cms_celltype_comparison_odds['Stromal cells'] = \
        pd.Series(pw_stromal.loc[pw_stromal['CMS'] == CMS_type]['Odds Ratio'].values)

    # get a dataframe full off the adjusted P-values mentioned in the pathway analysis
    cms_celltype_comparison_pvalue = pd.DataFrame()
    cms_celltype_comparison_pvalue['B cells'] = \
        pd.Series(pw_bcells.loc[pw_bcells['CMS'] == CMS_type]["Adjusted P-value"].values)
    cms_celltype_comparison_pvalue['T cells'] = \
        pd.Series(pw_tcells.loc[pw_tcells['CMS'] == CMS_type]["Adjusted P-value"].values)
    # cms_celltype_comparison_pvalue['Epithelial cells'] = \
    #     pd.Series(pw_epith.loc[pw_epith['CMS'] == CMS_type]["Adjusted P-value"].values)
    # cms_celltype_comparison_pvalue['Mast cells'] = \
    #     pd.Series(pw_mast.loc[pw_mast['CMS'] == CMS_type]["Adjusted P-value"].values)
    cms_celltype_comparison_pvalue['Myeloids'] = \
        pd.Series(pw_myel.loc[pw_myel['CMS'] == CMS_type]["Adjusted P-value"].values)
    cms_celltype_comparison_pvalue['Stromal cells'] = \
        pd.Series(pw_stromal.loc[pw_stromal['CMS'] == CMS_type]["Adjusted P-value"].values)

    # get a dataframe full off the adjusted P-values mentioned in the pathway analysis
    cms_celltype_comparison_overlap = pd.DataFrame()
    cms_celltype_comparison_overlap['B cells'] = \
        pd.Series(pw_bcells.loc[pw_bcells['CMS'] == CMS_type]["Overlap"].values)
    cms_celltype_comparison_overlap['T cells'] = \
        pd.Series(pw_tcells.loc[pw_tcells['CMS'] == CMS_type]["Overlap"].values)
    # cms_celltype_comparison_overlap['Epithelial cells'] = \
    #     pd.Series(pw_epith.loc[pw_epith['CMS'] == CMS_type]["Overlap"].values)
    # cms_celltype_comparison_overlap['Mast cells'] = \
    #     pd.Series(pw_mast.loc[pw_mast['CMS'] == CMS_type]["Overlap"].values)
    cms_celltype_comparison_overlap['Myeloids'] = \
        pd.Series(pw_myel.loc[pw_myel['CMS'] == CMS_type]["Overlap"].values)
    cms_celltype_comparison_overlap['Stromal cells'] = \
        pd.Series(pw_stromal.loc[pw_stromal['CMS'] == CMS_type]["Overlap"].values)
    cms_celltype_comparison_overlap = \
        cms_celltype_comparison_overlap.applymap(lambda x: float(Fraction(x)), na_action='ignore')


    # get a dataframe full off the adjusted P-values mentioned in the pathway analysis
    cms_celltype_comparison_genes = pd.DataFrame()
    cms_celltype_comparison_genes['B cells'] = \
        pd.Series(pw_bcells.loc[pw_bcells['CMS'] == CMS_type]["Genes"].values)
    cms_celltype_comparison_genes['T cells'] = \
        pd.Series(pw_tcells.loc[pw_tcells['CMS'] == CMS_type]["Genes"].values)
    # cms_celltype_comparison_genes['Epithelial cells'] = \
    #     pd.Series(pw_epith.loc[pw_epith['CMS'] == CMS_type]["Genes"].values)
    # cms_celltype_comparison_genes['Mast cells'] = \
    #     pd.Series(pw_mast.loc[pw_mast['CMS'] == CMS_type]["Genes"].values)
    cms_celltype_comparison_genes['Myeloids'] = \
        pd.Series(pw_myel.loc[pw_myel['CMS'] == CMS_type]["Genes"].values)
    cms_celltype_comparison_genes['Stromal cells'] = \
        pd.Series(pw_stromal.loc[pw_stromal['CMS'] == CMS_type]["Genes"].values)

    # cms_celltype_comparison_overlap = cms_celltype_comparison_overlap.apply(lambda x: float(Fraction(str(x))))

    # get the unique pathways and remove possible nans
    selected_pathways_cms = pd.unique(cms_celltype_comparison.values.ravel('K'))
    selected_pathways_cms = [x for x in selected_pathways_cms if x == x]

    pathway_ranking = pd.DataFrame(columns=['pathway', 'index_mean', 'index_std', 'index_max', 'index_min',
                                            'odd_mean', 'odd_max', 'odd_min', 'pvalues_mean', 'pvalues_max',
                                            'pvalues_min', 'overlaps_mean', 'overlaps_max', 'overlaps_min',
                                            'presence', 'gene_inter',
                                            'gene_union'])

    for pathway in tqdm.tqdm(selected_pathways_cms):
        indices, odds, pvalues, overlaps, genes = getIndexes(cms_celltype_comparison, pathway,
                                                          cms_celltype_comparison_odds,
                                                      cms_celltype_comparison_pvalue, cms_celltype_comparison_overlap,
                                                      cms_celltype_comparison_genes)

        presence_in_cell_types = len(overlaps)

        if presence_in_cell_types > 3:
            index_mean = np.mean([index[0] for index in indices])
            index_std = np.std([index[0] for index in indices])
            index_max = np.max([index[0] for index in indices])
            index_min = np.min([index[0] for index in indices])

            odds_mean = np.mean([odd[0] for odd in odds])
            odds_max = np.max([odd[0] for odd in odds])
            odds_min = np.min([odd[0] for odd in odds])

            pvalues_mean = np.mean([pvalue[0] for pvalue in pvalues])
            pvalues_max = np.max([pvalue[0] for pvalue in pvalues])
            pvalues_min = np.min([pvalue[0] for pvalue in pvalues])

            overlaps_mean = np.mean([overlap[0] for overlap in overlaps])
            overlaps_max = np.max([overlap[0] for overlap in overlaps])
            overlaps_min = np.min([overlap[0] for overlap in overlaps])


            try:
                gene_lists = [gene[0].split(';') for gene in genes]
            except Exception as e:
                print(genes)

            unique_genes = list(set([item for sublist in gene_lists for item in sublist]))

            genes_intersection = []
            for unique_gene in unique_genes:
                counter = 0
                for gene_list in gene_lists:
                    if unique_gene in gene_list:
                        counter += 1 
                        
                if counter / len(gene_lists) > 0.5:
                    genes_intersection.append(unique_gene)


            pathway_ranking = pathway_ranking.append({'pathway': pathway, 'index_mean': index_mean, 'index_std':
                index_std, 'index_max': index_max, 'index_min': index_min,
                                                      'odd_mean': odds_mean, 'odd_max': odds_max,
                                                      'odd_min': odds_min, 'pvalues_mean': pvalues_mean, 'pvalues_max':
                                                          pvalues_max, 'pvalues_min': pvalues_min, 'overlaps_mean':
                                                          overlaps_mean, 'overlaps_max': overlaps_max, 'overlaps_min':
                                                          overlaps_min, 'presence': presence_in_cell_types,
                                                      'gene_inter': genes_intersection, 'gene_union': unique_genes},
                                                     ignore_index=True)

    pathway_ranking.sort_values(by='pvalues_mean', inplace=True)

    print('This is the comparison of the important pathways for ', CMS_type)
    print(pathway_ranking[['pathway', 'pvalues_mean', 'pvalues_max',
                           'pvalues_min', 'overlaps_mean', 'overlaps_max', 'overlaps_min', 'presence',
                           'gene_inter']].head(20))

    return pathway_ranking


def analyse_pathways_diff_cms(cell_type):
    # get a dataframe with the pathways in mentioned order, for index extraction par example
    cms_celltype_comparison = pd.DataFrame()

    if cell_type == 'B cells':
        df_ = pw_bcells.copy()
    elif cell_type == 'T cells':
        df_ = pw_tcells.copy()
    elif cell_type == 'Epithelial cells':
        df_ = pw_epith.copy()
    elif cell_type == 'Mast cells':
        df_ = pw_mast.copy()
    elif cell_type == 'Myeloids':
        df_ = pw_myel.copy()
    elif cell_type == 'Stromal cells':
        df_ = pw_stromal.copy()

    cms_celltype_comparison['CMS1'] = pd.Series(df_.loc[df_['CMS'] == 'CMS1']['Term'].values)
    cms_celltype_comparison['CMS2'] = pd.Series(df_.loc[df_['CMS'] == 'CMS2']['Term'].values)
    cms_celltype_comparison['CMS3'] = pd.Series(df_.loc[df_['CMS'] == 'CMS3']['Term'].values)
    cms_celltype_comparison['CMS4'] = pd.Series(df_.loc[df_['CMS'] == 'CMS4']['Term'].values)

    # get a dataframe full off the odds ratios mentioned in the pathway analysis
    cms_celltype_comparison_odds = pd.DataFrame()
    cms_celltype_comparison_odds['CMS1'] = pd.Series(df_.loc[df_['CMS'] == 'CMS1']['Odds Ratio'].values)
    cms_celltype_comparison_odds['CMS2'] = pd.Series(df_.loc[df_['CMS'] == 'CMS2']['Odds Ratio'].values)
    cms_celltype_comparison_odds['CMS3'] = pd.Series(df_.loc[df_['CMS'] == 'CMS3']['Odds Ratio'].values)
    cms_celltype_comparison_odds['CMS4'] = pd.Series(df_.loc[df_['CMS'] == 'CMS4']['Odds Ratio'].values)

    # get a dataframe full off the adjusted P-values mentioned in the pathway analysis
    cms_celltype_comparison_pvalue = pd.DataFrame()
    cms_celltype_comparison_pvalue['CMS1'] = pd.Series(df_.loc[df_['CMS'] == 'CMS1']["Adjusted P-value"].values)
    cms_celltype_comparison_pvalue['CMS2'] = pd.Series(df_.loc[df_['CMS'] == 'CMS2']["Adjusted P-value"].values)
    cms_celltype_comparison_pvalue['CMS3'] = pd.Series(df_.loc[df_['CMS'] == 'CMS3']["Adjusted P-value"].values)
    cms_celltype_comparison_pvalue['CMS4'] = pd.Series(df_.loc[df_['CMS'] == 'CMS4']["Adjusted P-value"].values)

    # get a dataframe full off the adjusted P-values mentioned in the pathway analysis
    cms_celltype_comparison_overlap = pd.DataFrame()
    cms_celltype_comparison_overlap['CMS1'] = pd.Series(df_.loc[df_['CMS'] == 'CMS1']["Overlap"].values)
    cms_celltype_comparison_overlap['CMS2'] = pd.Series(df_.loc[df_['CMS'] == 'CMS2']["Overlap"].values)
    cms_celltype_comparison_overlap['CMS3'] = pd.Series(df_.loc[df_['CMS'] == 'CMS3']["Overlap"].values)
    cms_celltype_comparison_overlap['CMS4'] = pd.Series(df_.loc[df_['CMS'] == 'CMS4']["Overlap"].values)
    cms_celltype_comparison_overlap = \
        cms_celltype_comparison_overlap.applymap(lambda x: float(Fraction(x)), na_action='ignore')

    # cms_celltype_comparison_overlap = cms_celltype_comparison_overlap.apply(lambda x: float(Fraction(str(x))))

    selected_pathways_cms = pd.unique(cms_celltype_comparison.values.ravel('K'))

    pathway_ranking = pd.DataFrame(columns=['pathway', 'index_mean', 'index_std', 'index_max', 'index_min',
                                            'odd_mean', 'odd_max', 'odd_min', 'pvalues_mean', 'pvalues_max',
                                            'pvalues_min', 'overlaps_mean', 'overlaps_max', 'overlaps_min', 'presence'])
    for pathway in tqdm.tqdm(selected_pathways_cms):
        indices, odds, pvalues, overlaps = getIndexes(cms_celltype_comparison, pathway, cms_celltype_comparison_odds,
                                                      cms_celltype_comparison_pvalue, cms_celltype_comparison_overlap)

        presence_in_cell_types = len(overlaps)

        if presence_in_cell_types > 2:
            index_mean = np.mean([index[0] for index in indices])
            index_std = np.std([index[0] for index in indices])
            index_max = np.max([index[0] for index in indices])
            index_min = np.min([index[0] for index in indices])

            odds_mean = np.mean([odd[0] for odd in odds])
            odds_max = np.max([odd[0] for odd in odds])
            odds_min = np.min([odd[0] for odd in odds])

            pvalues_mean = np.mean([pvalue[0] for pvalue in pvalues])
            pvalues_max = np.max([pvalue[0] for pvalue in pvalues])
            pvalues_min = np.min([pvalue[0] for pvalue in pvalues])

            overlaps_mean = np.mean([overlap[0] for overlap in overlaps])
            overlaps_max = np.max([overlap[0] for overlap in overlaps])
            overlaps_min = np.min([overlap[0] for overlap in overlaps])

            pathway_ranking = pathway_ranking.append({'pathway': pathway, 'index_mean': index_mean, 'index_std':
                index_std, 'index_max': index_max, 'index_min': index_min,
                                                      'odd_mean': odds_mean, 'odd_max': odds_max,
                                                      'odd_min': odds_min, 'pvalues_mean': pvalues_mean, 'pvalues_max':
                                                          pvalues_max, 'pvalues_min': pvalues_min, 'overlaps_mean':
                                                          overlaps_mean, 'overlaps_max': overlaps_max, 'overlaps_min':
                                                          overlaps_min, 'presence': presence_in_cell_types},
                                                     ignore_index=True)

    pathway_ranking.sort_values(by='pvalues_mean', inplace=True)

    print('\n This is the comparison of the important pathways for ', cell_type)
    print(pathway_ranking[['pathway', 'odd_mean', 'odd_max', 'odd_min', 'pvalues_mean', 'pvalues_max',
                           'pvalues_min', 'overlaps_mean', 'overlaps_max', 'overlaps_min', 'presence']].head(20))

    return pathway_ranking


side = 'neg'
cutoff_p = '0025'
libraries = 'hallmark'
tissue_class = 'Normal'
scale = 'raw'

# import the results of the pathway analyses
pw_bcells = pd.read_csv('../results/pathway_analysis_{}_{}_{}_{}_{}.csv'.format(side, cutoff_p, 'B cells',
                                                                              tissue_class, scale))
pw_tcells = pd.read_csv('../results/pathway_analysis_{}_{}_{}_{}_{}.csv'.format(side, cutoff_p, 'T cells', tissue_class, scale))
# pw_epith = pd.read_csv('../results/pathway_analysis_{}_{}_{}_{}_{}.csv'.format(side, cutoff_p, 'Epithelial cells', tissue_class, scale))
# pw_mast = pd.read_csv('../results/pathway_analysis_{}_{}_{}_{}_{}.csv'.format(side, cutoff_p, 'Mast cells', tissue_class, scale))
pw_myel = pd.read_csv('../results/pathway_analysis_{}_{}_{}_{}_{}.csv'.format(side, cutoff_p, 'Myeloids', tissue_class, scale))
pw_stromal = pd.read_csv('../results/pathway_analysis_{}_{}_{}_{}_{}.csv'.format(side, cutoff_p, 'Stromal cells', tissue_class, scale))

# only select the libraries that we want
if libraries == 'hallmark':
    pw_bcells = pw_bcells.loc[pw_bcells.Gene_set == 'MSigDB_Hallmark_2020']
    pw_tcells = pw_tcells.loc[pw_tcells.Gene_set == 'MSigDB_Hallmark_2020']
    # pw_epith = pw_epith.loc[pw_epith.Gene_set == 'MSigDB_Hallmark_2020']
    # pw_mast = pw_mast.loc[pw_mast.Gene_set == 'MSigDB_Hallmark_2020']
    pw_myel = pw_myel.loc[pw_myel.Gene_set == 'MSigDB_Hallmark_2020']
    pw_stromal = pw_stromal.loc[pw_stromal.Gene_set == 'MSigDB_Hallmark_2020']

# pw_bcells = pw_bcells.loc[(pw_bcells.Gene_set != 'GO_Biological_Process_2018') & (pw_bcells.Gene_set !=
#                                                                                   'GO_Cellular_Component_2018')]
# pw_tcells = pw_tcells.loc[(pw_tcells.Gene_set != 'GO_Biological_Process_2018') & (pw_tcells.Gene_set !=
#                           'GO_Cellular_Component_2018')]
# pw_epith = pw_epith.loc[(pw_epith.Gene_set != 'GO_Biological_Process_2018') & (pw_epith.Gene_set !=
#                                                                                'GO_Cellular_Component_2018')]
# pw_mast = pw_mast.loc[(pw_mast.Gene_set != 'GO_Biological_Process_2018') & (pw_mast.Gene_set !=
#                                                                             'GO_Cellular_Component_2018')]
# pw_myel = pw_myel.loc[(pw_myel.Gene_set != 'GO_Biological_Process_2018') & (pw_myel.Gene_set !=
#                                                                             'GO_Cellular_Component_2018')]
# pw_stromal = pw_stromal.loc[(pw_stromal.Gene_set != 'GO_Biological_Process_2018') & (pw_stromal.Gene_set !=
#                                                                                      'GO_Cellular_Component_2018')]

''' analyse the found pathways among different cell subtypes '''

# analyse the found pathways among the different cell subtypes for CMS1
pathway_ranking_cms1 = analyse_pathways_diff_celltypes('CMS1')
pathway_ranking_cms2 = analyse_pathways_diff_celltypes('CMS2')
pathway_ranking_cms3 = analyse_pathways_diff_celltypes('CMS3')
pathway_ranking_cms4 = analyse_pathways_diff_celltypes('CMS4')

pathway_ranking_cms1.to_csv('../results/pathway_comparison/pathway_ranking_cms1_sign_{}_{}_{}_{}_{}.csv'.format(side,
                                                                                                          cutoff_p,
                                                                                                          libraries, tissue_class, scale))
pathway_ranking_cms2.to_csv('../results/pathway_comparison/pathway_ranking_cms2_sign_{}_{}_{}_{}_{}.csv'.format(side,
                                                                                                          cutoff_p,
                                                                                                          libraries, tissue_class, scale))
pathway_ranking_cms3.to_csv('../results/pathway_comparison/pathway_ranking_cms3_sign_{}_{}_{}_{}_{}.csv'.format(side,
                                                                                                          cutoff_p,
                                                                                                          libraries, tissue_class, scale))
pathway_ranking_cms4.to_csv('../results/pathway_comparison/pathway_ranking_cms4_sign_{}_{}_{}_{}_{}.csv'.format(side,
                                                                                                          cutoff_p,
                                                                                                          libraries, tissue_class, scale))

''' analyse the found pathways among different CMS subtypes '''
# analyse the found pathways among different cell subtypes
# pathway_ranking_bcells = analyse_pathways_diff_cms('B cells')
# pathway_ranking_tcells = analyse_pathways_diff_cms('T cells')
# pathway_ranking_epith = analyse_pathways_diff_cms('Epithelial cells')
# pathway_ranking_mast = analyse_pathways_diff_cms('Mast cells')
# pathway_ranking_myel = analyse_pathways_diff_cms('Myeloids')
# pathway_ranking_stromal = analyse_pathways_diff_cms('Stromal cells')
#
# pathway_ranking_bcells.to_csv('../results/pathway_comparison/pathway_ranking_bcells_hallmark_sign.csv')
# pathway_ranking_tcells.to_csv('../results/pathway_comparison/pathway_ranking_tcells_hallmark_sign.csv')
# pathway_ranking_epith.to_csv('../results/pathway_comparison/pathway_ranking_epith_hallmark_sign.csv')
# pathway_ranking_mast.to_csv('../results/pathway_comparison/pathway_ranking_mast_hallmark_sign.csv')
# pathway_ranking_myel.to_csv('../results/pathway_comparison/pathway_ranking_myel_hallmark_sign.csv')
# pathway_ranking_stromal.to_csv('../results/pathway_comparison/pathway_ranking_stromal_hallmark_sign.csv')
