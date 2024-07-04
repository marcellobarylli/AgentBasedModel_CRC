'''
This code runs a sensitivity analysis on the categorisation of bulk data with the genes that were deemed
important by the pathway analysis

Author: Robin van den Berg
Contact: rvdb7345@gmail.com
'''

import os
import ast
import mygene
import random
import json
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score

import warnings

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 20)
pd.set_option('display.width', 1700)
pd.set_option('display.max_rows', 500)

dir_path = os.path.dirname(os.path.realpath(__file__))
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin'

CMS_TYPES = ['CMS1', 'CMS2', 'CMS3', 'CMS4']


def map_pathway_genes(cms_lists, side, cutoff_p, libraries, tissue_class, scale):
    # map the gene ids to entrez genes
    mg = mygene.MyGeneInfo()

    cms_list = ['cms1', 'cms2', 'cms3', 'cms4']
    idx_cms = 0
    for df in tqdm(cms_lists, position=0, colour='red'):
        cms = cms_list[idx_cms]
        with tqdm(total=df.shape[0]) as pbar:
            for idx, row in tqdm(df.iterrows(), position=1, colour='green'):
                genes_to_map = row.gene_union
                genes_to_map = ast.literal_eval(genes_to_map)
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

                df.loc[idx, 'gene_union'] = str(genes_to_map.values)
                pbar.update(1)

        df.to_csv('../results/pathway_comparison/pathway_ranking_{}_sign_{}_{}_{}_{}_{}_mapped.csv'.format(cms, side,
                                                                                                           cutoff_p,
                                                                                                           libraries,
                                                                                                           tissue_class,
                                                                                                           scale))
        idx_cms += 1


def get_ranges(bulk_df, genes):
    overlapping_genes = []
    for gene in genes:
        if gene in (list(bulk_df.keys())):
            overlapping_genes.append(gene)
    gene_selection = bulk_df.loc[:, overlapping_genes]
    # gene_selection.plot.hist(alpha=0.5, bins=100)

    upper_bounds = gene_selection[np.abs(gene_selection-gene_selection.mean()) <= (4*gene_selection.std())].max(
        axis=0)
    lower_bounds = gene_selection[np.abs(gene_selection-gene_selection.mean()) <= (
            4*gene_selection.std())].min(axis=0)
    average_expression = gene_selection[np.abs(gene_selection-gene_selection.mean()) <= (
            1*gene_selection.std())].mean(axis=0)

    # average_expression_1 = gene_selection[(np.abs(gene_selection - gene_selection.mean()) <= (
    #         4 * gene_selection.std()))][bulk_df['fin_dec'] == 'CMS1'].mean(axis=0)
    # average_expression_2 = gene_selection[(np.abs(gene_selection - gene_selection.mean()) <= (
    #         4 * gene_selection.std()))][(bulk_df['fin_dec'] == 'CMS2')].mean(axis=0)
    # average_expression_3 = gene_selection[(np.abs(gene_selection - gene_selection.mean()) <= (
    #         4 * gene_selection.std()))][(bulk_df['fin_dec'] == 'CMS3')].mean(axis=0)
    # average_expression_4 = gene_selection[(np.abs(gene_selection - gene_selection.mean()) <= (
    #         4 * gene_selection.std()))][(bulk_df['fin_dec'] == 'CMS4')].mean(axis=0)
    #
    # average_expression = (average_expression_1 + average_expression_2 + average_expression_3 +
    #                       average_expression_4) / len(bulk_df['fin_dec'].unique())

    return overlapping_genes, upper_bounds, lower_bounds, average_expression


def get_well_classified_samples(df_merged, CMS_to_affect, num_samples, params):
    """ only get samples that were well classified in the first case to not obscure the effect of the regulation """

    indices_cms = df_merged.loc[df_merged.fin_dec == CMS_to_affect].index.values

    # get the index of the prediction value that should be largest
    if CMS_to_affect == 'CMS1':
        correct_idx = 0
    if CMS_to_affect == 'CMS2':
        correct_idx = 1
    if CMS_to_affect == 'CMS3':
        correct_idx = 2
    if CMS_to_affect == 'CMS4':
        correct_idx = 3

    obtained_samples = 0
    samples_to_change_idx = []
    while obtained_samples < num_samples:
        sample_to_change_idx = random.sample(list(indices_cms), 1)

        X_train = df_merged.loc[~df_merged.index.isin(sample_to_change_idx), df_merged.columns.difference(
            ['sample', 'dataset', 'CMS_network',
             'CMS_RFclassifier', 'fin_dec',
             'cat_code'])]
        X_test = df_merged.loc[df_merged.index.isin(sample_to_change_idx), df_merged.columns.difference(
            ['sample', 'dataset', 'CMS_network',
             'CMS_RFclassifier', 'fin_dec',
             'cat_code'])]
        y_train = df_merged.iloc[~df_merged.index.isin(sample_to_change_idx)].cat_code
        y_test = df_merged.iloc[df_merged.index.isin(sample_to_change_idx)].cat_code

        # train the
        clf = lgb.LGBMRegressor(**params)
        clf = clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)

        print(prediction)

        if prediction[0][correct_idx] > 0.90:
            obtained_samples += 1
            samples_to_change_idx.append(sample_to_change_idx[0])
        print(samples_to_change_idx)

    print(samples_to_change_idx)
    return samples_to_change_idx


def create_plots(regulated_pathway, CMS_to_affect, CMS_pathways, side, num_samples, average_preds, std_preds):
    plt.figure()
    plt.title('Regulating pathways in original {} cells'.format(CMS_to_affect))
    plt.plot(range(-num_samples + 1, num_samples), average_preds[:, 0], label='CMS1', color='blue')
    plt.fill_between(range(-num_samples + 1, num_samples), average_preds[:, 0] - std_preds[:, 0], average_preds[:,
                                                                                                  0] + std_preds[:, 0],
                     alpha=0.3, color='blue')

    plt.plot(range(-num_samples + 1, num_samples), average_preds[:, 1], label='CMS2', color='orange')
    plt.fill_between(range(-num_samples + 1, num_samples), average_preds[:, 1] - std_preds[:, 1],
                     average_preds[:, 1] + std_preds[:, 1],
                     alpha=0.3, color='orange')

    plt.plot(range(-num_samples + 1, num_samples), average_preds[:, 2], label='CMS3', color='green')
    plt.fill_between(range(-num_samples + 1, num_samples), average_preds[:, 2] - std_preds[:, 2],
                     average_preds[:, 2] + std_preds[:, 2],
                     alpha=0.3, color='green')

    plt.plot(range(-num_samples + 1, num_samples), average_preds[:, 3], label='CMS4', color='red')
    plt.fill_between(range(-num_samples + 1, num_samples), average_preds[:, 3] - std_preds[:, 3],
                     average_preds[:, 3] + std_preds[:, 3],
                     alpha=0.3, color='red')

    plt.xlabel('Regulation steps towards max expression')
    plt.ylabel('Regression values')
    plt.legend()
    plt.grid()
    plt.show()


def get_genes_to_regulate(pathways_to_be_up, upregulated_pathways_sc, downregulated_pathways_sc):
    """ Extracts the genes from a pathway df """

    regulated_genes_cummulative = []

    for direction, pathway_dfs in zip(['pos', 'neg'], [upregulated_pathways_sc, downregulated_pathways_sc]):
        for idx, type in enumerate(['CMS1', 'CMS2', 'CMS3', 'CMS4']):
            pathway_df = pathway_dfs[idx].copy()
            pathway_row = pathway_df.loc[pathway_df['pathway'].isin(pathways_to_be_up[direction][type])]
            for gene_list in pathway_row['gene_union'].values:
                regulated_genes = gene_list.replace(' ', ',')
                regulated_genes = ast.literal_eval(regulated_genes)
                regulated_genes_cummulative += [str(gene) for gene in regulated_genes]

    return list(set(regulated_genes_cummulative))


def load_pathways_ranked(side, cutoff_p, libraries, scale, tissue_class):
    ''' Load the ranked pathway dfs '''

    cms1_ranked = \
        pd.read_csv('../results/pathway_comparison/pathway_ranking_cms1_sign_{}_{}_{}_{}_{}_mapped.csv'.format(side,
                                                                                                               cutoff_p,
                                                                                                               libraries,
                                                                                                               tissue_class,
                                                                                                               scale))
    cms2_ranked = \
        pd.read_csv('../results/pathway_comparison/pathway_ranking_cms2_sign_{}_{}_{}_{}_{}_mapped.csv'.format(side,
                                                                                                               cutoff_p,
                                                                                                               libraries,
                                                                                                               tissue_class,
                                                                                                               scale))
    cms3_ranked = \
        pd.read_csv('../results/pathway_comparison/pathway_ranking_cms3_sign_{}_{}_{}_{}_{}_mapped.csv'.format(side,
                                                                                                               cutoff_p,
                                                                                                               libraries,
                                                                                                               tissue_class,
                                                                                                               scale))
    cms4_ranked = \
        pd.read_csv('../results/pathway_comparison/pathway_ranking_cms4_sign_{}_{}_{}_{}_{}_mapped.csv'.format(side,
                                                                                                               cutoff_p,
                                                                                                               libraries,
                                                                                                               tissue_class,
                                                                                                               scale))

    return cms1_ranked, cms2_ranked, cms3_ranked, cms4_ranked

def load_pathways_type_specific(side, cutoff_p, celltype, scale, tissue_class, date):
    ''' Load the ranked pathway dfs '''

    df_ = \
        pd.read_csv('../results/pathway_analysis_{}_{}_{}_{}_{}_{}_mapped.csv'.format(side,
                                                                               cutoff_p,
                                                                               celltype,
                                                                               tissue_class,
                                                                               scale,
                                                                               date))
    df_.rename(columns={'Genes': 'gene_union', 'Term': 'pathway'}, inplace=True)

    return df_[df_.CMS == 'CMS1'], df_[df_.CMS == 'CMS2'], df_[df_.CMS == 'CMS3'], df_[df_.CMS == 'CMS4']


if __name__ == '__main__':

    # load the data
    df_crc_data = pd.read_table(os.path.join(dir_path, '../data/formatted_crc_data.txt'))
    df_crc_data['sample'] = df_crc_data.index
    df_labels = pd.read_table(os.path.join(dir_path, '../data/cms_labels_public_all.txt'))

    # match up the assigned labels with the expression rates
    df_merged = pd.merge(df_crc_data, df_labels, on='sample')
    df_merged.rename({'CMS_final_network_plus_RFclassifier_in_nonconsensus_samples': 'fin_dec'}, axis='columns',
                     inplace=True)

    # drop the samples without labels
    df_merged = df_merged.loc[df_merged.fin_dec != 'NOLBL']

    df_merged.fin_dec = pd.Categorical(df_merged.fin_dec)
    df_merged['cat_code'] = df_merged.fin_dec.cat.codes


    # loop through all the cell types that are available
    for cell_type in ['T cells', 'B cells', 'Myeloids', 'Epithelial cells', 'Stromal cells']:

        # load the pathway genes
        cms1_ranked_pos, cms2_ranked_pos, cms3_ranked_pos, cms4_ranked_pos = load_pathways_type_specific(side='pos',
                                       cutoff_p='0025',
                                       celltype=cell_type,
                                       scale='raw',
                                       tissue_class='Tumor',
                                       date='20210401')
        cms1_ranked_neg, cms2_ranked_neg, cms3_ranked_neg, cms4_ranked_neg = load_pathways_type_specific(side='neg',
                                       cutoff_p='0025',
                                       celltype=cell_type,
                                       scale='raw',
                                       tissue_class='Tumor',
                                       date='20210401')

        upregulated_pathways_sc = [cms1_ranked_pos, cms2_ranked_pos, cms3_ranked_pos, cms4_ranked_pos]
        downregulated_pathways_sc = [cms1_ranked_neg, cms2_ranked_neg, cms3_ranked_neg, cms4_ranked_neg]


        # define the naming in the library vs how the pathways are nameed in the model
        pathway_name_map = {
            "Oxidative Phosphorylation": "oxidative_phosphorylation_activation",
            "Glycolysis": "glycolysis_activation",
            "p53 Pathway" :"P53_activation",
            "mTORC1 Signaling": "mTORC1_activation",
            "TNF-alpha Signaling via NF-kB": "TNFaviaNFkB_activation",
            "Unfolded Protein Response": "unfolded_protein_response_activation",
            "Hypoxia": "hypoxia_activation",
            "Epithelial Mesenchymal Transition": "EMT_activation",
            "Myogenesis": "myogenesis_activation",
            "Myc Targets V1": "MYC_target_v1_activation",
            "Reactive Oxygen Species Pathway": "ROS_pathway_activation",
            "IL-2/STAT5 Signaling": "IL2_STAT5_signalling_activation",
            "Pperoxisome": "peroxisome_activation",
            "Adipogenesis": "adipogenesis_activation",
            "Interferon Gamma Response": "IFN_gamma_activation",
            "KRAS Signaling Up": "kras_signalling_activation",
            "IL-6/JAK/STAT3 Signaling": "IL6_JAK_activation",
            "Complement": "complement_activation",
            "Interferon Alpha Response": "interferon_a_pathway_activation",
            "PI3K/AKT/mTOR  Signaling": "PI3K_activation"
        }

        # loop through the pathways that are available
        for influenced_pathway in pathway_name_map.keys():
            print('going for {}, {}'.format(cell_type, influenced_pathway))
            
            pathways_to_be_up = {'pos': {'CMS1': [influenced_pathway],
                                         'CMS2': [influenced_pathway],
                                         'CMS3': [influenced_pathway],
                                         'CMS4': [influenced_pathway]},
                                 'neg': {'CMS1': [influenced_pathway],
                                         'CMS2': [influenced_pathway],
                                         'CMS3': [influenced_pathway],
                                         'CMS4': [influenced_pathway]}
                                 }


            '''
            just possible pathways
                'Epithelial Mesenchymal Transition', 'Coagulation', 'Apoptosis', 'Angiogenesis',
                'TNF-alpha Signaling via NF-kB', 'Hypoxia', 'Allograft Rejection', 'IL-6/JAK/STAT3 Signaling',
                'Inflammatory Response', 'Complement', 'Myogenesis'
        
                'Complement', 'Fatty Acid Metabolism', 'Androgen Response'
        
        
                'mTORC1 Signaling', 'UV Response Up', 'Epithelial Mesenchymal Transition',
                                                    'p53 Pathway', 'Unfolded Protein Response', 'Hypoxia', 'Apoptosis',
                                                    'Interferon Gamma Response'
            '''

            # samples_to_change_idx = get_well_classified_samples(df_merged, CMS_to_affect, num_samples, params)

            # get a full list of genes that are mentioned as overlapping with the pathways
            regulated_genes_cummulative_pos = get_genes_to_regulate(pathways_to_be_up, upregulated_pathways_sc,
                                                                    downregulated_pathways_sc)

            # get the average values, upper and lower bounds of all genes associated with pathways over all samples
            if regulated_genes_cummulative_pos:
                overlapping_genes_pos, upper_bounds_pos, lower_bounds_pos, average_exp_pos_regulated_allcms = \
                    get_ranges(df_merged, regulated_genes_cummulative_pos)

                relative_activation_df = pd.DataFrame({'lower_bound': lower_bounds_pos,
                                                           'upper_bound': upper_bounds_pos,
                                                           'all_average': average_exp_pos_regulated_allcms})


            # get the average values, upper and lower bounds of all genes associated with pathways over specific CMSs
            for CMS_to_affect in CMS_TYPES:
                CMS_specific_df = df_merged.loc[df_merged.fin_dec == CMS_to_affect]

                if regulated_genes_cummulative_pos:
                    _, _, _, average_exp_pos_regulated_cms_spec = \
                        get_ranges(CMS_specific_df, regulated_genes_cummulative_pos)

                if regulated_genes_cummulative_pos:
                    relative_activation_df[CMS_to_affect + '_average'] = average_exp_pos_regulated_cms_spec

            activation_values = np.zeros((len(relative_activation_df), 4))

            # abstract the activation values for the simulation based on the procentage difference of a CMS to the
            idx = 0
            for index, row in relative_activation_df.iterrows():
                for idx_cms, cms in enumerate(['CMS1_average', 'CMS2_average', 'CMS3_average', 'CMS4_average']):
                    diff = (row[cms] - row['all_average'])

                    if diff < 0:
                        activation_values[idx, idx_cms] = diff / abs(row['all_average'] - row['lower_bound']) / 100
                    elif diff > 0:
                        activation_values[idx, idx_cms] = diff / abs(row['upper_bound'] - row['all_average'])  / 100

                idx += 1

            average_activation = activation_values.sum(axis=0)

            # write the activation values of the pathways to a json file for initiation of the simulation
            with open('model/init_pathway_activation.json') as json_file:
                data = json.load(json_file)

                pathway_name = pathway_name_map[influenced_pathway]
                if cell_type == 'Stromal cell':
                    cell_type_json = 'stromalcell'
                if cell_type == 'T cells':
                    cell_type_json = 'tcell'
                if cell_type == 'B cells':
                    cell_type_json = 'bcell'
                if cell_type == 'Epithelial cells':
                    cell_type_json = 'cancercell'
                if cell_type == 'Myeloids':
                    cell_type_json = 'myeloid'

                for i, cms in enumerate(['cms1', 'cms2', 'cms3', 'cms4']):
                    data[cms][cell_type_json][pathway_name] = average_activation[i]


            with open('model/init_pathway_activation_reav.json', "w") as json_file:
                json.dump(data, json_file, indent=4)

