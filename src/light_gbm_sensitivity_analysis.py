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
import lightgbm as lgb
import matplotlib
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

    gene_selection.plot.hist(alpha=0.5, bins=100)

    upper_bounds = gene_selection.max(axis=0)
    lower_bounds = gene_selection.min(axis=0)

    return overlapping_genes, upper_bounds, lower_bounds


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
    plt.plot(np.arange(-1+0.01, 1, 0.01), average_preds[:, 0], label='CMS1', color='blue')
    plt.fill_between(np.arange(-1+0.01, 1, 0.01), average_preds[:, 0] - std_preds[:, 0], average_preds[:,
                                                                                                  0] + std_preds[:, 0],
                     alpha=0.3, color='blue')

    plt.plot(np.arange(-1+0.01, 1, 0.01), average_preds[:, 1], label='CMS2', color='orange')
    plt.fill_between(np.arange(-1+0.01, 1, 0.01), average_preds[:, 1] - std_preds[:, 1],
                     average_preds[:, 1] + std_preds[:, 1],
                     alpha=0.3, color='orange')

    plt.plot(np.arange(-1+0.01, 1, 0.01), average_preds[:, 2], label='CMS3', color='green')
    plt.fill_between(np.arange(-1+0.01, 1, 0.01), average_preds[:, 2] - std_preds[:, 2],
                     average_preds[:, 2] + std_preds[:, 2],
                     alpha=0.3, color='green')

    plt.plot(np.arange(-1+0.01, 1, 0.01), average_preds[:, 3], label='CMS4', color='red')
    plt.fill_between(np.arange(-1+0.01, 1, 0.01), average_preds[:, 3] - std_preds[:, 3],
                     average_preds[:, 3] + std_preds[:, 3],
                     alpha=0.3, color='red')

    plt.xlabel(r'$\delta$', fontsize=16)
    plt.ylabel('Regression values', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.tight_layout()
    # plt.grid()
    plt.savefig('syntexp_cms4cms2_{}.pdf'.format(CMS_to_affect))
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

    print("This is the dataframe of the bulk data", df_merged)

    # load the pathway genes
    cms1_ranked_pos, cms2_ranked_pos, cms3_ranked_pos, cms4_ranked_pos = load_pathways_ranked(side='pos',
                                                                                              cutoff_p='0025',
                                                                                              libraries='hallmark',
                                                                                              scale='raw',
                                                                                              tissue_class='Tumor')
    cms1_ranked_neg, cms2_ranked_neg, cms3_ranked_neg, cms4_ranked_neg = load_pathways_ranked(side='neg',
                                                                                              cutoff_p='0025',
                                                                                              libraries='hallmark',
                                                                                              scale='raw',
                                                                                              tissue_class='Tumor')

    upregulated_pathways_sc = [cms1_ranked_pos, cms2_ranked_pos, cms3_ranked_pos, cms4_ranked_pos]
    downregulated_pathways_sc = [cms1_ranked_neg, cms2_ranked_neg, cms3_ranked_neg, cms4_ranked_neg]

    side = 'pos'
    cutoff_p = '0025'
    library = 'hallmark'
    tissue_class = 'Tumor'
    scale = 'raw'
    # map_pathway_genes(upregulated_pathways_sc, side, cutoff_p, library,
    #                   tissue_class, scale)
    # map_pathway_genes(downregulated_pathways_sc, side, cutoff_p, library,
    #                   tissue_class, scale)

    # setting the parameters for our lgbm model
    params = {
        'task': 'train', 'objective': 'multiclass', 'num_class': 4, 'boosting_type': 'gbdt',
        'min_data': 50, 'verbose': -1, 'max_depth': 10, 'bagging_fraction': 0.95, 'feature_fraction': 0.99,
        'num_leaves': 10, 'metric': ['multi_error', 'multi_logloss']
    }

    num_samples = 100  # number of bulk samples to select from a certain CMS

    pathways_to_be_down = {'pos': {'CMS1': [],
                                   'CMS2': [],
                                   'CMS3': [],
                                   'CMS4': ['Epithelial Mesenchymal Transition', 'Angiogenesis',  'TNF-alpha Signaling via NF-kB',
                                            'Hypoxia', 'Complement', 'Myogenesis', 'Glycolysis', 'p53 Pathway']},
                           'neg': {'CMS1': [],
                                   'CMS2': ['Epithelial Mesenchymal Transition', 'Angiogenesis',  'TNF-alpha Signaling via NF-kB',
                                            'Hypoxia', 'Complement', 'Myogenesis', 'Glycolysis'],
                                   'CMS3': [],
                                   'CMS4': []}
                           }  # which pathways we want to downregulation when moving to the right of the graph
    pathways_to_be_up = {'pos': {'CMS1': ['Fatty Acid Metabolism'],
                                 'CMS2': [],
                                 'CMS3': [],
                                 'CMS4': []},
                         'neg': {'CMS1': [],
                                 'CMS2': [],
                                 'CMS3': [],
                                 'CMS4': []}}

    for CMS_to_affect in CMS_TYPES:

        '''
            'Epithelial Mesenchymal Transition', 'Coagulation', 'Apoptosis', 'Angiogenesis',
            'TNF-alpha Signaling via NF-kB', 'Hypoxia', 'Allograft Rejection', 'IL-6/JAK/STAT3 Signaling',
            'Inflammatory Response', 'Complement', 'Myogenesis'
            
            'Complement', 'Fatty Acid Metabolism', 'Androgen Response'
            
            
            'mTORC1 Signaling', 'UV Response Up', 'Epithelial Mesenchymal Transition',
                                                'p53 Pathway', 'Unfolded Protein Response', 'Hypoxia', 'Apoptosis',
                                                'Interferon Gamma Response'
        '''

        # samples_to_change_idx = get_well_classified_samples(df_merged, CMS_to_affect, num_samples, params)

        # get some random samples of a certain CMS
        indices_cms = df_merged.loc[df_merged.fin_dec == CMS_to_affect].index.values
        samples_to_change_idx = random.sample(list(indices_cms), num_samples)

        X_train = df_merged.loc[
            ~df_merged.index.isin(samples_to_change_idx), df_merged.columns.difference(
                ['sample', 'dataset', 'CMS_network',
                 'CMS_RFclassifier', 'fin_dec',
                 'cat_code'])]
        X_test = df_merged.loc[
            df_merged.index.isin(samples_to_change_idx), df_merged.columns.difference(
                ['sample', 'dataset', 'CMS_network',
                 'CMS_RFclassifier', 'fin_dec',
                 'cat_code'])]
        y_train = df_merged.iloc[~df_merged.index.isin(samples_to_change_idx)].cat_code
        y_test = df_merged.iloc[df_merged.index.isin(samples_to_change_idx)].cat_code

        # train the regressor
        clf = lgb.LGBMRegressor(**params)
        clf = clf.fit(X_train, y_train)

        regulated_genes_cummulative_pos = get_genes_to_regulate(pathways_to_be_up, upregulated_pathways_sc,
                                                                downregulated_pathways_sc)
        if regulated_genes_cummulative_pos:
            overlapping_genes_pos, upper_bounds_pos, lower_bounds_pos = \
                get_ranges(df_merged, regulated_genes_cummulative_pos)

        regulated_genes_cummulative_neg = get_genes_to_regulate(pathways_to_be_down, upregulated_pathways_sc,
                                                                downregulated_pathways_sc)
        if regulated_genes_cummulative_neg:
            overlapping_genes_neg, upper_bounds_neg, lower_bounds_neg = \
                get_ranges(df_merged, regulated_genes_cummulative_neg)

        all_preds_sub = np.zeros((2 * num_samples - 1, 4, len(y_test)))
        all_preds = np.zeros((2 * num_samples - 1, 4, len(y_test)))

        for idx in range(len(y_test)):
            if regulated_genes_cummulative_pos:
                L_upper_distances_pos = (upper_bounds_pos - X_test.iloc[idx][overlapping_genes_pos])
                L_lower_distances_pos = (lower_bounds_pos - X_test.iloc[idx][overlapping_genes_pos])

                regulatory_values_up_pos = np.linspace(0, L_upper_distances_pos, num_samples)
                regulatory_values_down_pos = np.linspace(L_lower_distances_pos, 0, num_samples)

                regulatory_values_pos = np.concatenate((regulatory_values_down_pos[:-1, :], regulatory_values_up_pos),
                                                       axis=0)

                num_values = len(regulatory_values_pos)

            if regulated_genes_cummulative_neg:
                L_upper_distances_neg = (upper_bounds_neg - X_test.iloc[idx][overlapping_genes_neg])
                L_lower_distances_neg = (lower_bounds_neg - X_test.iloc[idx][overlapping_genes_neg])

                regulatory_values_up_neg = np.linspace(L_upper_distances_neg, 0, num_samples)
                regulatory_values_down_neg = np.linspace(0, L_lower_distances_neg, num_samples)

                regulatory_values_neg = np.concatenate((regulatory_values_up_neg[:-1, :], regulatory_values_down_neg),
                                                       axis=0)
                num_values = len(regulatory_values_neg)

            X_syn = X_test.iloc[idx]
            predictions = np.zeros((2 * num_samples - 1, 4))

            for value_idx in range(num_values):
                X_syn = X_test.iloc[idx].copy()
                if regulated_genes_cummulative_pos:
                    X_syn[overlapping_genes_pos] = X_syn[overlapping_genes_pos] + regulatory_values_pos[value_idx]
                if regulated_genes_cummulative_neg:
                    X_syn[overlapping_genes_neg] = X_syn[overlapping_genes_neg] + regulatory_values_neg[value_idx]

                prediction = clf.predict(X_syn.values.reshape(1, -1))

                predictions[value_idx, :] = prediction
                if regulated_genes_cummulative_pos:
                    if regulatory_values_pos[value_idx][0] == 0:
                        base_predicton = prediction
                if regulated_genes_cummulative_neg:
                    if regulatory_values_neg[value_idx][0] == 0:
                        base_predicton = prediction

            all_preds_sub[:, :, idx] = predictions - base_predicton
            all_preds[:, :, idx] = predictions

        # calculate the summarised results
        average_preds = all_preds.mean(axis=2)
        std_preds = all_preds.std(axis=2)

        average_preds_sub = all_preds_sub.mean(axis=2)
        std_preds_sub = all_preds_sub.std(axis=2)

        # create the plots showing the effect of the
        create_plots('mixed', CMS_to_affect, 'mex', 'bi', num_samples,
                     average_preds, std_preds)
        # create_plots('mixed', CMS_to_affect, 'mex',
        #              'bi', num_samples, average_preds_sub, std_preds_sub)

    #
    # accuracies = []
    # kf = KFold(n_splits=5, shuffle=True)
    # for train_index, test_index in kf.split(X):
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #
    #     clf = lgb.LGBMRegressor(**params)
    #     clf = clf.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=20, verbose=5)
    #     predictions = clf.predict(X_test)
    #
    #     accuracy = accuracy_score(predictions, y_test) * 100
    #     print("We classify with the following accuracy: ", accuracy, "%")
    #     accuracies.append(accuracy)
    #
    # print('The different accuracies are: ', accuracies)
    # print('Thus the average accuracy: {} +- {}'.format(np.mean(accuracies), np.std(accuracies)))
    #
