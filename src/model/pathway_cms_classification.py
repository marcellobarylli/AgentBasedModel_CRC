import os
import ast
import lightgbm as lgb
import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 20)
pd.set_option('display.width', 1700)
pd.set_option('display.max_rows', 500)

dir_path = os.path.dirname(os.path.realpath(__file__))
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

class sample_classifier:
    def __init__(self, location_bulk_dataset, location_pathway_files):

        # set location parameters
        self.location_bulk_dataset = location_bulk_dataset
        self.location_pathway_files = location_pathway_files

        # preprocess dataset
        self.bulk_dataset, self.only_samples_df = self.preprocess_bulk_dataset()
        self.genes_bulk_dataset = list(self.only_samples_df.columns)

        # train or load a classifier on the processed dataset
        model_path = 'classification_model_updated_1.txt'
        if os.path.isfile(model_path):
            self.classifier = lgb.Booster(model_file=model_path)
            # plot = lgb.create_tree_digraph(self.classifier, tree_index=1)
            # plot.render('tree.gv')
            # print(plot)
        else:
            self.classifier = self.train_classifier(model_path)

        # get the upper and lower bound and the average expression for translation with the pathways
        self.gene_upper_bounds, self.gene_lower_bounds, self.gene_average_exp = \
            self.get_average_expression_bulk()

        self.genes_to_regulate_per_pathway = self.get_genes_to_regulated_per_pathway()

        del self.bulk_dataset
        del self.only_samples_df



    def preprocess_bulk_dataset(self):

        # read in the necessary data
        # df_crc_data = pd.read_table(os.path.join(dir_path, '../../data/formatted_crc_data.txt'))
        with open(os.path.join(dir_path, '../../data/formatted_crc_data.txt')) as file:
            df_crc_data = pd.read_table(file)
            df_crc_data['sample'] = df_crc_data.index
            df_labels = pd.read_table(os.path.join(dir_path, '../../data/cms_labels_public_all.txt'))

            # match up the assigned labels with the expression rates
            df_merged = pd.merge(df_crc_data, df_labels, on='sample')
            df_merged.rename({'CMS_final_network_plus_RFclassifier_in_nonconsensus_samples': 'fin_dec'}, axis='columns',
                             inplace=True)

            # drop the samples without labels
            df_merged = df_merged.loc[df_merged.fin_dec != 'NOLBL']

            # set CMS classification strings as category for the regressor
            df_merged.fin_dec = pd.Categorical(df_merged.fin_dec)
            df_merged['cat_code'] = df_merged.fin_dec.cat.codes

            # create a dataframe without all classifications and only the gene expression columns
            only_samples_df = df_merged[df_merged.columns[~df_merged.columns.isin(['sample', 'dataset', 'CMS_network',
                                                                                   'CMS_RFclassifier',
                                                                                   'fin_dec', 'cat_code'])]]

        # # get the upper and lower bounds per cms expression as a test for classifying average cms expression
        # self.only_samples_df_1 = df_merged.loc[df_merged['fin_dec'] == 'CMS1', df_merged.columns[
        #     ~df_merged.columns.isin([
        #     'sample', 'dataset',
        #                                                                               'CMS_network',
        #                                                                        'CMS_RFclassifier',
        #                                                                        'fin_dec', 'cat_code'])]]
        # self.only_samples_df_2 = df_merged.loc[df_merged['fin_dec'] == 'CMS2', df_merged.columns[
        #     ~df_merged.columns.isin([
        #     'sample', 'dataset',
        #                                                                               'CMS_network',
        #                                                                        'CMS_RFclassifier',
        #                                                                        'fin_dec', 'cat_code'])]]
        # self.only_samples_df_3 = df_merged.loc[df_merged['fin_dec'] == 'CMS3', df_merged.columns[
        #     ~df_merged.columns.isin([
        #     'sample', 'dataset',
        #                                                                               'CMS_network',
        #                                                                        'CMS_RFclassifier',
        #                                                                        'fin_dec', 'cat_code'])]]
        # self.only_samples_df_4 = df_merged.loc[df_merged['fin_dec'] == 'CMS4', df_merged.columns[
        #     ~df_merged.columns.isin([
        #     'sample', 'dataset',
        #                                                                               'CMS_network',
        #                                                                        'CMS_RFclassifier',
        #                                                                        'fin_dec', 'cat_code'])]]

        return df_merged, only_samples_df

    def load_pathways_type_specific(self, side, cutoff_p, celltype, scale, tissue_class, date):
        ''' Load the ranked pathway dfs '''

        df_ = \
            pd.read_csv('../../results/pathway_analysis_{}_{}_{}_{}_{}_{}_mapped.csv'.format(side,
                                                                                             cutoff_p,
                                                                                             celltype,
                                                                                             tissue_class,
                                                                                             scale,
                                                                                             date))
        df_.rename(columns={'Genes': 'gene_union', 'Term': 'pathway'}, inplace=True)

        return df_[df_.CMS == 'CMS1'], df_[df_.CMS == 'CMS2'], df_[df_.CMS == 'CMS3'], df_[df_.CMS == 'CMS4']

    def get_genes_to_regulate(self, pathways_to_be_up, upregulated_pathways_sc, downregulated_pathways_sc):
        """ Extracts the genes from a pathway df and checks whether they are also in the bulk df """

        regulated_genes_cummulative = []

        for direction, pathway_dfs in zip(['pos', 'neg'], [upregulated_pathways_sc, downregulated_pathways_sc]):
            for idx, type in enumerate(['CMS1', 'CMS2', 'CMS3', 'CMS4']):
                pathway_df = pathway_dfs[idx].copy()
                pathway_row = pathway_df.loc[pathway_df['pathway'].isin(pathways_to_be_up[direction][type])]
                for gene_list in pathway_row['gene_union'].values:
                    regulated_genes = gene_list.replace(' ', ',')
                    regulated_genes = ast.literal_eval(regulated_genes)

                    regulated_genes_cummulative += [str(gene) for gene in regulated_genes if
                                                    gene in self.genes_bulk_dataset]

        return list(set(regulated_genes_cummulative))

    def get_genes_to_regulated_per_pathway(self):
        """ This function obtains the genes that are to be modified  """

        # self.pathway_name_map = {
        #     "Oxidative Phosphorylation": "oxidative_phosphorylation_activation",
        #     "Glycolysis": "glycolysis_activation",
        #     "p53 Pathway": "P53_activation",
        #     "mTORC1 Signaling": "mTORC1_activation",
        #     "TNF-alpha Signaling via NF-kB": "TNFaviaNFkB_activation",
        #     "Unfolded Protein Response": "unfolded_protein_response_activation",
        #     "Hypoxia": "hypoxia_activation",
        #     "Epithelial Mesenchymal Transition": "EMT_activation",
        #     "Myogenesis": "myogenesis_activation",
        #     "Myc Targets V1": "MYC_target_v1_activation",
        #     "Reactive Oxygen Species Pathway": "ROS_pathway_activation",
        #     "IL-2/STAT5 Signaling": "IL2_STAT5_signalling_activation",
        #     "Pperoxisome": "peroxisome_activation",
        #     "Adipogenesis": "adipogenesis_activation",
        #     "Interferon Gamma Response": "IFN_gamma_activation",
        #     "KRAS Signaling Up": "kras_signalling_activation",
        #     "IL-6/JAK/STAT3 Signaling": "IL6_JAK_activation",
        #     "Complement": "complement_activation",
        #     "Interferon Alpha Response": "interferon_a_pathway_activation",
        #     "PI3K/AKT/mTOR  Signaling": "PI3K_activation"
        # }

        self.pathway_name_map = {
            "p53 Pathway": "P53_activation",
            "mTORC1 Signaling": "mTORC1_activation",
            "TNF-alpha Signaling via NF-kB": "TNFaviaNFkB_activation",
            "Hypoxia": "hypoxia_activation",
            "Epithelial Mesenchymal Transition": "EMT_activation",
            "Myc Targets V1": "MYC_target_v1_activation",
            "Reactive Oxygen Species Pathway": "ROS_pathway_activation",
            "IL-2/STAT5 Signaling": "IL2_STAT5_signalling_activation",
            "Adipogenesis": "adipogenesis_activation",
            "Interferon Gamma Response": "IFN_gamma_activation",
            "KRAS Signaling Up": "kras_signalling_activation",
            "IL-6/JAK/STAT3 Signaling": "IL6_JAK_activation",
            "Complement": "complement_activation",
            "Interferon Alpha Response": "interferon_a_pathway_activation",
            "PI3K/AKT/mTOR  Signaling": "PI3K_activation"
        }


        genes_to_tweak_per_pathway = {}

        for influenced_pathway, mapped_pathway in zip(self.pathway_name_map.keys(), self.pathway_name_map.values()):
            pathways = {'pos': {'CMS1': [influenced_pathway],
                                'CMS2': [influenced_pathway],
                                'CMS3': [influenced_pathway],
                                'CMS4': [influenced_pathway]},
                        'neg': {'CMS1': [influenced_pathway],
                                'CMS2': [influenced_pathway],
                                'CMS3': [influenced_pathway],
                                'CMS4': [influenced_pathway]}
                        }

            genes_to_modify = []

            # loop through all the cell types that are available
            for cell_type in ['T cells', 'B cells', 'Myeloids', 'Epithelial cells', 'Stromal cells']:
                # load the pathway genes
                cms1_ranked_pos, cms2_ranked_pos, cms3_ranked_pos, cms4_ranked_pos = self.load_pathways_type_specific(
                    side='pos',
                    cutoff_p='0025',
                    celltype=cell_type,
                    scale='raw',
                    tissue_class='Tumor',
                    date='20210401')
                cms1_ranked_neg, cms2_ranked_neg, cms3_ranked_neg, cms4_ranked_neg = self.load_pathways_type_specific(
                    side='neg',
                    cutoff_p='0025',
                    celltype=cell_type,
                    scale='raw',
                    tissue_class='Tumor',
                    date='20210401')

                upregulated_pathways_sc = [cms1_ranked_pos, cms2_ranked_pos, cms3_ranked_pos, cms4_ranked_pos]
                downregulated_pathways_sc = [cms1_ranked_neg, cms2_ranked_neg, cms3_ranked_neg, cms4_ranked_neg]

                regulated_genes = self.get_genes_to_regulate(pathways, upregulated_pathways_sc,
                                                             downregulated_pathways_sc)

                genes_to_modify += regulated_genes

            genes_to_tweak_per_pathway[mapped_pathway] = list(set(genes_to_modify))

        return genes_to_tweak_per_pathway

    def train_classifier(self, model_path):
        """ Train the classifier on the consortium dataset """

        # params = {
        #     'task': 'train', 'objective': 'multiclass', 'num_class': 4, 'boosting_type': 'gbdt',
        #     'min_data': 50, 'verbose': -1, 'max_depth': 10, 'bagging_fraction': 0.95, 'feature_fraction': 0.99,
        #     'num_leaves': 10, 'metric': ['multi_error', 'multi_logloss'], 'is_unbalance': True
        # }

        params = {'num_leaves': 20, 'n_estimators': 750, 'min_child_samples': 105, 'max_depth': 12,
                  'is_unbalance': True, 'colsample_bytree': 0.7217777777777777, 'bagging_fraction': 0.9657777777777778}

        clf = lgb.LGBMClassifier(**params)
        clf = clf.fit(self.only_samples_df, self.bulk_dataset['cat_code'])

        # save the trained model if it doesn't already exist
        if not os.path.isfile(model_path):
            clf.booster_.save_model(model_path)

        return clf

    def get_average_expression_bulk(self):
        ''' This function gets the upper and lower bound on the expression range per gene '''

        upper_bounds = self.only_samples_df[
            np.abs(self.only_samples_df - self.only_samples_df.mean()) <= (4 * self.only_samples_df.std())].max(
            axis=0)
        lower_bounds = self.only_samples_df[np.abs(self.only_samples_df - self.only_samples_df.mean()) <= (
                4 * self.only_samples_df.std())].min(axis=0)

        # # calculate the average expressions per CMS individually to not let the present fraction be of influence
        # average_expression_1 = self.only_samples_df[(np.abs(self.only_samples_df - self.only_samples_df.mean()) <= (
        #         4 * self.only_samples_df.std()))][self.bulk_dataset['fin_dec'] == 'CMS1'].mean(axis=0)
        # average_expression_2 = self.only_samples_df[(np.abs(self.only_samples_df - self.only_samples_df.mean()) <= (
        #         4 * self.only_samples_df.std()))][(self.bulk_dataset['fin_dec'] == 'CMS2')].mean(axis=0)
        # average_expression_3 = self.only_samples_df[(np.abs(self.only_samples_df - self.only_samples_df.mean()) <= (
        #         4 * self.only_samples_df.std()))][(self.bulk_dataset['fin_dec'] == 'CMS3')].mean(axis=0)
        # average_expression_4 = self.only_samples_df[(np.abs(self.only_samples_df - self.only_samples_df.mean()) <= (
        #         4 * self.only_samples_df.std()))][(self.bulk_dataset['fin_dec'] == 'CMS4')].mean(axis=0)
        #
        # average_expression = (average_expression_1 + average_expression_2 + average_expression_3 +
        #                       average_expression_4) / 4

        average_expression = self.only_samples_df[(np.abs(self.only_samples_df - self.only_samples_df.mean()) <= (
                4 * self.only_samples_df.std()))].mean(axis=0)

        return upper_bounds, lower_bounds, average_expression

    def get_expression_bounds_average_df(self, df):
        upper_bounds = df[np.abs(df - df.mean()) <= (4 * df.std())].max(
            axis=0)
        lower_bounds = df[np.abs(df - df.mean()) <= (
                4 * df.std())].min(axis=0)
        average_expression = df[np.abs(df - df.mean()) <= (
                4 * df.std())].mean(axis=0)

        return upper_bounds, lower_bounds, average_expression

    def translate_activation_to_gene(self, pathway_activation_dict):
        """
        This function uses the pathway activation values to modify the base gene expression sample
        :param pathway_activation_dict: the dict with the gene expression values
        :return:
        """
        translated_sample = self.gene_average_exp

        for pathway in self.pathway_name_map.values():
            pathway_activation = pathway_activation_dict[pathway]
            if pathway_activation < 0:
                translated_sample.loc[self.genes_to_regulate_per_pathway[pathway]] = \
                    self.gene_average_exp[self.genes_to_regulate_per_pathway[pathway]] - \
                    (self.gene_average_exp[self.genes_to_regulate_per_pathway[pathway]] -
                     self.gene_lower_bounds[self.genes_to_regulate_per_pathway[pathway]]) * abs(
                        pathway_activation)

            elif pathway_activation > 0:
                translated_sample[self.genes_to_regulate_per_pathway[pathway]] = \
                    self.gene_average_exp[self.genes_to_regulate_per_pathway[pathway]] + \
                    (self.gene_upper_bounds[self.genes_to_regulate_per_pathway[pathway]] -
                     self.gene_average_exp[self.genes_to_regulate_per_pathway[pathway]]) * \
                    abs(pathway_activation)

        return translated_sample

    def classify_sample(self, pathway_activation):
        '''
        This function classifies a sample based on the pathway activation values
        :param pathway_activation: an array containing activation values for the pathways as given in the model
        :return: array featuring the regression values for the classification
        '''

        gene_exp_sample = self.translate_activation_to_gene(pathway_activation)

        classification = self.classifier.predict(gene_exp_sample.to_numpy().reshape(1, -1))
        print('classification of sample: ', classification)

        return classification


classifier_obj = sample_classifier('test', 'test')

test_sample = {'oxidative_phosphorylation_activation': 1.0, 'glycolysis_activation': 0.012798507564009574,
               'P53_activation': -1.0, 'mTORC1_activation': 1.0, 'TNFaviaNFkB_activation': -1.0,
               'unfolded_protein_response_activation': 0.0, 'hypoxia_activation': 1.0, 'EMT_activation': 1.0,
               'myogenesis_activation': 0.3055922491099715, 'MYC_target_v1_activation': 0.0,
               'ROS_pathway_activation': 0.0, 'IL2_STAT5_signalling_activation': 0.16649808761497747,
               'peroxisome_activation': 0.0, 'adipogenesis_activation': -0.30064683867264197,
               'IFN_gamma_activation': -1.0, 'kras_signalling_activation': 0.23428235676357206,
               'IL6_JAK_activation': -1.0, 'complement_activation': 0.18627475799534954,
               'interferon_a_pathway_activation': -1.0, 'PI3K_activation': -1.0}

# sim_after_while = {'oxidative_phosphorylation_activation': 1.0, 'glycolysis_activation': 0.0027071865053970136, 'P53_activation': -1.0, 'mTORC1_activation': 1.0, 'TNFaviaNFkB_activation': -0.9999999999999997, 'unfolded_protein_response_activation': 0.0, 'hypoxia_activation': 1.0, 'EMT_activation': 1.0, 'myogenesis_activation': 0.20659897970968716, 'MYC_target_v1_activation': 0.0, 'ROS_pathway_activation': 2.513215972781095e-16, 'IL2_STAT5_signalling_activation': 0.1390851732779388, 'peroxisome_activation': 0.0, 'adipogenesis_activation': 1.0, 'IFN_gamma_activation': -1.0, 'kras_signalling_activation': 0.1688812186145844, 'IL6_JAK_activation': -1.0, 'complement_activation': -0.32952832269120713, 'interferon_a_pathway_activation': -1.0, 'PI3K_activation': -1.0}


cms1_starting_configuration = {'oxidative_phosphorylation_activation': 0.044877203454510274,
                               'glycolysis_activation': 0.08622464500239845, 'P53_activation': 0.06549281846791855,
                               'mTORC1_activation': 0.1300658171599277, 'TNFaviaNFkB_activation': 0.21515511772282186,
                               'unfolded_protein_response_activation': 0.04696836762470545,
                               'hypoxia_activation': 0.11490037903442071, 'EMT_activation': 0.07166637493994778,
                               'myogenesis_activation': 0.009606017050475011,
                               'MYC_target_v1_activation': 0.08351238070220662,
                               'ROS_pathway_activation': 0.0253844079139689,
                               'IL2_STAT5_signalling_activation': 0.12261412833344815,
                               'peroxisome_activation': 0.012811720720966794,
                               'adipogenesis_activation': 0.041327680316779705,
                               'IFN_gamma_activation': 0.31033305672740064,
                               'kras_signalling_activation': 0.08713419751495276,
                               'IL6_JAK_activation': 0.0799669781401864, 'complement_activation': 0.14853786603497332,
                               'interferon_a_pathway_activation': 0.1531965710458719,
                               'PI3K_activation': 0.03021702553440537}
cms2_starting_configuration = {'oxidative_phosphorylation_activation': 0.025634223890701482,
                               'glycolysis_activation': -0.018457832097650423, 'P53_activation': -0.044306962324481884,
                               'mTORC1_activation': 0.0006538447634008688,
                               'TNFaviaNFkB_activation': -0.15518171952202287,
                               'unfolded_protein_response_activation': 0.001545898674497624,
                               'hypoxia_activation': -0.0904996716838063, 'EMT_activation': -0.17363175456592403,
                               'myogenesis_activation': -0.06642315816586446,
                               'MYC_target_v1_activation': 0.08194528849252485,
                               'ROS_pathway_activation': -0.0018147051204457528,
                               'IL2_STAT5_signalling_activation': -0.09592644178815296,
                               'peroxisome_activation': 0.00371351898452254,
                               'adipogenesis_activation': -0.015580043049140903,
                               'IFN_gamma_activation': -0.14957779203349877,
                               'kras_signalling_activation': -0.06469273262242185,
                               'IL6_JAK_activation': -0.03960663618100062,
                               'complement_activation': -0.09512382264582173,
                               'interferon_a_pathway_activation': -0.06512603531970204,
                               'PI3K_activation': -0.014012897046747028}
cms3_starting_configuration = {'oxidative_phosphorylation_activation': 0.004718150731261089, 'glycolysis_activation':
    -0.010315762529599927, 'P53_activation': -0.057712960041302164, 'mTORC1_activation': -0.07586478337619507,
                               'TNFaviaNFkB_activation': -0.1649894343308381,
                               'unfolded_protein_response_activation': -0.030891047765189437,
                               'hypoxia_activation': -0.10650980449017977, 'EMT_activation': -0.4527452993578811,
                               'myogenesis_activation': -0.11849566107214464,
                               'MYC_target_v1_activation': -0.06436219937367028,
                               'ROS_pathway_activation': -0.00044305903450670564,
                               'IL2_STAT5_signalling_activation': -0.07753112778930657,
                               'peroxisome_activation': -0.002719058068003497,
                               'adipogenesis_activation': -0.043314911756427425,
                               'IFN_gamma_activation': -0.19149359741510041,
                               'kras_signalling_activation': -0.09292864340490999,
                               'IL6_JAK_activation': -0.04122831884530266, 'complement_activation': -0.1385145584580805,
                               'interferon_a_pathway_activation': -0.08694074046606154,
                               'PI3K_activation': -0.0029747241713828005}
cms4_starting_configuration = {'oxidative_phosphorylation_activation': -0.08563391142025031,
                               'glycolysis_activation': 0.003620748346920589, 'P53_activation': 0.06609632411888636,
                               'mTORC1_activation': -0.03645808597814066, 'TNFaviaNFkB_activation': 0.2792908397785773,
                               'unfolded_protein_response_activation': -0.009888786925881356,
                               'hypoxia_activation': 0.1728767628634149, 'EMT_activation': 0.6451268811110097,
                               'myogenesis_activation': 0.21141792598658682,
                               'MYC_target_v1_activation': -0.1690354861740059,
                               'ROS_pathway_activation': -0.010990034535492797,
                               'IL2_STAT5_signalling_activation': 0.14209894053726352,
                               'peroxisome_activation': -0.011309171009003641,
                               'adipogenesis_activation': 0.03707278003860049,
                               'IFN_gamma_activation': 0.21054303255301055,
                               'kras_signalling_activation': 0.17329851024524984,
                               'IL6_JAK_activation': 0.06633320553948664, 'complement_activation': 0.2099422545205,
                               'interferon_a_pathway_activation': 0.07167464985182195,
                               'PI3K_activation': 0.003028252767371499}

# classifier_obj.classify_sample(cms1_starting_configuration)
# classifier_obj.classify_sample(cms2_starting_configuration)
# classifier_obj.classify_sample(cms3_starting_configuration)
# classifier_obj.classify_sample(cms4_starting_configuration)


# for i in range(10):
#     random_sample = test_sample
#     for key in random_sample.keys():
#         random_sample[key] = 1
#     classifier_obj.classify_sample(random_sample)
