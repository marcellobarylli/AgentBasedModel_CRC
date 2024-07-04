import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, ttest_ind, ttest_rel, f_oneway

dir_path = os.path.dirname(os.path.realpath(__file__))

# load the dataframes
df_crc_data = pd.read_table(os.path.join(dir_path, '../data/formatted_crc_data.txt'))
df_crc_data['sample'] = df_crc_data.index

df_labels = pd.read_table(os.path.join(dir_path, '../data/cms_labels_public_all.txt'))
df_merged = pd.merge(df_crc_data, df_labels, on='sample')
df_merged.rename({'CMS_final_network_plus_RFclassifier_in_nonconsensus_samples': 'fin_dec'}, axis='columns',
                 inplace=True)
df_merged.drop(['dataset', 'CMS_network', 'CMS_RFclassifier', 'sample'], axis=1, inplace=True)
df_merged[df_merged.columns.difference(['fin_dec'])] = df_merged[df_merged.columns.difference(['fin_dec'])].astype(float)

# drop the samples without labels
df_merged = df_merged.loc[df_merged.fin_dec != 'NOLBL']

# get most important features
method = 'decision_tree'
df_import_feats = pd.read_table(os.path.join(dir_path, 'important_features/important_features_' + method + '.csv'),
                                sep=',')
import_feats = df_import_feats.head(15)['geneid'].values
import_feats = import_feats.astype(str)

import_feats = ['6696']
# show expression rates in the same plot
nbins = 30
for feature in import_feats:
    low = min(df_merged[feature])
    high = max(df_merged[feature])
    bin_edges = np.linspace(low, high, nbins + 1)
    
    plt.figure()
    cms_dists = []
    for idx, group in df_merged.groupby('fin_dec'):
        n, _, _ = plt.hist(group[feature], label=str(group.fin_dec.values[0]), density=True, alpha=0.5, bins=bin_edges)
        cms_dists.append(n)
    plt.legend()
    plt.grid()
    plt.ylabel('frequency', fontsize=14)
    plt.xlabel('Gene expression', fontsize=14)
    plt.title(feature)
    plt.tight_layout()
    if not os.path.exists(os.path.join(dir_path, '../figures/hist_feat_sep_' + method)):
        os.makedirs(os.path.join(dir_path, '../figures/hist_feat_sep_' + method))
    plt.savefig(os.path.join(dir_path, '../figures/hist_feat_sep_' + method + '/hist_' + feature + '.png'), dpi=300)
    plt.show()

    # carry out kolmogorov-Smirnov test and ttest to evaluate the difference between the distributions
    cms1_data = df_merged.loc[df_merged.fin_dec == 'CMS1'][feature]
    cms2_data = df_merged.loc[df_merged.fin_dec == 'CMS2'][feature]
    cms3_data = df_merged.loc[df_merged.fin_dec == 'CMS3'][feature]
    cms4_data = df_merged.loc[df_merged.fin_dec == 'CMS4'][feature]

    kolsmir_12, ttest_12 = ks_2samp(cms1_data, cms2_data), ttest_ind(cms1_data, cms2_data, equal_var=False)
    kolsmir_13, ttest_13 = ks_2samp(cms1_data, cms3_data), ttest_ind(cms1_data, cms3_data, equal_var=False)
    kolsmir_14, ttest_14 = ks_2samp(cms1_data, cms4_data), ttest_ind(cms1_data, cms4_data, equal_var=False)
    kolsmir_23, ttest_23 = ks_2samp(cms2_data, cms3_data), ttest_ind(cms2_data, cms3_data, equal_var=False)
    kolsmir_24, ttest_24 = ks_2samp(cms2_data, cms4_data), ttest_ind(cms2_data, cms4_data, equal_var=False)
    kolsmir_34, ttest_34 = ks_2samp(cms3_data, cms4_data), ttest_ind(cms3_data, cms4_data, equal_var=False)

    print('For gene {}, the differences between the distributions are: '.format(feature))
    print('CMS1, CMS2: \t kolsmir p: {:.2e}, \t ttest p: {:.2e} '.format( kolsmir_12[1], ttest_12[1]))
    print('CMS1, CMS3: \t kolsmir p: {:.2e}, \t ttest p: {:.2e} '.format( kolsmir_13[1], ttest_13[1]))
    print('CMS1, CMS4: \t kolsmir p: {:.2e}, \t ttest p: {:.2e} '.format( kolsmir_14[1], ttest_14[1]))
    print('CMS2, CMS3: \t kolsmir p: {:.2e}, \t ttest p: {:.2e} '.format( kolsmir_23[1], ttest_23[1]))
    print('CMS2, CMS4: \t kolsmir p: {:.2e}, \t ttest p: {:.2e} '.format( kolsmir_24[1], ttest_24[1]))
    print('CMS3, CMS4: \t kolsmir p: {:.2e}, \t ttest p: {:.2e} '.format( kolsmir_34[1], ttest_34[1]))

# show expression rates in different plots
df_merged[import_feats[0]].hist(by=df_merged['fin_dec'])
plt.show()