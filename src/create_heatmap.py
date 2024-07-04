import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))


# load the dataframes
df_crc_data = pd.read_table(os.path.join(dir_path, '../data/formatted_crc_data.txt'))
df_crc_data['sample'] = df_crc_data.index

df_labels = pd.read_table(os.path.join(dir_path, '../data/cms_labels_public_all.txt'))
df_merged = pd.merge(df_crc_data, df_labels, on='sample')
df_merged.rename({'CMS_final_network_plus_RFclassifier_in_nonconsensus_samples': 'fin_dec'}, axis='columns',
                 inplace=True)
df_merged.drop(['dataset', 'CMS_network', 'CMS_RFclassifier', 'sample'], axis=1, inplace=True)


# average per groupv
df_merged[df_merged.columns.difference(['fin_dec'])] = df_merged[df_merged.columns.difference(['fin_dec'])].astype(float)
print(df_merged)
df_merged_std = df_merged.groupby('fin_dec', as_index=False)[df_merged.columns.difference(['fin_dec'])].apply(np.std)
df_merged = df_merged.groupby('fin_dec', as_index=False)[df_merged.columns.difference(['fin_dec'])].mean()
df_merged.set_index('fin_dec', inplace=True)


# get most important features
method = 'random_forest'
df_import_feats = pd.read_table(os.path.join(dir_path, 'important_features/important_features_' + method + '.csv'),
                                sep=',')
import_feats = df_import_feats.head(15)['geneid'].values
import_feats = import_feats.astype(str)

# because the average expression is not the same
df_merged_norm = df_merged - df_merged.mean()
df_merged_norm = df_merged_norm.transpose()

# plot the heat map of the average expressions
plt.figure()
sns.heatmap(df_merged_norm.loc[import_feats], annot=True)
plt.xlabel('CMS subtype', fontsize=14)
plt.ylabel('Entrez id', fontsize=14)
plt.savefig(os.path.join(dir_path, '../figures/heatmap_' + method + '.png'), dpi=300)
plt.show()

# because the average expression is not the same
df_merged_std = df_merged_std.transpose()
plt.figure()
sns.heatmap(df_merged_std.loc[import_feats], annot=True)
plt.xlabel('CMS subtype', fontsize=14)
plt.ylabel('Entrez id', fontsize=14)
plt.savefig(os.path.join(dir_path, '../figures/heatmap_std_' + method + '.png'), dpi=300)
plt.show()
