import os
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

dir_path = os.path.dirname(os.path.realpath(__file__))


df_crc_data = pd.read_table(os.path.join(dir_path, '../data/formatted_crc_data.txt'))
df_crc_data['sample'] = df_crc_data.index

df_labels = pd.read_table(os.path.join(dir_path, '../data/cms_labels_public_all.txt'))


# print(len(intersection(df_crc_data['sample'].values, df_labels['sample'].values)))

df_merged = pd.merge(df_crc_data, df_labels, on='sample')
df_merged.rename({'CMS_final_network_plus_RFclassifier_in_nonconsensus_samples': 'fin_dec'}, axis='columns',
                 inplace=True)
df_merged.fin_dec = pd.Categorical(df_merged.fin_dec)
df_merged['cat_code'] = df_merged.fin_dec.cat.codes

# drop the samples without labels
df_merged = df_merged.loc[df_merged.fin_dec != 'NOLBL']


clf = LinearDiscriminantAnalysis(n_components=3, store_covariance=True)

score = cross_val_score(clf, df_merged[df_merged.columns.difference(['sample', 'dataset', 'CMS_network',
                        'CMS_RFclassifier', 'fin_dec', 'cat_code'])].values,
                        df_merged.cat_code.values,
                        cv=3, n_jobs=-1)
clf.fit(df_merged[df_merged.columns.difference(['sample', 'dataset', 'CMS_network',
                        'CMS_RFclassifier', 'fin_dec', 'cat_code'])].values,
        df_merged.cat_code.values)

print(score)
print(clf.explained_variance_ratio_)
print(clf.covariance_)
