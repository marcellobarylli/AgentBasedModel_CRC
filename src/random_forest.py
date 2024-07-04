import os
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier

dir_path = os.path.dirname(os.path.realpath(__file__))


def create_cluster_coloured_graph_aggl(df):
    """
    show scatter plot coloured according to respective cluster
    :param df: dataframe to cluster
    :param num_clusters: number of clusters to divide the
    :param columns_to_plot:
    :return:
    """
    agglomerativeModel = AgglomerativeClustering(n_clusters=2**2, affinity='euclidean', linkage='ward',
                                                 distance_threshold=None)
    df['agglomerative'] = agglomerativeModel.fit_predict(df[df.columns.difference(['sample', 'dataset', 'CMS_network',
                                                                                   'CMS_RFclassifier', 'fin_dec',
                                                                                   'cat_code'])])

    return  df


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

df_merged_cat = create_cluster_coloured_graph_aggl(df_merged)

clf = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=5000)

score = cross_val_score(clf, df_merged_cat[df_merged_cat.columns.difference(['sample', 'dataset', 'CMS_network',
                        'CMS_RFclassifier', 'fin_dec', 'agglomerative', 'cat_code'])].values,
                        df_merged_cat.cat_code.values,
                        cv=5)
clf.fit(df_merged_cat[df_merged_cat.columns.difference(['sample', 'dataset', 'CMS_network',
                        'CMS_RFclassifier', 'fin_dec', 'agglomerative', 'cat_code'])].values,
        df_merged_cat.cat_code.values)

print('Thus the average accuracy: {} +- {}'.format(np.mean(score), np.std(score)))
print("The depth of the tree: ", clf.decision_path(df_merged_cat[df_merged_cat.columns.difference(['sample',
                        'dataset', 'CMS_network',
                        'CMS_RFclassifier', 'fin_dec', 'agglomerative', 'cat_code'])].values))

# plot the importance of the features to show the most informational features
feats_imports_df = pd.DataFrame(
    {'geneid': list(df_merged_cat[df_merged_cat.columns.difference(['sample', 'dataset', 'CMS_network',
                                                    'CMS_RFclassifier', 'fin_dec', 'agglomerative',
                                                    'cat_code'])].keys()),
     'importance': clf.feature_importances_
    })

feats_imports_df.sort_values('importance', inplace=True, ascending=False)
feats_imports_df.to_csv(os.path.join(dir_path, 'important_features/important_features_random_forest.csv'),
                        index=False)