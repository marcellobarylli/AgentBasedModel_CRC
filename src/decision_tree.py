import os
import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from datetime import datetime
from sklearn.model_selection import cross_val_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

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

# load the data
df_crc_data = pd.read_table(os.path.join(dir_path, '../data/formatted_crc_data.txt'))
df_crc_data['sample'] = df_crc_data.index
df_labels = pd.read_table(os.path.join(dir_path, '../data/cms_labels_public_all.txt'))

# match up the assigned labels with the expression rates
df_merged = pd.merge(df_crc_data, df_labels, on='sample')
df_merged.rename({'CMS_final_network_plus_RFclassifier_in_nonconsensus_samples': 'fin_dec'}, axis='columns',
                 inplace=True)
df_merged.fin_dec = pd.Categorical(df_merged.fin_dec)
df_merged['cat_code'] = df_merged.fin_dec.cat.codes

# drop the samples without labels
df_merged = df_merged.loc[df_merged.fin_dec != 'NOLBL']

# create labels based on hierarchical clustering
df_merged_cat = create_cluster_coloured_graph_aggl(df_merged)

# score en fit the decision tree classifier
clf = DecisionTreeClassifier(random_state=42, max_depth=15, min_samples_split=10, min_samples_leaf=10)
score = cross_val_score(clf, df_merged_cat[df_merged_cat.columns.difference(['sample', 'dataset', 'CMS_network',
                        'CMS_RFclassifier', 'fin_dec', 'agglomerative', 'cat_code'])].values, df_merged_cat.cat_code.values,
                        cv=5, n_jobs=-1, verbose=1)
clf.fit(df_merged_cat[df_merged_cat.columns.difference(['sample', 'dataset', 'CMS_network',
                        'CMS_RFclassifier', 'fin_dec', 'agglomerative', 'cat_code'])].values, df_merged_cat.cat_code.values)

print("The score per fold: ", score)
print('Thus the average accuracy: {} +- {}'.format(np.mean(score), np.std(score)))
print("The depth of the tree: ", clf.get_depth())
print("The importance of the different features: ", clf.feature_importances_)
print("The len of feature importances: ", len(clf.feature_importances_))

# plot the tree to identify the important genes
ax = plt.subplot()
plt.tight_layout()
tree.plot_tree(clf, class_names=True, ax=ax)
plt.savefig('decision_tree' + str(datetime.now().strftime("%d%m%Y%H%M%S")) + '.pdf', bbox_inches='tight')
plt.show()

# plot the importance of the features to show the most informational features
feats_imports_df = pd.DataFrame(
    {'geneid': list(df_merged_cat[df_merged_cat.columns.difference(['sample', 'dataset', 'CMS_network',
                                                    'CMS_RFclassifier', 'fin_dec', 'agglomerative',
                                                    'cat_code'])].keys()),
     'importance': clf.feature_importances_
    })

feats_imports_df.sort_values('importance', inplace=True, ascending=False)
feats_imports_df.to_csv(os.path.join(dir_path, 'important_features/important_features_decision_tree.csv'),
                        index=False)