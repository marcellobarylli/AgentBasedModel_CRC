import os
import csv
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import RandomizedSearchCV, HalvingRandomSearchCV


dir_path = os.path.dirname(os.path.realpath(__file__))
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin'


def create_cluster_coloured_graph_aggl(df):
    """
    show scatter plot coloured according to respective cluster
    :param df: dataframe to cluster
    :param num_clusters: number of clusters to divide the
    :param columns_to_plot:
    :return:
    """
    agglomerativeModel = AgglomerativeClustering(n_clusters=2 ** 3, affinity='euclidean', linkage='ward',
                                                 distance_threshold=None)
    df['agglomerative'] = agglomerativeModel.fit_predict(df[df.columns.difference(['sample', 'dataset', 'CMS_network',
                                                                                   'CMS_RFclassifier', 'fin_dec'])])

    return df


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
# df_merged_cat = create_cluster_coloured_graph_aggl(df_merged)

# setting the parameters for our lgbm model
# params = {
#     'task': 'train', 'boosting_type': 'dart', 'objective': 'multiclass', 'num_class': 4,
#     'min_data': 50, 'verbose': -1, 'max_depth': 10, 'bagging_fraction': 0.95, 'feature_fraction': 0.99,
#     'num_leaves': 10, 'metric': ['multi_error', 'multi_logloss']
# }



rs_params = {
        'bagging_fraction': np.linspace(0.7, 0.999, 10),
        'colsample_bytree': np.linspace(0.5, 0.999, 10),
        'max_depth': np.arange(3, 15, 1),
        'min_child_samples': np.arange(15, 120, 5),
        'num_leaves': np.arange(5, 100, 5),
        'n_estimators': np.arange(50, 1000, 50),
    'is_unbalance': [False, True]
}

# Initialize a RandomizedSearchCV object using 5-fold CV-
rs_cv = RandomizedSearchCV(estimator=lgb.LGBMClassifier(), param_distributions=rs_params, cv=3, n_iter=20, verbose=5)

X = df_merged[df_merged.columns.difference(['sample', 'dataset', 'CMS_network',
                                            'CMS_RFclassifier', 'fin_dec',
                                            'cat_code'])].values
y = df_merged.cat_code.values

# Train on training data-
results = rs_cv.fit(X, y, verbose=5)
print('These are the results of the gridsearch: ',  results.best_params_)

w = csv.writer(open("results of random search.csv", "w"))
for key, val in results.best_params_.items():
    w.writerow([key, val])

# # create a dataset structure used for the lgbm classifier
# x_train = df_merged_cat[df_merged_cat.columns.difference(['sample', 'dataset', 'CMS_network',
#                         'CMS_RFclassifier', 'fin_dec', 'agglomerative', 'cat_code'])].values
# y_train = df_merged_cat.agglomerative.values

# lgb_train = lgb.Dataset(X_train, y_train,
#                         feature_name = list(df_merged_cat[df_merged_cat.columns.difference(['sample',
#                                                      'dataset', 'CMS_network', 'CMS_RFclassifier', 'fin_dec',
#                                                      'agglomerative', 'cat_code'])].keys())
# )

# # creating a test-train split
# X_train, X_test, y_train, y_test = train_test_split(
#     df_merged_cat[df_merged_cat.columns.difference(['sample', 'dataset', 'CMS_network',
#                                                     'CMS_RFclassifier', 'fin_dec', 'agglomerative',
#                                                     'cat_code'])].values,
#     df_merged_cat.cat_code.values, test_size=0.33, random_state=42)
#
# # train the classifier and predict the test set
# clf = lgb.LGBMClassifier(**params)
# clf = clf.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=20, verbose=5)
# predictions = clf.predict(X_test)

# calculate the accuracy
# accuracy = accuracy_score(predictions, y_test) * 100
# print("We classify with the following accuracy: ", accuracy, "%")

X = df_merged[df_merged.columns.difference(['sample', 'dataset', 'CMS_network',
                                            'CMS_RFclassifier', 'fin_dec',
                                            'cat_code'])].values
y = df_merged.cat_code.values

accuracies = []
confusion_matrices = np.ndarray((4,4,5))
kf = KFold(n_splits=5, shuffle=True)
idx = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = lgb.LGBMClassifier(**results.best_params_)
    clf = clf.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=30, verbose=10)
    predictions = clf.predict(X_test)

    accuracy = accuracy_score(predictions, y_test) * 100
    confusion = confusion_matrix(predictions, y_test)

    confusion_matrices[:,:, idx] = confusion
    print("We classify with the following accuracy: ", accuracy, "%")
    accuracies.append(accuracy)
    idx += 1

print('The different accuracies are: ', accuracies)
print('Thus the average accuracy: {} +- {}'.format(np.mean(accuracies), np.std(accuracies)))

print('The confusion matrix: ',  confusion_matrices.mean(axis=2))

# actuals_onehot = pd.get_dummies(y_test).values
# false_positive_rate, recall, thresholds = roc_curve(actuals_onehot[0], np.round(predictions)[0])
# roc_auc = auc(false_positive_rate, recall)
# print("AUC score ", roc_auc)

# # plot the importance of the features to show the most informational features
# feats_imports_df = pd.DataFrame(
#     {'geneid': list(df_merged[df_merged.columns.difference(['sample', 'dataset', 'CMS_network',
#                                                     'CMS_RFclassifier', 'fin_dec', 'agglomerative',
#                                                     'cat_code'])].keys()),
#      'importance': clf.feature_importances_
#     })
#
# feats_imports_df.sort_values('importance', inplace=True, ascending=False)
# feats_imports_df.to_csv(os.path.join(dir_path, 'important_features/important_features_lgbm.csv'), index=False)
#
# ax1 = lgb.plot_importance(clf, max_num_features=40)
# plt.show()
#
# # create decision tree
# ax = lgb.create_tree_digraph(clf, format='png')

# print("The depth of the tree: ", clf.decision_path(df_merged_cat[df_merged_cat.columns.difference(['sample', 'dataset', 'CMS_network',
#                         'CMS_RFclassifier', 'fin_dec', 'agglomerative'])].values))
# print("The importance of the different features: ", clf.feature_importances_)
# print("The len of feature importances: ", len(clf.feature_importances_))

#
# plt.figure()
# tree.plot_tree(clf)
# plt.savefig('decision_tree' + str(datetime.now().strftime("%d%m%Y%H%M%S")) + '.png', dpi=600)
# plt.show()
