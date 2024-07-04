import os

import pandas as pd
# from scipy.spatial.distance import jaccard
import numpy as np


def jaccard(a, b):
    """ Calculates the Jaccard distance between two lists """
    a = set(a)
    b = set(b)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def intersection(lst1, lst2): 
    ''' Retuns the intersection of two lists '''
    return list(set(lst1) & set(lst2)) 

# set the path to this file (important for both windows and mac users to use the same code.
dir_path = os.path.dirname(os.path.realpath(__file__))

# get most important features
method = 'decision_tree'
df_import_feats = pd.read_table(os.path.join(dir_path, 'important_features/important_features_' + method + '.csv'),
                                sep=',')
import_feats = df_import_feats.head(28)['geneid'].values
import_feats_dec_tree = import_feats.astype(str)

# get most important features
method = 'lgbm'
df_import_feats = pd.read_table(os.path.join(dir_path, 'important_features/important_features_' + method + '.csv'),
                                sep=',')
import_feats = df_import_feats.head(28)['geneid'].values
import_feats_lgbm = import_feats.astype(str)


print('The Jaccard distance between the two different most important features sets: ',
      jaccard(import_feats_dec_tree, import_feats_lgbm))
print('The matching genes are: ', intersection(import_feats_dec_tree, import_feats_lgbm))





