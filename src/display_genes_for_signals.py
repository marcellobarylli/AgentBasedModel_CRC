"""
This file contains the functions for displaying the distributions per CMS for the signalling molecules included in
the model
Author: Robin van den Berg
Contact: rvdb7345@gmail.com
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


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

print(df_merged)

# df_merged[['3558']]

def get_signal_exp_hist(signal, map_id_signal):

    signal_CMS1 = df_merged.loc[df_merged.fin_dec == 'CMS1', signal]
    signal_CMS2 = df_merged.loc[df_merged.fin_dec == 'CMS2', signal]
    signal_CMS3 = df_merged.loc[df_merged.fin_dec == 'CMS3', signal]
    signal_CMS4 = df_merged.loc[df_merged.fin_dec == 'CMS4', signal]

    plt.figure()
    plt.title('{} expression distribution'.format(map_id_signal[signal]))
    signal_CMS1.hist(bins=20, label='CMS1', density=1, edgecolor='green', histtype='step', linewidth=2)
    signal_CMS2.hist(bins=20, label='CMS2', density=1, edgecolor='red', histtype='step', linewidth=2)
    signal_CMS3.hist(bins=20, label='CMS3', density=1, edgecolor='black', histtype='step', linewidth=2)
    signal_CMS4.hist(bins=20, label='CMS4', density=1, edgecolor='blue', histtype='step', linewidth=2)
    plt.legend()
    plt.savefig('signal_expression_distributions/hist_{}.pdf'.format(map_id_signal[signal]))
    plt.show()
map_id_signal = {'3569': 'IL6', '16183': 'IL2', '1950': 'EGF', '7472': 'WNT2', '4609': 'MYC', '6469': 'SHH',
                 '7040': 'TGFb', '2247': 'FGF2', '3082': 'HGF', '5468': 'PPARG', '3458': 'IFNg'}

signal = '3458'
get_signal_exp_hist(signal, map_id_signal)

