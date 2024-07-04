import pandas as pd
import mygene
# from PyEntrezId import Conversion1

from tqdm import tqdm

def Intersection(lst1, lst2):
    return set(lst1).intersection(lst2)

''' Create the correct stuff for deconvolution of the single cell data'''


# CMStype = 'CMS1'

for CMStype in ['CMS1', 'CMS2', 'CMS3', 'CMS4']:
    print('Starting to load the sc gene expression data')
    sampled_expression_df = pd.read_csv('../data/selected_sc_gene_express_Tumor_raw.csv', index_col=[0], header=[0, 1, 2])

    # add the bulk classification
    list_of_cms1_patients = ['SMC03', 'SMC06', 'SMC08', 'SMC10', 'SMC15']
    list_of_cms2_patients = ['SMC07', 'SMC09', 'SMC11', 'SMC18', 'SMC21', 'SMC22', 'SMC23', 'SMC25']
    list_of_cms3_patients = ['SMC01', 'SMC05', 'SMC16', 'SMC19']
    list_of_cms4_patients = ['SMC02', 'SMC04', 'SMC14', 'SMC17', 'SMC20', 'SMC24']

    if CMStype == 'CMS1':
        sampled_expression_df = sampled_expression_df[list_of_cms1_patients]
    if CMStype == 'CMS2':
        sampled_expression_df = sampled_expression_df[list_of_cms2_patients]
    if CMStype == 'CMS3':
        sampled_expression_df = sampled_expression_df[list_of_cms3_patients]
    if CMStype == 'CMS4':
        sampled_expression_df = sampled_expression_df[list_of_cms4_patients]

    print(sampled_expression_df)
    # assert False, print('This is my stop sign')

    # load the dataframe with ids mapped online
    already_mapped_ids = pd.read_csv('../data/hgnc-symbol-check.csv')
    already_mapped_ids['HGNC ID'] = already_mapped_ids['HGNC ID'].str.lower()
    already_mapped_ids['HGNC ID'] = already_mapped_ids['HGNC ID'].apply(lambda x: str(x).replace('hgnc:', ''))
    already_mapped_ids.dropna(axis=0, inplace=True)


    # map the hgnc symbols to HGNC IDs
    mapping_df = already_mapped_ids[['Input', 'HGNC ID']].set_index('Input')
    sampled_expression_df.index = sampled_expression_df.index.map(mapping_df.to_dict()['HGNC ID'])


    # map the genes from the single cell data from hgnc id to entrezgene
    mg = mygene.MyGeneInfo()
    print(already_mapped_ids['HGNC ID'])
    ginfo_sc = mg.querymany(already_mapped_ids['HGNC ID'], scopes='hgnc', species='human', as_dataframe=True)
    mapping_df = ginfo_sc['entrezgene']
    sampled_expression_df.index = sampled_expression_df.index.map(mapping_df.to_dict())

    # transpose and select the genes that are we were able to map
    sampled_expression_df_t = sampled_expression_df.T
    overlapping_cols = Intersection(ginfo_sc['entrezgene'].values.astype(str), sampled_expression_df_t.columns.to_list())
    sampled_expression_df_t = sampled_expression_df_t[overlapping_cols]

    # save the cell subtypes
    sampled_expression_df_t.reset_index(level=[0, 1], inplace=True)
    cell_subtype_labels = sampled_expression_df_t['subtype']
    cell_subtype_labels.to_csv('../data/cell_subtype_labels_{}.csv'.format(CMStype), index=False, header=False)

    # save the cell types
    annotations_df = pd.read_csv('../data/GSE132465_GEO_processed_CRC_10X_cell_annotation.txt', delimiter='\t')
    annotations_mapping_df = annotations_df[['Cell_type', 'Cell_subtype']]
    annotations_mapping_df.drop_duplicates(inplace=True)
    annotations_mapping_df.set_index('Cell_subtype', inplace=True)
    cell_type_labels = cell_subtype_labels.map(annotations_mapping_df.to_dict()['Cell_type'])
    cell_type_labels.to_csv('../data/cell_type_labels_{}.csv'.format(CMStype), index=False, header=False)

    # save the single cell data
    refdat = sampled_expression_df_t.drop(columns=['patient', 'subtype'])
    refdat.to_csv('../data/refdat_{}.csv'.format(CMStype))



    # Select the data from the bulk dataset belonging to only one CMSd

    df_crc_data = pd.read_table('../data/formatted_crc_data.txt')
    df_crc_data['sample'] = df_crc_data.index
    df_labels = pd.read_table('../data/cms_labels_public_all.txt',
                              usecols=['sample', 'CMS_final_network_plus_RFclassifier_in_nonconsensus_samples'])

    # match up the assigned labels with the expression rates
    df_merged = pd.merge(df_crc_data, df_labels, on='sample')
    df_merged.rename({'CMS_final_network_plus_RFclassifier_in_nonconsensus_samples': 'fin_dec'}, axis='columns',
                     inplace=True)

    df_selected_CMS = df_merged.loc[df_merged['fin_dec'] == CMStype]
    df_selected_CMS.drop(columns='fin_dec', inplace=True)
    df_selected_CMS.set_index('sample', inplace=True)
    df_selected_CMS.to_csv('../data/formatted_crc_data_{}.txt'.format(CMStype))



# Id = Conversion1('rvdb7345@gmail.com')
# for id in ids_to_translate:
#     print(id)
#     EntrezId = Id.convert_symbol_to_entrezid(id)
#     print(EntrezId)
#
# print(EntrezId)
# print(ginfo)
# mapped_ids = ids_to_translate.copy()
#
# print('Map the list of indices')
# for dict in ginfo:
#     mapped_ids[mapped_ids.index(dict['query'])] = dict['entrezgene']
#
#
# sampled_expression_df.index = mapped_ids
# print(sampled_expression_df)
#
# transposed_sampled_expression_df = sampled_expression_df.transpose()
#
# print(transposed_sampled_expression_df)
#
# transposed_sampled_expression_df.reset_index(inplace=True)
# print(transposed_sampled_expression_df)
#
#
# refdat = transposed_sampled_expression_df.drop(columns=['patient', 'subtype'])
#
# refdat.to_csv('../data/refdat.csv')
#
# cell_type_labels = transposed_sampled_expression_df['subtype']
# cell_type_labels.to_csv('../data/cell_type_labels.csv')


# annotations_df = pd.read_csv('../data/GSE132465_GEO_processed_CRC_10X_cell_annotation.txt', delimiter='\t')
