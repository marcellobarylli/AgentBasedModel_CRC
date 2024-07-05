def create_selection_sc_gene_expression_df(indices_to_group, scale='raw', n=100, tissue_class='Tumor'):
    indices_to_group = indices_to_group.apply(lambda x: x if len(x) <= n else random.sample(x, n))

    flat_list = [item for sublist in indices_to_group.values for item in sublist]

    print('The number of columns to load would be: ', len(flat_list))

    start = time.time()
    print('Start to load the dataframe')
    if scale == 'raw':
        raw_counts_df = pd.read_csv('../data/GSE132465_GEO_processed_CRC_10X_raw_UMI_count_matrix.txt',
                                    delimiter='\t', usecols=['Index'] + flat_list, index_col=['Index'])
    elif scale == 'log':
        raw_counts_df = pd.read_csv('../data/GSE132465_GEO_processed_CRC_10X_natural_log_TPM_matrix.txt',
                                    delimiter='\t', usecols=['Index'] + flat_list, index_col=['Index'])
    print('Finished loading the dataframe after: ', time.time() - start)

    averaged_df = pd.DataFrame(index=raw_counts_df.index)
    for patient, new_df in indices_to_group.groupby(level=0):
        for subtype, newer_df in new_df.groupby(level=1):
            averaged_df[list(zip([str(patient)] * n, [str(subtype)] * n, list(newer_df.values[0])))] = \
                raw_counts_df[list(newer_df.values)[0]]
            averaged_df[list(zip([str(patient)] * n, [str(subtype)] * n, list(newer_df.values[0])))] = \
                raw_counts_df[list(newer_df.values)[0]]

    averaged_df.columns = pd.MultiIndex.from_tuples(list(averaged_df.columns.values),
                                                    names=['patient', 'subtype', 'samples'])

    print('This much time has elapsed: ', time.time() - start)

    averaged_df.to_csv('../data/selected_sc_gene_express_{}_{}.csv'.format(tissue_class, scale))