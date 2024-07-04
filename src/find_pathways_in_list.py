import mygene
import json
import os
from tabulate import tabulate

import pandas as pd
import gseapy as gp
import matplotlib.pyplot as plt

CMS_TYPES = ['CMS1', 'CMS2', 'CMS3', 'CMS4']

# mg = mygene.MyGeneInfo()
# xli = [1017, 1018, 695]
# out = mg.getgenes(xli, fields="name,symbol,pathways")
# print(out)

names_library = gp.get_library_name()  # default: Human
pd.set_option('display.max_columns', 10)

cell_type_to_look_at = 'Mast cells'

cell_types = ['T cells', 'B cells', 'Myeloids', 'Stromal cells', 'Epithelial cells']
side = 'neg'
p_cutoff = '0025'
tissue_class = 'Tumor'
scale = 'raw'

for cell_type_to_look_at in cell_types:
    with open('../results/diff_exp_genes_sign_{}_{}_{}_{}_{}.txt'.format(side, p_cutoff, cell_type_to_look_at,
                                                                      tissue_class, scale)) as \
            json_file:
        data = json.load(json_file)

        result_df = pd.DataFrame(columns=['CMS', 'Gene_set', 'Term', 'Overlap', 'P-value', 'Adjusted P-value',
                                          'Old P-value',
                                          'Old Adjusted P-value', 'Odds Ratio', 'Combined Score'])

        for subtype in CMS_TYPES:
            print('We find the following pathway overlaps for {} with {}'.format(subtype, cell_type_to_look_at))
            genes = data[subtype]
            print("The number of genes inputted to enrichr: ", len(genes))

            if not len(genes) == 0:

                with open('temp_genes.txt', 'w') as f:
                    for item in genes:
                        # print(item)
                        f.write("%s\n" % item)

                enr = gp.enrichr(gene_list=genes,
                                 gene_sets=['MSigDB_Hallmark_2020',
                                            'KEGG_2019_Human',
                                            'GO_Biological_Process_2018',
                                            'GO_Cellular_Component_2018',
                                            'Human_Phenotype_Ontology',
                                            'MSigDB_Oncogenic_Signatures'
                                            ],
                                 organism='Human',  # don't forget to set organism to the one you desired! e.g. Yeast
                                 description='test_name',
                                 outdir='../results/{}/{}/enrichr_kegg'.format(subtype, cell_type_to_look_at),
                                 no_plot=True
                                 )

                os.remove('temp_genes.txt')
                gp.plot.barplot(enr.res2d, title='KEGG_2013', )
                plt.show()

                subtype_result_df = enr.results
                subtype_result_df['CMS'] = subtype
                result_df = result_df.append(subtype_result_df)

                print(tabulate(enr.results[['Gene_set', 'Term', 'Overlap', 'Genes']].head(15),
                               headers='keys',
                               tablefmt='csv'))
            result_df.to_csv('../results/pathway_analysis_{}_{}_{}_{}_{}_20210401.csv'.format(side, p_cutoff,
                                                                                    cell_type_to_look_at,
                                                                                tissue_class, scale))
