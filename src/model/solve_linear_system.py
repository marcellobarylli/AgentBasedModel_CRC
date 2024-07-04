import numpy as np

# coefficient_matrix = \
#     [
#         [0, 0.33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.33, 0, 0],
#         [0.33, 0, 0, 0.33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0],
#         [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0.25, 0, 0, 0.25, 0.25, 0.25, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0.33, 0.33, 0, 0, 0, 0, 0, 0, 0.33, 0],
#         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0.33, 0, 0.33, 0, 0, 0, 0, 0, 0, 0, 0, 0.33],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#         [0, 0, 0.33, 0, 0, 0, 0, 0.33, 0, 0, 0, 0.33, 0, 0, 0],
#         [0.33, 0, 0, 0.33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     ]


coefficient_matrix = \
    [
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

CMS = 'cms3'
pathway_expression_dict = \
    {
        'cms1': {'oxidative_phosphorylation_activation': 0.05658521640597041,
                 'glycolysis_activation': 0.10211212204412067,
                 'P53_activation': 0.0649869958483929, 'mTORC1_activation': 0.12526592327576827,
                 'TNFaviaNFkB_activation': 0.1994749700329655,
                 'unfolded_protein_response_activation': 0.04799894941797198,
                 'hypoxia_activation': 0.12001219426306613, 'EMT_activation': 0.07729747890186392,
                 'myogenesis_activation': 0.002557946582632461, 'MYC_target_v1_activation': 0.11068818498019588,
                 'ROS_pathway_activation': 0.017895697635618837, 'IL2_STAT5_signalling_activation': 0.09919522440889245,
                 'peroxisome_activation': -0.0012883969830358706, 'adipogenesis_activation': 0.0366904115356803,
                 'IFN_gamma_activation': 0.22217050245996225, 'kras_signalling_activation': 0.0816408696359293,
                 'IL6_JAK_activation': 0.05583661938087957, 'complement_activation': 0.12708976258460092,
                 'interferon_a_pathway_activation': 0.10412814411037, 'PI3K_activation': 0.02401074253847812},
        'cms2': {'oxidative_phosphorylation_activation': 0.03367812791294895, 'glycolysis_activation':
            -0.028521567720437804, 'P53_activation': -0.04926411910202519, 'mTORC1_activation': 0.007775579648450627,
                 'TNFaviaNFkB_activation': -0.1465407148681032,
                 'unfolded_protein_response_activation': -0.0002791087212549789,
                 'hypoxia_activation': -0.09577581529916933, 'EMT_activation': -0.22414708355040594,
                 'myogenesis_activation': -0.07693979328156893, 'MYC_target_v1_activation': 0.1019236716756058,
                 'ROS_pathway_activation': 0.0016787626212658455,
                 'IL2_STAT5_signalling_activation': -0.08909064478986109,
                 'peroxisome_activation': 0.008967021246729165, 'adipogenesis_activation': -0.02129608689991211,
                 'IFN_gamma_activation': -0.1108068430714532, 'kras_signalling_activation': -0.07443874269571442,
                 'IL6_JAK_activation': -0.03332160559370862, 'complement_activation': -0.09726316565947277,
                 'interferon_a_pathway_activation': -0.043009060222903024, 'PI3K_activation': -0.006111280117365784},
        'cms3': {'oxidative_phosphorylation_activation': 0.00718410972406589, 'glycolysis_activation':
            -0.0037051073159330037, 'P53_activation': -0.04662004685753977, 'mTORC1_activation': -0.0778845747734362,
                 'TNFaviaNFkB_activation': -0.14908169710876695,
                 'unfolded_protein_response_activation': -0.024049341569615395,
                 'hypoxia_activation': -0.10160660690021499,
                 'EMT_activation': -0.5176315913436587, 'myogenesis_activation': -0.14948727145330548,
                 'MYC_target_v1_activation': -0.07759479262673326, 'ROS_pathway_activation': -0.002659010921654546,
                 'IL2_STAT5_signalling_activation': -0.07746741197461625,
                 'peroxisome_activation': -0.008304799866119083,
                 'adipogenesis_activation': -0.05842992995324141, 'IFN_gamma_activation': -0.1559417339711343,
                 'kras_signalling_activation': -0.09186657604939871, 'IL6_JAK_activation': -0.036363722962138245,
                 'complement_activation': -0.12794670272215317, 'interferon_a_pathway_activation': -0.06634602499446263,
                 'PI3K_activation': -0.0008352702942353822},
        'cms4':
            {'oxidative_phosphorylation_activation': -0.09813570236604596, 'glycolysis_activation': 0.00318166283486331,
             'P53_activation': 0.06409466424708105, 'mTORC1_activation': -0.04318551199028102,
             'TNFaviaNFkB_activation': 0.24017120045222198,
             'unfolded_protein_response_activation': -0.012666527927498351,
             'hypoxia_activation': 0.1603597487848913, 'EMT_activation': 0.7032493561530904,
             'myogenesis_activation': 0.23427716183134947, 'MYC_target_v1_activation': -0.19420935498141464,
             'ROS_pathway_activation': -0.010550985859068086, 'IL2_STAT5_signalling_activation': 0.1381813615077819,
             'peroxisome_activation': -0.007835475067780252, 'adipogenesis_activation': 0.054162109311943984,
             'IFN_gamma_activation': 0.16158514905974636, 'kras_signalling_activation': 0.16550492141553927,
             'IL6_JAK_activation': 0.05760688771763545, 'complement_activation': 0.19735962643723684,
             'interferon_a_pathway_activation': 0.050168786332570055, 'PI3K_activation': -0.0042969529353661376}

    }

signals = ['ROS', 'Ox', 'NFkb', 'Insu', 'EGF', 'WNT', 'EMT', 'STAT3', 'IL2', 'TNFa', 'IFNg', 'IL6', 'MYC', 'SSH',
           'Meta']

pathway_expression = []
for pathway in ['P53_activation', 'mTORC1_activation', 'TNFaviaNFkB_activation', 'hypoxia_activation',
                'EMT_activation', 'MYC_target_v1_activation',
                'IL2_STAT5_signalling_activation', 'adipogenesis_activation', 'myogenesis_activation',
                'IFN_gamma_activation', 'IL6_JAK_activation', 'PI3K_activation']:
    pathway_expression.append(pathway_expression_dict[CMS][pathway])

solution, residuals, rank, s = np.linalg.lstsq(coefficient_matrix, pathway_expression, rcond=None)
print('the solution: ', list(zip(signals, solution)))
print('the residuals: ', residuals)

print('Thus the signals should be: ', dict(zip(signals, solution * 0.5 + 0.5)))

print('the thresholds should then be the following based on ending concentrations of the ')
# print('NFkb: ', 5.617754205260381 / (1 + 0.02172898)) #
# # print('Insulin: ', 5.617754205260381 / (1 + 0.02172898))
# print('EGF: ', 0.17684034342303628 / (1 + 0.0840297)) #
# print('WNT: ', 0.15609058636275353 / (1 + 0.02172898))
# print('EMT_signalling: ', 2.1511444241038244 / (1 + 0.3589553))
# print('STAT3: ', 6.152597278072701 / (1 + 0.02172898))
# print('IL2: ', 0.09347880143982377 / (1 + 0.13991807))
# print('TNFalpha: ', 0.08495970705022526 / (1 + 0.28599421))
# print('IFNgamma: ', 0.09344587227483377 / (1 + 0.20743617))
# print('IL6: ', 2.5773156462819347 / (1 + 0.02172898))
# print('MYC: ', 0.16833287042295783 / (1 + -0.10494237))
# print('Shh: ', 5.617754205260381 / (1 + 0.02172898))
# print('metabolistic_signalling: ', 5.617754205260381 / (1 + 0.02172898))


# [(1.6617662458808493, 'NFkB'), (0.4968400299999999, 'insulin'),
# (2.4759937991280507, 'EGF'), (1.1193075315113263, 'WNT'), (8.52225558403703, 'EMT_signalling'),
# (3.3605561297552624, 'STAT3'), (0.29103094161262233, 'IL2'), (0.19722275938790323, 'TNFalpha'),
# (0.1979070004381227, 'IFNgamma'), (0.13331078495848978, 'IL6'), (1.2070930424222468, 'MYC'),
# (0.36253720000000006, 'Shh'),
# (0.5315927499999998, 'metabolistic_signalling'), (0.5, 'triiodothyronine'), (0.5, 'IFNalpha')]

[(1.6526242431937737, 'NFkB'), (0.4968400299999999, 'insulin'), (2.4830754246783275, 'EGF'),
 (1.117789049010211, 'WNT'), (8.550454482715718, 'EMT_signalling'), (3.3554428872581887, 'STAT3'),
 (0.3219129873879852, 'IL2'), (0.21876974616173264, 'TNFalpha'), (0.219528887449659, 'IFNgamma'),
 (0.10293373930824215, 'IL6'), (1.2054558286060453, 'MYC'), (0.36253720000000006, 'Shh'),
 (0.5315927499999998, 'metabolistic_signalling'), (0.5, 'triiodothyronine'), (0.5, 'IFNalpha')]

ox_thres = 4.974731078879899e-10 / (1 - 0.1697287818735205)
print('oxygen_threshold= ', 4.974731078879899e-10 / (1 - 0.1697287818735205))

ros_thres = 0.671 / (1 - 0.010843978222684014)
print('ROS_threshold= ', ros_thres)  #
print('NFkB_threshold= ', 1.6526242431937737 / (1 + 0.02172898))  #
# print('Insulin: ', 5.617754205260381 / (1 + 0.02172898))
print('EGF_threshold= ', 2.4830754246783275 / (1 + 0.0840297))  #
print('WNT_threshold= ', 1.117789049010211 / (1 + 0.020844210519968126))
print('EMT_signalling_threshold= ', 8.550454482715718 / (1 + 0.3589553049626846))
print('STAT3_threshold= ', 3.3554428872581887 / (1 + 0.021728984158019597))
print('IL2_threshold= ', 0.3219129873879852 / (1 + 0.13991806974999998))
print('TNFalpha_threshold= ', 0.21876974616173264 / (1 + 0.2859942140997279))
print('IFN_gamma_threshold= ', 0.219528887449659 / (1 + 0.20743616940768425))
print('IL6_threshold= ', 0.10293373930824215 / (1 + 0.021728984158019472))
print('MYC_threshold= ', 1.2054558286060453 / (1 - 0.10494237065324154))
print('Shh_threshold= ', 0.36253720000000006 / (1 - 0.2749256016774252))
print('metabolistic_signalling_threshold= ', 0.5315927499999998 / (1 + 0.06318549276529041))

print('validating pathway activations: ')

# [(-0.08790, 'oxidative_phosphorylation_activation'), (0.002956278520466611, 'glycolysis_activation'), (0.06478641122027849, 'P53_activation'),
# (-0.036822840940473556, 'mTORC1_activation'), (0.27515023587704307, 'TNFaviaNFkB_activation'),
#  (-0.01037, 'unfolded_protein_response_activation'),  (0.16972878187352022, 'hypoxia_activation'),
#  (0.6335580006414304, 'EMT_activation'), (0.20723174305590256, 'myogenesis_activation'),
# (-0.17005168787220, 'MYC_target_v1_activation'), (-0.010843978222685, 'ROS_pathway_activation'),
#  (0.13991806975, 'IL2_STAT5_signalling_activation'), (-0.011730178049753456, 'peroxisome_activation'),
#  (0.0360213423373136, 'adipogenesis_activation'), (0.2074361694076842, 'IFN_gamma_activation'),
#  (0.169860611537902, 'kras_signalling_activation'), (0.065186952474058, 'IL6_JAK_activation'),
#  (0.2072402014628054, 'complement_activation'), (0.0703395639, 'interferon_a_pathway_activation'),
#  (0.0024950046790856682, 'PI3K_activation')]
