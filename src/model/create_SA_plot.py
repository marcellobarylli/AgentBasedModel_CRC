import pickle

import tqdm

from src.model.load_parameters import load_resource_influx, load_pathway_activation_thresholds, load_cell_fractions, \
    load_cell_parameters, load_inherent_signal_production, load_initial_grid_parameters
from src.model.run_experiment import create_input_proliferation, run_sim_parallel
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec


def SA_thesholds():
    sim_specification = '1'

    # ROS_threshold = 0.6783560785429451
    # NFkB_threshold = 7.6174780940379843
    # EGF_threshold = 5.5905972268825545
    # WNT_threshold = 0.60949653605233884
    # EMT_signalling_threshold = 5.51932083042646
    # STAT3_threshold = 8.2840830976556106
    # IL2_threshold = 1.724001092101165
    # TNFalpha_threshold = 1.17011720874256378
    # IFN_gamma_threshold = 1.18181407267048355
    # IL6_threshold = 2.70074466018311812
    # MYC_threshold = 2.3467913004504808
    # Shh_threshold = 1.5000000011567264
    # metabolistic_signalling_threshold = 1.5000000034023739

    threshold_list = ['ROS_threshold', 'NFkB_threshold', 'EGF_threshold', 'WNT_threshold', 'EMT_signalling_threshold',
                      'STAT3_threshold', 'IL2_threshold', 'TNFalpha_threshold', 'IFN_gamma_threshold',
                      'IL6_threshold', 'MYC_threshold', 'Shh_threshold']

    fig = plt.figure(figsize = (10,20))
    gs = gridspec.GridSpec(len(threshold_list), 1, height_ratios=[1 for i in threshold_list])
    axes = []

    for idx, threshold_type in enumerate(threshold_list):
        # threshold_type = 'metabolistic_signalling_threshold'


        threshold_dict = load_pathway_activation_thresholds()

        threshold_dicts = []

        for percentage_change in [-0.12, -0.08, -0.04, 0, 0.04, 0.08, 0.12]:
            new_threshold_dict = threshold_dict.copy()
            new_threshold_dict[threshold_type] = threshold_dict[threshold_type] + \
                                                 threshold_dict[threshold_type] * percentage_change
            threshold_dicts.append(new_threshold_dict)


        with open('exp_results/pathway_activation_thresholds/pathway_activation_thresholds_classification_{}_{}.pkl'.format(threshold_type, sim_specification),
                  'rb') as output:
            CMS_specific_results = pickle.load(output)

            end_cms_classification_mean = np.zeros((len(threshold_dicts), 4))
            end_cms_classification_std = np.zeros((len(threshold_dicts), 4))

            for ox_idx, ox_val in enumerate(threshold_dicts):
                cms_mean = CMS_specific_results[ox_idx, :, :, :].mean(axis=0)
                cms_std = CMS_specific_results[ox_idx, :, :, :].std(axis=0)
                end_cms_classification_mean[ox_idx, :] = cms_mean[-1, :]
                end_cms_classification_std[ox_idx, :] = cms_std[-1, :]

            cms_colors = ['orange', 'blue', 'pink', 'green']
            cms_names = ['CMS1', 'CMS2', 'CMS3', 'CMS4']

            if idx == 0:
                axes.append(plt.subplot(gs[idx]))
            else:
                axes.append(plt.subplot(gs[idx], sharex=axes[idx-1]))

            axes[idx].errorbar([-0.12, -0.08, -0.04, 0, 0.04, 0.08, 0.12],
                         end_cms_classification_mean[:, 3]-end_cms_classification_mean[3, 3],
                         end_cms_classification_std[:, 3],
                         linestyle='-',
                         marker='o',
                         label=threshold_type,
                         markersize=1)
            axes[idx].axhline(y=0, color='r', linestyle='--')
            axes[idx].yaxis.set_label_position("right")
            axes[idx].set_ylabel(threshold_type.strip('_threshold'))

            if not idx == len(threshold_list) - 1:
                plt.setp(axes[idx].get_xticklabels(), visible=False)

    # plt.legend(threshold_list, loc='upper center', bbox_to_anchor=(0.5, -0.18), fontsize=8, ncol=4)
    plt.xlabel('Change (%)', fontsize=14)
    fig.text(0, 0.5, 'Change in CMS4 regression value', va='center', rotation='vertical')

    # plt.ylabel('Change in CMS4 regression value', fontsize=14)
    # plt.axvline(x=threshold_dicts[cell_type_idx], linestyle='--', color=cms_colors[3])
    # plt.axvline(x=threshold_dicts[cell_type_idx], linestyle='--', color=cms_colors[1])
    plt.tight_layout()
    plt.subplots_adjust(hspace=.0)
    plt.savefig('exp_results/pathway_activation_thresholds/SA.pdf')
    plt.show()

SA_thesholds()