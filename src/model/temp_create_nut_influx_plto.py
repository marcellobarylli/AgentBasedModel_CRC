import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from load_parameters import load_cell_parameters, load_cell_fractions, load_biological_parameters, \
    load_initial_grid_parameters, load_inherent_signal_production, load_resource_influx

matplotlib.rcParams.update({'errorbar.capsize': 2})


def nut_var_class_merge():
    sim_specification = 1

    nutrient_values_low = [0.001 * 10 ** -15, 0.0025 * 10 ** -15, 0.005 * 10 ** -15, 0.0075 * 10 ** -15,
                       0.01 * 10 **
                       -15,
                       0.025 * 10 ** -15, 0.05 * 10 ** -15, 0.075 * 10 ** -15]

    nutrient_values_high = [0.1 * 10 ** -15, 0.5 * 10 ** -15, 1 * 10 ** -15, 2.5 * 10 ** -15, 5 * 10 ** -15, 10 * 10 ** -15,
                       15 * 10 ** -15, 20 * 10 ** -15, 25 * 10 ** -15]

    with open('exp_results/nut_var/nut_var_classification_{}.pkl'.format(sim_specification), 'rb') as output:
        CMS_specific_results = pickle.load(output)
        print(CMS_specific_results)
        end_cms_classification_mean_high = np.zeros((len(nutrient_values_high), 4))
        end_cms_classification_std_high = np.zeros((len(nutrient_values_high), 4))

        for ox_idx, ox_val in enumerate(nutrient_values_high):
            cms_mean = CMS_specific_results[ox_idx, :, :, :].mean(axis=0)
            cms_std = CMS_specific_results[ox_idx, :, :, :].std(axis=0)
            end_cms_classification_mean_high[ox_idx, :] = cms_mean[-1, :]
            end_cms_classification_std_high[ox_idx, :] = cms_std[-1, :]

    sim_specification = 2

    with open('exp_results/nut_var/nut_var_classification_{}.pkl'.format(sim_specification), 'rb') as output:
        CMS_specific_results = pickle.load(output)
        print(CMS_specific_results)
        end_cms_classification_mean_low = np.zeros((len(nutrient_values_low), 4))
        end_cms_classification_std_low = np.zeros((len(nutrient_values_low), 4))

        for ox_idx, ox_val in enumerate(nutrient_values_low):
            cms_mean = CMS_specific_results[ox_idx, :, :, :].mean(axis=0)
            cms_std = CMS_specific_results[ox_idx, :, :, :].std(axis=0)
            end_cms_classification_mean_low[ox_idx, :] = cms_mean[-1, :]
            end_cms_classification_std_low[ox_idx, :] = cms_std[-1, :]



    cms_colors = ['blue', 'orange', 'green', 'red']
    cms_names = ['CMS1', 'CMS2', 'CMS3', 'CMS4']

    nutrient_values = np.concatenate((nutrient_values_low, nutrient_values_high))
    end_cms_classification_mean = np.concatenate((end_cms_classification_mean_low, end_cms_classification_mean_high))
    end_cms_classification_std = np.concatenate((end_cms_classification_std_low, end_cms_classification_std_high))

    plt.figure()
    for idx, (color, cms) in enumerate(list(zip(*(cms_colors, cms_names)))):
        plt.errorbar(nutrient_values, end_cms_classification_mean[:, idx], end_cms_classification_std[:, idx],
                     linestyle='-',
                     marker='o',
                     label=cms,
                     markersize=1, color=color)

    plt.legend()
    plt.xlabel(r'Nutrient influx ($\mathrm{mol} h^{-1} \mathrm{grid}_{\mathrm{vessel}}^{-1}$)', fontsize=14)
    plt.xscale('log')
    plt.ylabel('CMS classification (#)', fontsize=14)
    plt.savefig('exp_results/nut_var/nut_influx_classification_{}.pdf'.format(cms))
    plt.show()

def nut_var_num_cells_merge():
    nutrient_values_low = [0.001 * 10 ** -15, 0.0025 * 10 ** -15, 0.005 * 10 ** -15, 0.0075 * 10 ** -15,
                           0.01 * 10 **
                           -15,
                           0.025 * 10 ** -15, 0.05 * 10 ** -15, 0.075 * 10 ** -15]

    nutrient_values_high = [0.1 * 10 ** -15, 0.5 * 10 ** -15, 1 * 10 ** -15, 2.5 * 10 ** -15, 5 * 10 ** -15,
                            10 * 10 ** -15,
                            15 * 10 ** -15, 20 * 10 ** -15, 25 * 10 ** -15]

    sim_specification = 2
    with open('exp_results/nut_var/nut_var_num_cells_{}.pkl'.format(sim_specification), 'rb') as output:
        CMS_specific_results = pickle.load(output)

        end_cms_classification_mean_low = np.zeros((len(nutrient_values_low), 5))
        end_cms_classification_std_low = np.zeros((len(nutrient_values_low), 5))

        for ox_idx, ox_val in enumerate(nutrient_values_low):
            cms_mean = CMS_specific_results[ox_idx, :, :, :].mean(axis=0)
            cms_std = CMS_specific_results[ox_idx, :, :, :].std(axis=0)
            end_cms_classification_mean_low[ox_idx, :] = cms_mean[-1, :]
            end_cms_classification_std_low[ox_idx, :] = cms_std[-1, :]

        cell_colors = ['brown', 'blue', 'pink', 'green', 'red']
        cell_names = ['Cancer', 'T cell', 'B cell', 'Myeloid', 'Stromal cells']

    sim_specification = 1
    with open('exp_results/nut_var/nut_var_num_cells_{}.pkl'.format(sim_specification), 'rb') as output:
        CMS_specific_results = pickle.load(output)

        end_cms_classification_mean_high = np.zeros((len(nutrient_values_high), 5))
        end_cms_classification_std_high = np.zeros((len(nutrient_values_high), 5))

        for ox_idx, ox_val in enumerate(nutrient_values_high):
            cms_mean = CMS_specific_results[ox_idx, :, :, :].mean(axis=0)
            cms_std = CMS_specific_results[ox_idx, :, :, :].std(axis=0)
            end_cms_classification_mean_high[ox_idx, :] = cms_mean[-1, :]
            end_cms_classification_std_high[ox_idx, :] = cms_std[-1, :]

        cell_colors = ['brown', 'blue', 'pink', 'green', 'red']
        cell_names = ['Cancer', 'T cell', 'B cell', 'Myeloid', 'Stromal cells']

    nutrient_values = np.concatenate((nutrient_values_low, nutrient_values_high))
    end_cms_classification_mean = np.concatenate((end_cms_classification_mean_low, end_cms_classification_mean_high))
    end_cms_classification_std = np.concatenate((end_cms_classification_std_low, end_cms_classification_std_high))

    plt.figure()
    for idx, (color, cms) in enumerate(list(zip(*(cell_colors, cell_names)))):
        plt.errorbar(nutrient_values, end_cms_classification_mean[:, idx], end_cms_classification_std[:, idx],
                     linestyle='-',
                     marker='o',
                     label=cms,
                     markersize=1, color=color)

    plt.legend()
    plt.xlabel(r'Nutrient influx ($\mathrm{mol} h^{-1} \mathrm{grid}_{\mathrm{vessel}}^{-1}$)', fontsize=14)
    plt.ylabel('Cell count (#)', fontsize=14)
    plt.xscale('log')
    plt.savefig('exp_results/nut_var/nut_influx_cell_count_{}.pdf'.format(sim_specification))
    plt.show()


def t_frac_var_class_merge():
    sim_specification = 1
    cell_type_idx = 2
    cell_type = 'tcell'

    t_cell_fraction_low = [0.27, 0.32, 0.37, 0.42, 0.47, 0.52, 0.57, 0.62, 0.67, 0.72, 0.77] # first run
    t_cell_fraction_high = [0.87, 0.97, 1.07, 1.17, 1.27, 1.37, 1.47, 1.57]

    fraction_types_cms4 = load_cell_fractions('cms4')
    base_fraction_cms4 = fraction_types_cms4.copy()
    base_fraction_cms4[1:] = fraction_types_cms4[1:] / sum(fraction_types_cms4[1:])
    fraction_types_cms2 = load_cell_fractions('cms2')
    base_fraction_cms2 = fraction_types_cms2.copy()
    base_fraction_cms2[1:] = fraction_types_cms2[1:] / sum(fraction_types_cms2[1:])
    # np.array([0.045, 0.32, 0.37, 0.12, 0.18])

    cell_type_fraction = t_cell_fraction_low + t_cell_fraction_high
    fractions_list = np.array([fraction_types_cms4 for i in range(len(cell_type_fraction))])
    fractions_list[:, cell_type_idx] = cell_type_fraction

    for frac_idx, fractions in enumerate(fractions_list):
        fractions_list[frac_idx, 1:] = fractions_list[frac_idx, 1:] / sum(fractions_list[frac_idx, 1:])

    sim_specification = 1
    with open('exp_results/fraction_var/fraction_var_classification_{}_{}.pkl'.format(cell_type,sim_specification),
              'rb') as output:
        CMS_specific_results = pickle.load(output)

        end_cms_classification_mean_low = np.zeros((len(t_cell_fraction_low), 4))
        end_cms_classification_std_low = np.zeros((len(t_cell_fraction_low), 4))

        for ox_idx, ox_val in enumerate(t_cell_fraction_low):
            cms_mean = CMS_specific_results[ox_idx, :, :, :].mean(axis=0)
            cms_std = CMS_specific_results[ox_idx, :, :, :].std(axis=0)
            end_cms_classification_mean_low[ox_idx, :] = cms_mean[-1, :]
            end_cms_classification_std_low[ox_idx, :] = cms_std[-1, :]


    sim_specification = 2
    with open('exp_results/fraction_var/fraction_var_classification_{}_{}.pkl'.format(cell_type,sim_specification),
              'rb') as output:
        CMS_specific_results = pickle.load(output)

        end_cms_classification_mean_high = np.zeros((len(t_cell_fraction_high), 4))
        end_cms_classification_std_high = np.zeros((len(t_cell_fraction_high), 4))

        for ox_idx, ox_val in enumerate(t_cell_fraction_high):
            cms_mean = CMS_specific_results[ox_idx, :, :, :].mean(axis=0)
            cms_std = CMS_specific_results[ox_idx, :, :, :].std(axis=0)
            end_cms_classification_mean_high[ox_idx, :] = cms_mean[-1, :]
            end_cms_classification_std_high[ox_idx, :] = cms_std[-1, :]



    cms_colors = ['blue', 'orange', 'green', 'red']
    cms_names = ['CMS1', 'CMS2', 'CMS3', 'CMS4']
    end_cms_classification_mean = np.concatenate((end_cms_classification_mean_low, end_cms_classification_mean_high))
    end_cms_classification_std = np.concatenate((end_cms_classification_std_low, end_cms_classification_std_high))

    plt.figure()
    for idx, (color, cms) in enumerate(list(zip(*(cms_colors, cms_names)))):
        plt.errorbar(fractions_list[:, cell_type_idx], end_cms_classification_mean[:, idx],
                     end_cms_classification_std[:, idx],
                     linestyle='-',
                     marker='o',
                     label=cms,
                     markersize=1, color=color)

    plt.legend()
    plt.xlabel('T cell fraction in ME'.format(cell_type), fontsize=14)
    plt.ylabel('CMS classification (#)', fontsize=14)
    plt.axvline(x=base_fraction_cms4[cell_type_idx], linestyle='--', color=cms_colors[3])
    plt.axvline(x=base_fraction_cms2[cell_type_idx], linestyle='--', color=cms_colors[1])
    plt.savefig('exp_results/fraction_var/fraction_var_classification_merged_{}_{}.pdf'.format(cell_type,
                                                                                        sim_specification))
    plt.show()

nut_var_class_merge()
nut_var_num_cells_merge()