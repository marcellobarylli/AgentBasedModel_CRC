import tqdm
import time
import random
import matplotlib.pyplot as plt
import sys
import pickle
import numpy as np
import matplotlib
import scipy.stats as st

from matplotlib import colors
from utils import calc_metrics, get_num_neighbours, choice, cdf, create_metric_lists, save_metrics, trim_lists, \
    save_raw_data, get_moore
from load_parameters import load_cell_parameters, load_cell_fractions, load_biological_parameters, \
    load_initial_grid_parameters, load_inherent_signal_production, load_resource_influx, \
    load_pathway_activation_thresholds
from simulation_initiation import initiate_simulation_grids
from pathway_cms_classification import sample_classifier
from diffusion_scripts.diffusion_glutamine import diffusion_nutrients
from diffusion_scripts.diffusion_oxygen import diffusion_oxygen
from diffusion_scripts.diffusion_ROS import diffusion_ROS
from diffusion_scripts.diffusion_signal import diffusion_signal
from plotting import create_final_plots, create_animation, update_figs, init_plots
from cell_definitions import cell, cancercell, myeloid, tcell, bcell, stromalcell
from scipy.optimize import curve_fit

from uncertainties import unumpy
from uncertainties import ufloat
from main import run_sim
import multiprocessing
from multiprocessing import Pool
import os, sys

matplotlib.rcParams.update({'errorbar.capsize': 2})


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# create input for changing same_cell_int values
def create_input_proliferation(n_repeats, n_steps, grid_size, n_cells, init_cms, fraction_types,
                               params_tumour, params_tcell, params_bcell,
                               params_myeloid, params_stromal, resources_init_value,
                               oxygen_init_value, ros_init_value, signals_init_value,
                               influxes,
                               plot, get_signal_strengths, do_animation, save_data, get_diffused_grids,
                               proliferation_plot, proliferation_cell,
                               replenish_time, prediffusion_step, return_data, fixed_inflammation, ox_modulation):
    # n_steps, grid_size, n_cells, init_cms, fraction_types, params_tumour, params_tcell, params_bcell,
    # params_myeloid, params_stromal, resources_init_value, oxygen_init_value, ros_init_value,
    # signals_init_value, influxes, plot = True, get_signalling_strengths = False,
    # do_animation = False, save_data = True, get_diffused_grids = False, proliferation_plot = False,
    # proliferation_cell = None, replenish_time = 24, prediffision_step = 1, return_data = False
    input = []
    for same_cell_int in range(n_repeats):
        input.append((n_steps, grid_size, n_cells, init_cms, fraction_types,
                      params_tumour, params_tcell, params_bcell,
                      params_myeloid, params_stromal, resources_init_value,
                      oxygen_init_value, ros_init_value, signals_init_value,
                      influxes,
                      plot, get_signal_strengths, do_animation, save_data, get_diffused_grids,
                      proliferation_plot, proliferation_cell,
                      replenish_time, prediffusion_step, return_data, fixed_inflammation, ox_modulation))

    return input


def mute():
    sys.stdout = open(os.devnull, 'w')


# runs the simulation in parallel, using the parameter combinations specified in input
def run_sim_parallel(input):
    pool = Pool(os.cpu_count() - 12, initializer=mute)
    pool_list = pool.starmap(run_sim, input)
    pool.close()
    pool.join()

    # write_output(pool_list)

    return pool_list


# do logarithmic fit
def gompertz(x, l, m, n):
    # a = upper asymptote
    # b = negative = x axis displacement
    # c = negative = growth rate
    return l * (np.exp(m * (np.exp(n * x))))


def fit_gompertz(cms1_mean, cms2_mean, cms3_mean, cms4_mean):
    popt, _ = curve_fit(gompertz, range(0, len(cms1_mean)), cms1_mean, maxfev=2000)
    l, m, n = popt
    cms1_fit = gompertz(range(0, len(cms1_mean)), l, m, n)

    print('m is: ', 1 / n)

    popt, _ = curve_fit(gompertz, range(0, len(cms2_mean)), cms2_mean, maxfev=2000)
    l, m, n = popt
    cms2_fit = gompertz(range(0, len(cms1_mean)), l, m, n)

    print('m is: ', 1 / n)

    popt, _ = curve_fit(gompertz, range(0, len(cms3_mean)), cms3_mean, maxfev=2000)
    l, m, n = popt
    cms3_fit = gompertz(range(0, len(cms1_mean)), l, m, n)

    print('m is: ', 1 / n)

    popt, _ = curve_fit(gompertz, range(0, len(cms4_mean)), cms4_mean, maxfev=2000)
    l, m, n = popt
    cms4_fit = gompertz(range(0, len(cms1_mean)), l, m, n)

    print('m is: ', 1 / n)

    return cms1_fit, cms2_fit, cms3_fit, cms4_fit


def calc_doubling_time(cms1_mean, cms2_mean, cms3_mean, cms4_mean, cms1_std, cms2_std, cms3_std, cms4_std):
    print('Doubling time CMS1', unumpy.log(2) / (unumpy.log(ufloat(cms1_mean[-1], cms1_std[-1]) / 60) / range(len(
        cms1_mean))[-1]))
    print('Doubling time CMS2', unumpy.log(2) / (unumpy.log(ufloat(cms2_mean[-1], cms2_std[-1]) / 60) / range(len(
        cms1_mean))[-1]))
    print('Doubling time CMS3', unumpy.log(2) / (unumpy.log(ufloat(cms3_mean[-1], cms3_std[-1]) / 60) / range(len(
        cms1_mean))[-1]))
    print('Doubling time CMS4', unumpy.log(2) / (unumpy.log(ufloat(cms4_mean[-1], cms4_std[-1]) / 60) / range(len(
        cms1_mean))[-1]))


def proliferation_plot():
    # return_data = True
    # n_repeats = 100
    #
    # # set variables
    # grid_size = 100
    # n_steps = 80
    # n_cells = 20000
    #
    # # set settings for the proliferation plots
    # proliferation_plot = True
    # proliferation_cell = 'tumour'
    # replenish_time = 24
    # prediffusion_step = 1
    #
    # # cancer cell, myeloid, tcell, bcell, stromal
    # resources_init_value, oxygen_init_value, ros_init_value, signals_init_value = load_initial_grid_parameters()
    # influxes = load_resource_influx(proliferation_plot)
    #
    # # overwrite the initial nutrient deposit to the influx in case of a growth curve experiment
    # if proliferation_plot:
    #     resources_init_value = influxes['nutrient']
    #
    threshold_dict = load_pathway_activation_thresholds()

    # params_tumour, params_tcell, params_bcell, params_myeloid, params_stromal = load_cell_parameters(threshold_dict)
    #
    # cms_types = ['cms1', 'cms2', 'cms3', 'cms4']
    #
    # CMS_specific_results = np.zeros((len(cms_types), n_repeats, n_steps + 1))
    #
    # for cms_idx, init_cms in enumerate(tqdm.tqdm(cms_types, position=0, desc='CMS type', colour='blue', leave=True)):
    #     params_tumour, params_tcell, params_bcell, params_myeloid, params_stromal = load_inherent_signal_production(
    #         init_cms, params_tumour, params_tcell, params_bcell, params_myeloid, params_stromal)
    #     fraction_types = load_cell_fractions(init_cms)
    #
    #     input = create_input_proliferation(n_repeats, n_steps, grid_size, n_cells, init_cms, fraction_types,
    #                         params_tumour, params_tcell, params_bcell,
    #                         params_myeloid, params_stromal, resources_init_value,
    #                         oxygen_init_value, ros_init_value, signals_init_value,
    #                         influxes, False,
    #                         False, False, False, False,
    #                         proliferation_plot, proliferation_cell,
    #                         replenish_time, prediffusion_step, return_data, False)
    #
    #     output = run_sim_parallel(input)
    #
    #     for repeat_idx in range(n_repeats):
    #         CMS_specific_results[cms_idx, repeat_idx, :] = output[repeat_idx]['num_cells_list'][:, 0]
    #
    # with open('exp_results/proliferation_plots_data.pkl', 'wb') as output:
    #     pickle.dump(CMS_specific_results, output)

    with open('exp_results/proliferation_plots_data.pkl', 'rb') as output:
        CMS_specific_results = pickle.load(output)
        cms1_mean = CMS_specific_results[0, :, :].mean(axis=0)
        cms2_mean = CMS_specific_results[1, :, :].mean(axis=0)
        cms3_mean = CMS_specific_results[2, :, :].mean(axis=0)
        cms4_mean = CMS_specific_results[3, :, :].mean(axis=0)

        # code to calculate a 95% confidence ratio
        # cms1_lower, cms1_upper = st.t.interval(alpha=0.95, df=len(CMS_specific_results[0, :, :])-1,
        # loc=CMS_specific_results[0, :, :].mean(axis=0), scale=st.sem(CMS_specific_results[0, :, :].mean(axis=0)))
        # cms2_lower, cms2_upper = st.t.interval(alpha=0.95, df=len(CMS_specific_results[1, :, :])-1,
        #                                        loc=CMS_specific_results[1, :, :].mean(axis=0), scale=st.sem(
        #         CMS_specific_results[1, :, :].mean(axis=0)))
        # cms3_lower, cms3_upper = st.t.interval(alpha=0.95, df=len(CMS_specific_results[2, :, :])-1,
        #                                        loc=CMS_specific_results[2, :, :].mean(axis=0), scale=st.sem(
        #         CMS_specific_results[2, :, :].mean(axis=0)))
        # cms4_lower, cms4_upper = st.t.interval(alpha=0.95, df=len(CMS_specific_results[3, :, :])-1,
        #                                        loc=CMS_specific_results[3, :, :].mean(axis=0), scale=st.sem(
        #         CMS_specific_results[3, :, :].mean(axis=0)))

        cms1_std = CMS_specific_results[0, :, :].std(axis=0)
        cms2_std = CMS_specific_results[1, :, :].std(axis=0)
        cms3_std = CMS_specific_results[2, :, :].std(axis=0)
        cms4_std = CMS_specific_results[3, :, :].std(axis=0)

        calc_doubling_time(cms1_mean, cms2_mean, cms3_mean, cms4_mean, cms1_std, cms2_std, cms3_std, cms4_std)
        cms1_fit, cms2_fit, cms3_fit, cms4_fit = fit_gompertz(cms1_mean, cms2_mean, cms3_mean, cms4_mean)

        plt.figure()
        # plt.plot(range(len(cms1_mean)), cms1_fit, label='CMS1-fit', color='blue')
        # plt.plot(range(len(cms1_mean)), cms2_fit, label='CMS2-fit', color='red')
        # plt.plot(range(len(cms1_mean)), cms3_fit, label='CMS3-fit', color='green')
        # plt.plot(range(len(cms1_mean)), cms4_fit, label='CMS4-fit', color='black')

        plt.errorbar(range(len(cms1_mean)), cms1_mean, cms1_std, linestyle='-', marker='^', label='CMS1',
                     markersize=1, color='blue')
        plt.errorbar(range(len(cms2_mean)), cms2_mean, cms2_std, linestyle='-', marker='o', label='CMS2',
                     markersize=1, color='red')
        plt.errorbar(range(len(cms3_mean)), cms3_mean, cms3_std, linestyle='-', marker='s', label='CMS3',
                     markersize=1, color='green')
        plt.errorbar(range(len(cms4_mean)), cms4_mean, cms4_std, linestyle='-', marker='p', label='CMS4',
                     markersize=1, color='black')
        plt.legend()
        # plt.grid()
        plt.xlabel('Time (h)', fontsize=14)
        plt.ylabel('Cells (#)', fontsize=14)
        plt.xlim(0)
        plt.savefig('exp_results/proliferation_plots.pdf')
        plt.show()


def vary_me_composition():
    sim_specification = 'start__cms4_tweaking_3'

    return_data = True
    n_repeats = 30

    # set variables
    grid_size = 100
    n_steps = 200
    n_cells = 20000

    # set settings for the proliferation plots
    proliferation_plot = False
    proliferation_cell = 'tumour'
    replenish_time = 24
    prediffusion_step = 1

    # cancer cell, myeloid, tcell, bcell, stromal
    resources_init_value, oxygen_init_value, ros_init_value, signals_init_value = load_initial_grid_parameters()
    influxes = load_resource_influx(proliferation_plot)

    # overwrite the initial nutrient deposit to the influx in case of a growth curve experiment
    if proliferation_plot:
        resources_init_value = influxes['nutrient']

    threshold_dict = load_pathway_activation_thresholds()
    params_tumour, params_tcell, params_bcell, params_myeloid, params_stromal = load_cell_parameters(threshold_dict)

    cms_types = ['cms2', 'cms4']

    # print(load_cell_fractions('cms2'), load_cell_fractions('cms4'))

    CMS_specific_results = np.zeros((len(cms_types), n_repeats, n_steps + 1, 4))

    for cms_idx, cms_me_fraction in enumerate(tqdm.tqdm(cms_types, position=0, desc='CMS type', colour='blue',
                                                        leave=True)):
        params_tumour, params_tcell, params_bcell, params_myeloid, params_stromal = load_inherent_signal_production(
            'cms4', params_tumour, params_tcell, params_bcell, params_myeloid, params_stromal)
        fraction_types = load_cell_fractions(cms_me_fraction)

        input = create_input_proliferation(n_repeats, n_steps, grid_size, n_cells, 'cms4', fraction_types,
                                           params_tumour, params_tcell, params_bcell,
                                           params_myeloid, params_stromal, resources_init_value,
                                           oxygen_init_value, ros_init_value, signals_init_value,
                                           influxes, False,
                                           False, False, False, False,
                                           proliferation_plot, proliferation_cell,
                                           replenish_time, prediffusion_step, return_data, False)

        output = run_sim_parallel(input)

        for repeat_idx in range(n_repeats):
            CMS_specific_results[cms_idx, repeat_idx, :, :] = output[repeat_idx]['cms_classification_list'][:, :]

    print('saving data')
    with open('exp_results/ME/mc_env_{}.pkl'.format(sim_specification), 'wb') as output:
        pickle.dump(CMS_specific_results, output)

    with open('exp_results/ME/mc_env_{}.pkl'.format(sim_specification), 'rb') as output:
        # np.zeros((len(cms_types), n_repeats, n_steps + 1, 4))
        CMS_specific_results = pickle.load(output)
        cms2_mean = CMS_specific_results[0, :, :, :].mean(axis=0)
        cms4_mean = CMS_specific_results[1, :, :, :].mean(axis=0)

        print(cms2_mean)
        # code to calculate a 95% confidence ratio
        # cms1_lower, cms1_upper = st.t.interval(alpha=0.95, df=len(CMS_specific_results[0, :, :])-1,
        # loc=CMS_specific_results[0, :, :].mean(axis=0), scale=st.sem(CMS_specific_results[0, :, :].mean(axis=0)))
        # cms2_lower, cms2_upper = st.t.interval(alpha=0.95, df=len(CMS_specific_results[1, :, :])-1,
        #                                        loc=CMS_specific_results[1, :, :].mean(axis=0), scale=st.sem(
        #         CMS_specific_results[1, :, :].mean(axis=0)))
        # cms3_lower, cms3_upper = st.t.interval(alpha=0.95, df=len(CMS_specific_results[2, :, :])-1,
        #                                        loc=CMS_specific_results[2, :, :].mean(axis=0), scale=st.sem(
        #         CMS_specific_results[2, :, :].mean(axis=0)))
        # cms4_lower, cms4_upper = st.t.interval(alpha=0.95, df=len(CMS_specific_results[3, :, :])-1,
        #                                        loc=CMS_specific_results[3, :, :].mean(axis=0), scale=st.sem(
        #         CMS_specific_results[3, :, :].mean(axis=0)))

        cms2_std = CMS_specific_results[0, :, :, :].std(axis=0)
        cms4_std = CMS_specific_results[1, :, :, :].std(axis=0)

        plt.figure()

        plt.errorbar(range(len(cms2_mean))[0::10], cms2_mean[::10, 0], cms2_std[::10, 0], linestyle='-', marker='o',
                     label='CMS2-CMS1',
                     markersize=1, color='red')
        plt.errorbar(range(len(cms2_mean))[0::10], cms2_mean[::10, 1], cms2_std[::10, 1], linestyle='-', marker='o',
                     label='CMS2-CMS2',
                     markersize=1, color='black')
        plt.errorbar(range(len(cms2_mean))[0::10], cms2_mean[::10, 2], cms2_std[::10, 2], linestyle='-', marker='o',
                     label='CMS2-CMS3',
                     markersize=1, color='green')
        plt.errorbar(range(len(cms2_mean))[0::10], cms2_mean[::10, 3], cms2_std[::10, 3], linestyle='-', marker='o',
                     label='CMS2-CMS4',
                     markersize=1, color='orange')
        plt.errorbar(range(len(cms4_mean))[0::10], cms4_mean[::10, 0], cms4_std[::10, 0], linestyle='--', marker='o',
                     label='CMS4-CMS1',
                     markersize=1, color='red')
        plt.errorbar(range(len(cms4_mean))[0::10], cms4_mean[::10, 1], cms4_std[::10, 1], linestyle='--', marker='o',
                     label='CMS4-CMS2',
                     markersize=1, color='black')
        plt.errorbar(range(len(cms4_mean))[0::10], cms4_mean[::10, 2], cms4_std[::10, 2], linestyle='--', marker='o',
                     label='CMS4-CMS3',
                     markersize=1, color='green')
        plt.errorbar(range(len(cms4_mean))[0::10], cms4_mean[::10, 3], cms4_std[::10, 3], linestyle='--', marker='o',
                     label='CMS4-CMS4',
                     markersize=1, color='orange')
        plt.legend()
        plt.xlabel('Time (h)', fontsize=14)
        plt.ylabel('CMS classification (#)', fontsize=14)
        plt.savefig('exp_results/ME/ME_composition.pdf')
        plt.show()


def repeated_base_sim():
    sim_specification = '2'
    cms_types = ['cms2', 'cms4']
    return_data = True
    n_repeats = 30

    # set variables
    grid_size = 100
    n_steps = 2000
    n_cells = 20000

    # set settings for the proliferation plots
    proliferation_plot = False
    proliferation_cell = 'tumour'
    replenish_time = 24
    prediffusion_step = 1

    # cancer cell, myeloid, tcell, bcell, stromal
    resources_init_value, oxygen_init_value, ros_init_value, signals_init_value = load_initial_grid_parameters()
    influxes = load_resource_influx(proliferation_plot)

    # overwrite the initial nutrient deposit to the influx in case of a growth curve experiment
    if proliferation_plot:
        resources_init_value = influxes['nutrient']

    threshold_dict = load_pathway_activation_thresholds()
    params_tumour, params_tcell, params_bcell, params_myeloid, params_stromal = load_cell_parameters(threshold_dict)

    CMS_specific_results_classification = np.zeros((len(cms_types), n_repeats, n_steps + 1, 4))
    CMS_specific_results_num_cells = np.zeros((len(cms_types), n_repeats, n_steps + 1, 5))
    CMS_specific_results_inflammation = np.zeros((len(cms_types), n_repeats, n_steps + 1))
    CMS_specific_results_pathway_activation = np.zeros((len(cms_types), n_repeats, n_steps + 1, 15))
    CMS_specific_results_signal_list = np.zeros((len(cms_types), n_repeats, n_steps + 1, 14))

    for cms_idx, cms_init in enumerate(tqdm.tqdm(cms_types, position=0, desc='CMS type', colour='blue',
                                                 leave=True)):
        params_tumour, params_tcell, params_bcell, params_myeloid, params_stromal = load_inherent_signal_production(
            cms_init, params_tumour, params_tcell, params_bcell, params_myeloid, params_stromal)
        fraction_types = load_cell_fractions(cms_init)

        input = create_input_proliferation(n_repeats, n_steps, grid_size, n_cells, cms_init, fraction_types,
                                           params_tumour, params_tcell, params_bcell,
                                           params_myeloid, params_stromal, resources_init_value,
                                           oxygen_init_value, ros_init_value, signals_init_value,
                                           influxes, False,
                                           False, False, False, False,
                                           proliferation_plot, proliferation_cell,
                                           replenish_time, prediffusion_step, return_data, False, 1.5)

        output = run_sim_parallel(input)

        for repeat_idx in range(n_repeats):
            CMS_specific_results_classification[cms_idx, repeat_idx, :, :] = \
                output[repeat_idx]['cms_classification_list'][:, :]
            CMS_specific_results_num_cells[cms_idx, repeat_idx, :, :] = \
                output[repeat_idx]['num_cells_list'][:, :]
            CMS_specific_results_inflammation[cms_idx, repeat_idx, :] = \
                output[repeat_idx]['inflammation_list'][:]
            CMS_specific_results_pathway_activation[cms_idx, repeat_idx, :, :] = \
                output[repeat_idx]['pathway_activation_list'][:, :]
            CMS_specific_results_signal_list[cms_idx, repeat_idx, :, :] = \
                output[repeat_idx]['average_signal_list'][:, :]

    print('saving data')
    with open('exp_results/base_sim/base_sim_classification_{}.pkl'.format(sim_specification), 'wb') as output:
        pickle.dump(CMS_specific_results_classification, output)
    with open('exp_results/base_sim/base_sim_num_cells_{}.pkl'.format(sim_specification), 'wb') as output:
        pickle.dump(CMS_specific_results_num_cells, output)
    with open('exp_results/base_sim/base_sim_inflammation_{}.pkl'.format(sim_specification), 'wb') as output:
        pickle.dump(CMS_specific_results_inflammation, output)
    with open('exp_results/base_sim/base_sim_pathway_activation_{}.pkl'.format(sim_specification), 'wb') as output:
        pickle.dump(CMS_specific_results_pathway_activation, output)
    with open('exp_results/base_sim/base_sim_signal{}.pkl'.format(sim_specification), 'wb') as output:
        pickle.dump(CMS_specific_results_signal_list, output)

    with open('exp_results/base_sim/base_sim_classification_{}.pkl'.format(sim_specification), 'rb') as output:
        CMS_specific_results = pickle.load(output)
        for cms_idx, cms in enumerate(cms_types):
            cms_mean = CMS_specific_results[cms_idx, :, :, :].mean(axis=0)

            cms_std = CMS_specific_results[cms_idx, :, :, :].std(axis=0)

            cms_colors = ['blue', 'orange', 'green', 'red']
            cms_names = ['CMS1', 'CMS2', 'CMS3', 'CMS4']

            plt.figure()
            for idx, (color, cms_plot) in enumerate(list(zip(*(cms_colors, cms_names)))):
                plt.errorbar(range(len(cms_mean))[0::10], cms_mean[::10, idx], cms_std[::10, idx], linestyle='-',
                             marker='o',
                             label=cms_plot,
                             markersize=1, color=color)

            plt.legend()
            plt.grid()
            plt.xlabel('Time (h)', fontsize=14)
            plt.ylabel('CMS classification (#)', fontsize=14)
            plt.savefig('exp_results/base_sim/base_sim_repeated_classification_{}.pdf'.format(cms))
            plt.show()

    with open('exp_results/base_sim/base_sim_num_cells_{}.pkl'.format(sim_specification), 'rb') as output:
        CMS_specific_results = pickle.load(output)
        for cms_idx, cms in enumerate(cms_types):
            cms_mean = CMS_specific_results[cms_idx, :, :, :].mean(axis=0)

            cms_std = CMS_specific_results[cms_idx, :, :, :].std(axis=0)

            cell_colors = ['brown', 'blue', 'pink', 'green', 'red']
            cell_names = ['Cancer', 'T cell', 'B cell', 'Myeloid', 'Stromal cells']

            plt.figure()
            for idx, (color, cell) in enumerate(list(zip(*(cell_colors, cell_names)))):
                plt.errorbar(range(len(cms_mean))[0::10], cms_mean[::10, idx], cms_std[::10, idx], linestyle='-',
                             marker='o',
                             label=cell,
                             markersize=1, color=color)
            plt.legend()
            plt.xlabel('Time (h)', fontsize=14)
            plt.ylabel('Count (#)', fontsize=14)
            plt.savefig('exp_results/base_sim/base_sim_repeated_cell_count{}.pdf'.format(cms))
            plt.show()


def vary_oxygen_supply():
    return_data = True
    sim_specification = '2'

    n_repeats = 20

    # set variables
    grid_size = 100
    n_steps = 500
    n_cells = 20000

    # set settings for the proliferation plots
    proliferation_plot = False
    proliferation_cell = 'tumour'
    replenish_time = 24
    prediffusion_step = 1

    # cancer cell, myeloid, tcell, bcell, stromal
    resources_init_value, oxygen_init_value, ros_init_value, signals_init_value = load_initial_grid_parameters()
    oxygen_values = [1 * 10 ** -13, 5 * 10 ** -13, 10 * 10 ** -13, 15 * 10 ** -13, 20 * 10 ** -13, 25 * 10 ** -13,
                     30 * 10 ** -13, 40 * 10 ** -13, 50 * 10 ** -13, 60 * 10 ** -13]
    influx_list = []

    for ox_val in oxygen_values:
        influx_list.append({
            'oxygen': ox_val,
            'nutrient': 5 * 10 ** -15
        })

    threshold_dict = load_pathway_activation_thresholds()
    params_tumour, params_tcell, params_bcell, params_myeloid, params_stromal = load_cell_parameters(threshold_dict)

    # CMS_specific_results_classification = np.zeros((len(oxygen_values), n_repeats, n_steps + 1, 4))
    # CMS_specific_results_num_cells = np.zeros((len(oxygen_values), n_repeats, n_steps + 1, 5))
    # CMS_specific_results_inflammation = np.zeros((len(oxygen_values), n_repeats, n_steps + 1))
    # CMS_specific_results_pathway_activation = np.zeros((len(oxygen_values), n_repeats, n_steps + 1, 15))
    # CMS_specific_results_signal_list = np.zeros((len(oxygen_values), n_repeats, n_steps + 1, 14))
    #
    # for influx_idx, influxes in enumerate(tqdm.tqdm(influx_list, position=0, desc='ox_value',
    #                                    colour='blue', leave=True)):
    #     params_tumour, params_tcell, params_bcell, params_myeloid, params_stromal = load_inherent_signal_production(
    #         'cms4', params_tumour, params_tcell, params_bcell, params_myeloid, params_stromal)
    #     fraction_types = load_cell_fractions('cms4')
    #
    #     input = create_input_proliferation(n_repeats, n_steps, grid_size, n_cells, 'cms4', fraction_types,
    #                         params_tumour, params_tcell, params_bcell,
    #                         params_myeloid, params_stromal, resources_init_value,
    #                         oxygen_init_value, ros_init_value, signals_init_value,
    #                         influxes, False,
    #                         False, False, False, False,
    #                         proliferation_plot, proliferation_cell,
    #                         replenish_time, prediffusion_step, return_data, False)
    #
    #     output = run_sim_parallel(input)
    #
    #     for repeat_idx in range(n_repeats):
    #         CMS_specific_results_classification[influx_idx, repeat_idx, :, :] = \
    #             output[repeat_idx]['cms_classification_list'][:, :]
    #         CMS_specific_results_num_cells[influx_idx, repeat_idx, :, :] = \
    #             output[repeat_idx]['num_cells_list'][:, :]
    #         CMS_specific_results_inflammation[influx_idx, repeat_idx, :] = \
    #             output[repeat_idx]['inflammation_list'][:]
    #         CMS_specific_results_pathway_activation[influx_idx, repeat_idx, :, :] = \
    #             output[repeat_idx]['pathway_activation_list'][:, :]
    #         CMS_specific_results_signal_list[influx_idx, repeat_idx, :, :] = \
    #             output[repeat_idx]['average_signal_list'][:, :]
    #
    # print('saving data')
    # with open('exp_results/ox_var/ox_var_classification_{}.pkl'.format(sim_specification), 'wb') as output:
    #     pickle.dump(CMS_specific_results_classification, output)
    # with open('exp_results/ox_var/ox_var_num_cells_{}.pkl'.format(sim_specification), 'wb') as output:
    #     pickle.dump(CMS_specific_results_num_cells, output)
    # with open('exp_results/ox_var/ox_var_inflammation_{}.pkl'.format(sim_specification), 'wb') as output:
    #     pickle.dump(CMS_specific_results_inflammation, output)
    # with open('exp_results/ox_var/ox_var_pathway_activation_{}.pkl'.format(sim_specification), 'wb') as output:
    #     pickle.dump(CMS_specific_results_pathway_activation, output)
    # with open('exp_results/ox_var/ox_var_signal{}.pkl'.format(sim_specification), 'wb') as output:
    #     pickle.dump(CMS_specific_results_signal_list, output)
    #
    with open('exp_results/ox_var/ox_var_classification_{}.pkl'.format(sim_specification), 'rb') as output:
        CMS_specific_results = pickle.load(output)

        end_cms_classification_mean = np.zeros((len(oxygen_values), 4))
        end_cms_classification_std = np.zeros((len(oxygen_values), 4))

        for ox_idx, ox_val in enumerate(oxygen_values):
            cms_mean = CMS_specific_results[ox_idx, :, :, :].mean(axis=0)
            cms_std = CMS_specific_results[ox_idx, :, :, :].std(axis=0)
            end_cms_classification_mean[ox_idx, :] = cms_mean[-1, :]
            end_cms_classification_std[ox_idx, :] = cms_std[-1, :]

        cms_colors = ['blue', 'orange', 'green', 'red']
        cms_names = ['CMS1', 'CMS2', 'CMS3', 'CMS4']

        plt.figure()
        for idx, (color, cms) in enumerate(list(zip(*(cms_colors, cms_names)))):
            plt.errorbar(oxygen_values, end_cms_classification_mean[:, idx], end_cms_classification_std[:, idx],
                         linestyle='-',
                         marker='o',
                         label=cms,
                         markersize=1, color=color)

        plt.legend()
        plt.xlabel(r'Oxygen influx ($\mathrm{mol} h^{-1} \mathrm{grid}_{\mathrm{vessel}}^{-1}$)', fontsize=14)
        plt.ylabel('CMS classification (#)', fontsize=14)
        plt.savefig('exp_results/ox_var/ox_influx_classification_{}.pdf'.format(cms))
        plt.show()

    with open('exp_results/ox_var/ox_var_num_cells_{}.pkl'.format(sim_specification), 'rb') as output:
        CMS_specific_results = pickle.load(output)

        end_cms_classification_mean = np.zeros((len(oxygen_values), 5))
        end_cms_classification_std = np.zeros((len(oxygen_values), 5))

        for ox_idx, ox_val in enumerate(oxygen_values):
            cms_mean = CMS_specific_results[ox_idx, :, :, :].mean(axis=0)
            cms_std = CMS_specific_results[ox_idx, :, :, :].std(axis=0)
            end_cms_classification_mean[ox_idx, :] = cms_mean[-1, :]
            end_cms_classification_std[ox_idx, :] = cms_std[-1, :]

        cell_colors = ['brown', 'blue', 'pink', 'green', 'red']
        cell_names = ['Cancer', 'T cell', 'B cell', 'Myeloid', 'Stromal cells']

        plt.figure()
        for idx, (color, cms) in enumerate(list(zip(*(cell_colors, cell_names)))):
            plt.errorbar(oxygen_values, end_cms_classification_mean[:, idx], end_cms_classification_std[:, idx],
                         linestyle='-',
                         marker='o',
                         label=cms,
                         markersize=1, color=color)

        plt.legend()
        plt.xlabel(r'Oxygen influx ($\mathrm{mol} h^{-1} \mathrm{grid}_{\mathrm{vessel}}^{-1}$)', fontsize=14)
        plt.ylabel('Cell count (#)', fontsize=14)
        plt.savefig('exp_results/ox_var/ox_influx_cell_count_{}.pdf'.format(sim_specification))
        plt.show()


def vary_nutrient_supply():
    return_data = True
    sim_specification = '2'

    n_repeats = 30

    # set variables
    grid_size = 100
    n_steps = 500
    n_cells = 20000

    # set settings for the proliferation plots
    proliferation_plot = False
    proliferation_cell = 'tumour'
    replenish_time = 24
    prediffusion_step = 1

    # cancer cell, myeloid, tcell, bcell, stromal
    resources_init_value, oxygen_init_value, ros_init_value, signals_init_value = load_initial_grid_parameters()
    nutrient_values = [0.001 * 10 ** -15, 0.0025 * 10 ** -15, 0.005 * 10 ** -15, 0.0075 * 10 ** -15,
                       0.01 * 10 **
                       -15,
                       0.025 * 10 ** -15, 0.05 * 10 ** -15, 0.075 * 10 ** -15]

    # nutrient_values_low = [0.001 * 10 ** -15, 0.0025 * 10 ** -15, 0.005 * 10 ** -15, 0.0075 * 10 ** -15,
    #                    0.01 * 10 **
    #                    -15,
    #                    0.025 * 10 ** -15, 0.05 * 10 ** -15, 0.075 * 10 ** -15]
    #
    #
    # nutrient_values_high = [0.1 * 10 ** -15, 0.5 * 10 ** -15, 1 * 10 ** -15, 2.5 * 10 ** -15, 5 * 10 ** -15, 10 * 10 **
    #                      -15,
    #                    15 * 10 ** -15, 20 * 10 ** -15, 25 * 10 ** -15]

    influx_list = []

    for nut_val in nutrient_values:
        influx_list.append({
            'oxygen': 100 * 10 ** -13,
            'nutrient': nut_val
        })

    threshold_dict = load_pathway_activation_thresholds()
    params_tumour, params_tcell, params_bcell, params_myeloid, params_stromal = load_cell_parameters(threshold_dict)

    # CMS_specific_results_classification = np.zeros((len(nutrient_values), n_repeats, n_steps + 1, 4))
    # CMS_specific_results_num_cells = np.zeros((len(nutrient_values), n_repeats, n_steps + 1, 5))
    # CMS_specific_results_inflammation = np.zeros((len(nutrient_values), n_repeats, n_steps + 1))
    # CMS_specific_results_pathway_activation = np.zeros((len(nutrient_values), n_repeats, n_steps + 1, 15))
    # CMS_specific_results_signal_list = np.zeros((len(nutrient_values), n_repeats, n_steps + 1, 14))
    #
    # for influx_idx, influxes in enumerate(tqdm.tqdm(influx_list, position=0, desc='nut_value',
    #                                    colour='blue', leave=True)):
    #     params_tumour, params_tcell, params_bcell, params_myeloid, params_stromal = load_inherent_signal_production(
    #         'cms4', params_tumour, params_tcell, params_bcell, params_myeloid, params_stromal)
    #     fraction_types = load_cell_fractions('cms4')
    #     print('the current influxes: ', influxes)
    #
    #     input = create_input_proliferation(n_repeats, n_steps, grid_size, n_cells, 'cms4', fraction_types,
    #                         params_tumour, params_tcell, params_bcell,
    #                         params_myeloid, params_stromal, resources_init_value,
    #                         oxygen_init_value, ros_init_value, signals_init_value,
    #                         influxes, False,
    #                         False, False, False, False,
    #                         proliferation_plot, proliferation_cell,
    #                         replenish_time, prediffusion_step, return_data, False)
    #
    #     output = run_sim_parallel(input)
    #
    #     for repeat_idx in range(n_repeats):
    #         CMS_specific_results_classification[influx_idx, repeat_idx, :, :] = \
    #             output[repeat_idx]['cms_classification_list'][:, :]
    #         CMS_specific_results_num_cells[influx_idx, repeat_idx, :, :] = \
    #             output[repeat_idx]['num_cells_list'][:, :]
    #         CMS_specific_results_inflammation[influx_idx, repeat_idx, :] = \
    #             output[repeat_idx]['inflammation_list'][:]
    #         CMS_specific_results_pathway_activation[influx_idx, repeat_idx, :, :] = \
    #             output[repeat_idx]['pathway_activation_list'][:, :]
    #         CMS_specific_results_signal_list[influx_idx, repeat_idx, :, :] = \
    #             output[repeat_idx]['average_signal_list'][:, :]
    #
    # print('saving data')
    # with open('exp_results/nut_var/nut_var_classification_{}.pkl'.format(sim_specification), 'wb') as output:
    #     pickle.dump(CMS_specific_results_classification, output)
    # with open('exp_results/nut_var/nut_var_num_cells_{}.pkl'.format(sim_specification), 'wb') as output:
    #     pickle.dump(CMS_specific_results_num_cells, output)
    # with open('exp_results/nut_var/nut_var_inflammation_{}.pkl'.format(sim_specification), 'wb') as output:
    #     pickle.dump(CMS_specific_results_inflammation, output)
    # with open('exp_results/nut_var/nut_var_pathway_activation_{}.pkl'.format(sim_specification), 'wb') as output:
    #     pickle.dump(CMS_specific_results_pathway_activation, output)
    # with open('exp_results/nut_var/nut_var_signal{}.pkl'.format(sim_specification), 'wb') as output:
    #     pickle.dump(CMS_specific_results_signal_list, output)

    with open('exp_results/nut_var/nut_var_classification_{}.pkl'.format(sim_specification), 'rb') as output:
        CMS_specific_results = pickle.load(output)

        end_cms_classification_mean = np.zeros((len(nutrient_values), 4))
        end_cms_classification_std = np.zeros((len(nutrient_values), 4))

        for ox_idx, ox_val in enumerate(nutrient_values):
            cms_mean = CMS_specific_results[ox_idx, :, :, :].mean(axis=0)
            cms_std = CMS_specific_results[ox_idx, :, :, :].std(axis=0)
            end_cms_classification_mean[ox_idx, :] = cms_mean[-1, :]
            end_cms_classification_std[ox_idx, :] = cms_std[-1, :]

        cms_colors = ['blue', 'orange', 'green', 'red']
        cms_names = ['CMS1', 'CMS2', 'CMS3', 'CMS4']

        plt.figure()
        for idx, (color, cms) in enumerate(list(zip(*(cms_colors, cms_names)))):
            plt.errorbar(nutrient_values, end_cms_classification_mean[:, idx], end_cms_classification_std[:, idx],
                         linestyle='-',
                         marker='o',
                         label=cms,
                         markersize=1, color=color)

        plt.legend()
        plt.xlabel(r'Oxygen influx ($\mathrm{mol} h^{-1} \mathrm{grid}_{\mathrm{vessel}}^{-1}$)', fontsize=14)
        plt.ylabel('CMS classification (#)', fontsize=14)
        plt.savefig('exp_results/nut_var/nut_influx_classification_{}.pdf'.format(cms))
        plt.show()


def vary_inflammation():
    return_data = True
    sim_specification = '1'

    n_repeats = 30

    # set variables
    grid_size = 100
    n_steps = 500
    n_cells = 20000

    # set settings for the proliferation plots
    proliferation_plot = False
    proliferation_cell = 'tumour'
    replenish_time = 24
    prediffusion_step = 1
    influxes = load_resource_influx(proliferation_plot)

    # cancer cell, myeloid, tcell, bcell, stromal
    resources_init_value, oxygen_init_value, ros_init_value, signals_init_value = load_initial_grid_parameters()
    inflammation_values = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
    #
    threshold_dict = load_pathway_activation_thresholds()
    # params_tumour, params_tcell, params_bcell, params_myeloid, params_stromal = load_cell_parameters(threshold_dict)
    #
    # CMS_specific_results_classification = np.zeros((len(inflammation_values), n_repeats, n_steps + 1, 4))
    # CMS_specific_results_num_cells = np.zeros((len(inflammation_values), n_repeats, n_steps + 1, 5))
    # CMS_specific_results_inflammation = np.zeros((len(inflammation_values), n_repeats, n_steps + 1))
    # CMS_specific_results_pathway_activation = np.zeros((len(inflammation_values), n_repeats, n_steps + 1, 15))
    # CMS_specific_results_signal_list = np.zeros((len(inflammation_values), n_repeats, n_steps + 1, 14))
    #
    # for inflam_idx, inflam_value in enumerate(tqdm.tqdm(inflammation_values, position=0, desc='inflam_value',
    #                                    colour='blue', leave=True)):
    #     params_tumour, params_tcell, params_bcell, params_myeloid, params_stromal = load_inherent_signal_production(
    #         'cms4', params_tumour, params_tcell, params_bcell, params_myeloid, params_stromal)
    #     fraction_types = load_cell_fractions('cms4')
    #     print('the current inflammation value: ', inflam_value)
    #
    #     input = create_input_proliferation(n_repeats, n_steps, grid_size, n_cells, 'cms4', fraction_types,
    #                         params_tumour, params_tcell, params_bcell,
    #                         params_myeloid, params_stromal, resources_init_value,
    #                         oxygen_init_value, ros_init_value, signals_init_value,
    #                         influxes, False,
    #                         False, False, False, False,
    #                         proliferation_plot, proliferation_cell,
    #                         replenish_time, prediffusion_step, return_data, inflam_value)
    #
    #     output = run_sim_parallel(input)
    #
    #     for repeat_idx in range(n_repeats):
    #         CMS_specific_results_classification[inflam_idx, repeat_idx, :, :] = \
    #             output[repeat_idx]['cms_classification_list'][:, :]
    #         CMS_specific_results_num_cells[inflam_idx, repeat_idx, :, :] = \
    #             output[repeat_idx]['num_cells_list'][:, :]
    #         CMS_specific_results_inflammation[inflam_idx, repeat_idx, :] = \
    #             output[repeat_idx]['inflammation_list'][:]
    #         CMS_specific_results_pathway_activation[inflam_idx, repeat_idx, :, :] = \
    #             output[repeat_idx]['pathway_activation_list'][:, :]
    #         CMS_specific_results_signal_list[inflam_idx, repeat_idx, :, :] = \
    #             output[repeat_idx]['average_signal_list'][:, :]
    #
    # print('saving data')
    # with open('exp_results/inflam_var/inflam_var_classification_{}.pkl'.format(sim_specification), 'wb') as output:
    #     pickle.dump(CMS_specific_results_classification, output)
    # with open('exp_results/inflam_var/inflam_var_num_cells_{}.pkl'.format(sim_specification), 'wb') as output:
    #     pickle.dump(CMS_specific_results_num_cells, output)
    # with open('exp_results/inflam_var/inflam_var_inflammation_{}.pkl'.format(sim_specification), 'wb') as output:
    #     pickle.dump(CMS_specific_results_inflammation, output)
    # with open('exp_results/inflam_var/inflam_var_pathway_activation_{}.pkl'.format(sim_specification), 'wb') as output:
    #     pickle.dump(CMS_specific_results_pathway_activation, output)
    # with open('exp_results/inflam_var/inflam_var_signal{}.pkl'.format(sim_specification), 'wb') as output:
    #     pickle.dump(CMS_specific_results_signal_list, output)

    with open('exp_results/inflam_var/inflam_var_classification_{}.pkl'.format(sim_specification), 'rb') as output:
        CMS_specific_results = pickle.load(output)

        from scipy.stats import ks_2samp
        dist_compa = ks_2samp(CMS_specific_results[0, :, -1, 3], CMS_specific_results[-2, :, -1, 3])
        print(CMS_specific_results[0, :, -1, 3], CMS_specific_results[-2, :, -1, 3])
        print(dist_compa)

        end_cms_classification_mean = np.zeros((len(inflammation_values), 4))
        end_cms_classification_std = np.zeros((len(inflammation_values), 4))

        for ox_idx, ox_val in enumerate(inflammation_values):
            cms_mean = CMS_specific_results[ox_idx, :, :, :].mean(axis=0)
            cms_std = CMS_specific_results[ox_idx, :, :, :].std(axis=0)
            end_cms_classification_mean[ox_idx, :] = cms_mean[-1, :]
            end_cms_classification_std[ox_idx, :] = cms_std[-1, :]

        cms_colors = ['blue', 'orange', 'green', 'red']
        cms_names = ['CMS1', 'CMS2', 'CMS3', 'CMS4']

        plt.figure()
        for idx, (color, cms) in enumerate(list(zip(*(cms_colors, cms_names)))):
            plt.errorbar(inflammation_values, end_cms_classification_mean[:, idx], end_cms_classification_std[:, idx],
                         linestyle='-',
                         marker='o',
                         label=cms,
                         markersize=1, color=color)

        plt.legend()
        plt.xlabel('Inflammation (a.u.)', fontsize=14)
        plt.ylabel('CMS classification (#)', fontsize=14)
        plt.savefig('exp_results/inflam_var/inflam_classification_{}.pdf'.format(cms))
        plt.show()


def vary_me_fractions():
    sim_specification = '1'

    myeloid_fraction = [0.07, 0.12, 0.17, 0.22, 0.27, 0.32, 0.37, 0.42, 0.47, 0.52]
    # t_cell_fraction = [0.27, 0.32, 0.37, 0.42, 0.47, 0.52, 0.57, 0.62, 0.67, 0.72, 0.77] # first run
    t_cell_fraction = [0.87, 0.97, 1.07, 1.17, 1.27, 1.37, 1.47, 1.57]

    b_cell_fraction = [0.02, 0.07, 0.12, 0.17, 0.22, 0.27, 0.32, 0.37]
    stromal_cell_fraction = [0.01, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27]
    cell_type = 'stromal'
    if cell_type == 'myeloid':
        cell_type_idx = 1
        cell_type_fraction = myeloid_fraction

    elif cell_type == 'tcell':
        cell_type_idx = 2
        cell_type_fraction = t_cell_fraction

    elif cell_type == 'bcell':
        cell_type_idx = 3
        cell_type_fraction = b_cell_fraction

    elif cell_type == 'stromal':
        cell_type_idx = 4
        cell_type_fraction = stromal_cell_fraction

    return_data = True
    n_repeats = 20

    # set variables
    grid_size = 100
    n_steps = 500
    n_cells = 20000

    # set settings for the proliferation plots
    proliferation_plot = False
    proliferation_cell = 'tumour'
    replenish_time = 24
    prediffusion_step = 1

    # cancer cell, myeloid, tcell, bcell, stromal
    resources_init_value, oxygen_init_value, ros_init_value, signals_init_value = load_initial_grid_parameters()
    influxes = load_resource_influx(proliferation_plot)

    # overwrite the initial nutrient deposit to the influx in case of a growth curve experiment
    if proliferation_plot:
        resources_init_value = influxes['nutrient']

    threshold_dict = load_pathway_activation_thresholds()
    params_tumour, params_tcell, params_bcell, params_myeloid, params_stromal = load_cell_parameters(threshold_dict)

    fraction_types_cms4 = load_cell_fractions('cms4')
    base_fraction_cms4 = fraction_types_cms4.copy()
    base_fraction_cms4[1:] = fraction_types_cms4[1:] / sum(fraction_types_cms4[1:])
    fraction_types_cms2 = load_cell_fractions('cms2')
    base_fraction_cms2 = fraction_types_cms2.copy()
    base_fraction_cms2[1:] = fraction_types_cms2[1:] / sum(fraction_types_cms2[1:])
    # np.array([0.045, 0.32, 0.37, 0.12, 0.18])

    fractions_list = np.array([fraction_types_cms4 for i in range(len(cell_type_fraction))])
    fractions_list[:, cell_type_idx] = cell_type_fraction

    for frac_idx, fractions in enumerate(fractions_list):
        fractions_list[frac_idx, 1:] = fractions_list[frac_idx, 1:] / sum(fractions_list[frac_idx, 1:])

    print(fractions_list)

    # CMS_specific_results_classification = np.zeros((len(cell_type_fraction), n_repeats, n_steps + 1, 4))
    # CMS_specific_results_num_cells = np.zeros((len(cell_type_fraction), n_repeats, n_steps + 1, 5))
    # CMS_specific_results_inflammation = np.zeros((len(cell_type_fraction), n_repeats, n_steps + 1))
    # CMS_specific_results_pathway_activation = np.zeros((len(cell_type_fraction), n_repeats, n_steps + 1, 15))
    # CMS_specific_results_signal_list = np.zeros((len(cell_type_fraction), n_repeats, n_steps + 1, 14))
    #
    # for fraction_idx, cms_me_fraction in enumerate(tqdm.tqdm(fractions_list, position=0, desc='fraction', colour='blue',
    #                                                  leave=True)):
    #     params_tumour, params_tcell, params_bcell, params_myeloid, params_stromal = load_inherent_signal_production(
    #         'cms4', params_tumour, params_tcell, params_bcell, params_myeloid, params_stromal)
    #     print(cms_me_fraction)
    #     input = create_input_proliferation(n_repeats, n_steps, grid_size, n_cells, 'cms4', cms_me_fraction,
    #                         params_tumour, params_tcell, params_bcell,
    #                         params_myeloid, params_stromal, resources_init_value,
    #                         oxygen_init_value, ros_init_value, signals_init_value,
    #                         influxes, False,
    #                         False, False, False, False,
    #                         proliferation_plot, proliferation_cell,
    #                         replenish_time, prediffusion_step, return_data, False)
    #
    #     output = run_sim_parallel(input)
    #
    #     for repeat_idx in range(n_repeats):
    #         CMS_specific_results_classification[fraction_idx, repeat_idx, :, :] = \
    #             output[repeat_idx]['cms_classification_list'][:, :]
    #         CMS_specific_results_num_cells[fraction_idx, repeat_idx, :, :] = \
    #             output[repeat_idx]['num_cells_list'][:, :]
    #         CMS_specific_results_inflammation[fraction_idx, repeat_idx, :] = \
    #             output[repeat_idx]['inflammation_list'][:]
    #         CMS_specific_results_pathway_activation[fraction_idx, repeat_idx, :, :] = \
    #             output[repeat_idx]['pathway_activation_list'][:, :]
    #         CMS_specific_results_signal_list[fraction_idx, repeat_idx, :, :] = \
    #             output[repeat_idx]['average_signal_list'][:, :]
    #
    # print('saving data')
    # with open('exp_results/fraction_var/fraction_var_classification_{}_{}.pkl'.format(cell_type,sim_specification),
    #           'wb') as output:
    #     pickle.dump(CMS_specific_results_classification, output)
    # with open('exp_results/fraction_var/fraction_var_num_cells_{}_{}.pkl'.format(cell_type,sim_specification), 'wb') as output:
    #     pickle.dump(CMS_specific_results_num_cells, output)
    # with open('exp_results/fraction_var/fraction_var_inflammation_{}_{}.pkl'.format(cell_type,sim_specification), 'wb') as output:
    #     pickle.dump(CMS_specific_results_inflammation, output)
    # with open('exp_results/fraction_var/fraction_var_pathway_activation_{}_{}.pkl'.format(cell_type,sim_specification),
    #           'wb') as output:
    #     pickle.dump(CMS_specific_results_pathway_activation, output)
    # with open('exp_results/fraction_var/fraction_var_signal_{}_{}.pkl'.format(cell_type,sim_specification), 'wb') as output:
    #     pickle.dump(CMS_specific_results_signal_list, output)

    with open('exp_results/fraction_var/fraction_var_classification_{}_{}.pkl'.format(cell_type, sim_specification),
              'rb') as output:
        CMS_specific_results = pickle.load(output)

        end_cms_classification_mean = np.zeros((len(fractions_list), 4))
        end_cms_classification_std = np.zeros((len(fractions_list), 4))

        for ox_idx, ox_val in enumerate(fractions_list):
            cms_mean = CMS_specific_results[ox_idx, :, :, :].mean(axis=0)
            cms_std = CMS_specific_results[ox_idx, :, :, :].std(axis=0)
            end_cms_classification_mean[ox_idx, :] = cms_mean[-1, :]
            end_cms_classification_std[ox_idx, :] = cms_std[-1, :]

        cms_colors = ['blue', 'orange', 'green', 'red']
        cms_names = ['CMS1', 'CMS2', 'CMS3', 'CMS4']

        plt.figure()
        for idx, (color, cms) in enumerate(list(zip(*(cms_colors, cms_names)))):
            plt.errorbar(fractions_list[:, cell_type_idx], end_cms_classification_mean[:, idx],
                         end_cms_classification_std[:, idx],
                         linestyle='-',
                         marker='o',
                         label=cms,
                         markersize=1, color=color)

        plt.legend()
        plt.xlabel('Stromal fraction in ME'.format(cell_type), fontsize=14)
        plt.ylabel('CMS classification (#)', fontsize=14)
        plt.axvline(x=base_fraction_cms4[cell_type_idx], linestyle='--', color=cms_colors[3])
        plt.axvline(x=base_fraction_cms2[cell_type_idx], linestyle='--', color=cms_colors[1])
        plt.savefig('exp_results/fraction_var/fraction_var_classification_{}_{}.pdf'.format(cell_type,
                                                                                            sim_specification))
        plt.show()


def vary_anti_oxidants():
    return_data = True
    sim_specification = '1'

    n_repeats = 30

    # set variables
    grid_size = 100
    n_steps = 500
    n_cells = 20000

    # set settings for the proliferation plots
    proliferation_plot = False
    proliferation_cell = 'tumour'
    replenish_time = 24
    prediffusion_step = 1
    influxes = load_resource_influx(proliferation_plot)

    # cancer cell, myeloid, tcell, bcell, stromal
    resources_init_value, oxygen_init_value, ros_init_value, signals_init_value = load_initial_grid_parameters()
    ox_modulation_values = [0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7]

    threshold_dict = load_pathway_activation_thresholds()
    params_tumour, params_tcell, params_bcell, params_myeloid, params_stromal = load_cell_parameters(threshold_dict)

    CMS_specific_results_classification = np.zeros((len(ox_modulation_values), n_repeats, n_steps + 1, 4))
    CMS_specific_results_num_cells = np.zeros((len(ox_modulation_values), n_repeats, n_steps + 1, 5))
    CMS_specific_results_inflammation = np.zeros((len(ox_modulation_values), n_repeats, n_steps + 1))
    CMS_specific_results_pathway_activation = np.zeros((len(ox_modulation_values), n_repeats, n_steps + 1, 15))
    CMS_specific_results_signal_list = np.zeros((len(ox_modulation_values), n_repeats, n_steps + 1, 14))

    with open('exp_results/anti_oxidant_var/oxidant_var_classification_{}.pkl'.format(sim_specification),
              'rb') as output:
        CMS_specific_results_classification = pickle.load(output)
    with open('exp_results/anti_oxidant_var/oxidant_var_num_cells_{}.pkl'.format(sim_specification), 'rb') as output:
        CMS_specific_results_num_cells = pickle.load(output)
    with open('exp_results/anti_oxidant_var/oxidant_var_inflammation_{}.pkl'.format(sim_specification), 'rb') as output:
        CMS_specific_results_inflammation = pickle.load(output)
    with open('exp_results/anti_oxidant_var/oxidant_var_pathway_activation_{}.pkl'.format(sim_specification),
              'rb') as output:
        CMS_specific_results_pathway_activation = pickle.load(output)
    with open('exp_results/anti_oxidant_var/oxidant_var_signal{}.pkl'.format(sim_specification), 'rb') as output:
        CMS_specific_results_signal_list = pickle.load(output)

    for ox_modulation_idx, ox_modulation_value in enumerate(tqdm.tqdm(ox_modulation_values, position=0,
                                                                      desc='ox_mod_value',
                                                                      colour='blue', leave=True)):
        if ox_modulation_value > 0.6:
            params_tumour, params_tcell, params_bcell, params_myeloid, params_stromal = load_inherent_signal_production(
                'cms4', params_tumour, params_tcell, params_bcell, params_myeloid, params_stromal)
            fraction_types = load_cell_fractions('cms4')
            print('the current ROS modulation value: ', ox_modulation_value)

            input = create_input_proliferation(n_repeats, n_steps, grid_size, n_cells, 'cms4', fraction_types,
                                               params_tumour, params_tcell, params_bcell,
                                               params_myeloid, params_stromal, resources_init_value,
                                               oxygen_init_value, ros_init_value, signals_init_value,
                                               influxes, False,
                                               False, False, False, False,
                                               proliferation_plot, proliferation_cell,
                                               replenish_time, prediffusion_step, return_data, False,
                                               ox_modulation_value)

            output = run_sim_parallel(input)

            for repeat_idx in range(n_repeats):
                CMS_specific_results_classification[ox_modulation_idx, repeat_idx, :, :] = \
                    output[repeat_idx]['cms_classification_list'][:, :]
                CMS_specific_results_num_cells[ox_modulation_idx, repeat_idx, :, :] = \
                    output[repeat_idx]['num_cells_list'][:, :]
                CMS_specific_results_inflammation[ox_modulation_idx, repeat_idx, :] = \
                    output[repeat_idx]['inflammation_list'][:]
                CMS_specific_results_pathway_activation[ox_modulation_idx, repeat_idx, :, :] = \
                    output[repeat_idx]['pathway_activation_list'][:, :]
                CMS_specific_results_signal_list[ox_modulation_idx, repeat_idx, :, :] = \
                    output[repeat_idx]['average_signal_list'][:, :]

            print('saving data')
            with open('exp_results/anti_oxidant_var/oxidant_var_classification_{}.pkl'.format(sim_specification),
                      'wb') as output:
                pickle.dump(CMS_specific_results_classification, output)
            with open('exp_results/anti_oxidant_var/oxidant_var_num_cells_{}.pkl'.format(sim_specification),
                      'wb') as output:
                pickle.dump(CMS_specific_results_num_cells, output)
            with open('exp_results/anti_oxidant_var/oxidant_var_inflammation_{}.pkl'.format(sim_specification),
                      'wb') as output:
                pickle.dump(CMS_specific_results_inflammation, output)
            with open('exp_results/anti_oxidant_var/oxidant_var_pathway_activation_{}.pkl'.format(sim_specification),
                      'wb') as output:
                pickle.dump(CMS_specific_results_pathway_activation, output)
            with open('exp_results/anti_oxidant_var/oxidant_var_signal{}.pkl'.format(sim_specification),
                      'wb') as output:
                pickle.dump(CMS_specific_results_signal_list, output)

    with open('exp_results/anti_oxidant_var/oxidant_var_classification_{}.pkl'.format(sim_specification),
              'rb') as output:
        CMS_specific_results = pickle.load(output)

        end_cms_classification_mean = np.zeros((len(ox_modulation_values), 4))
        end_cms_classification_std = np.zeros((len(ox_modulation_values), 4))

        for ox_idx, ox_val in enumerate(ox_modulation_values):
            cms_mean = CMS_specific_results[ox_idx, :, :, :].mean(axis=0)
            cms_std = CMS_specific_results[ox_idx, :, :, :].std(axis=0)
            end_cms_classification_mean[ox_idx, :] = cms_mean[-1, :]
            end_cms_classification_std[ox_idx, :] = cms_std[-1, :]

        cms_colors = ['blue', 'orange', 'green', 'red']
        cms_names = ['CMS1', 'CMS2', 'CMS3', 'CMS4']

        plt.figure()
        for idx, (color, cms) in enumerate(list(zip(*(cms_colors, cms_names)))):
            plt.errorbar(np.array(ox_modulation_values) / 1.5, end_cms_classification_mean[:, idx],
                         end_cms_classification_std[:,idx], linestyle='-',
                         marker='o',
                         label=cms,
                         markersize=1, color=color)

        plt.legend()
        plt.xlabel(r'ROS reduction ($R^{prod}_{base}$)', fontsize=14)
        plt.ylabel('CMS classification (#)', fontsize=14)
        plt.savefig('exp_results/anti_oxidant_var/oxidant_classification_{}.pdf'.format(cms))
        plt.show()



def vary_pathway_activation_thresholds():
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

    threshold_type = 'metabolistic_signalling_threshold'

    return_data = True
    n_repeats = 30

    # set variables
    grid_size = 100
    n_steps = 500
    n_cells = 20000

    # set settings for the proliferation plots
    proliferation_plot = False
    proliferation_cell = 'tumour'
    replenish_time = 24
    prediffusion_step = 1

    # cancer cell, myeloid, tcell, bcell, stromal
    resources_init_value, oxygen_init_value, ros_init_value, signals_init_value = load_initial_grid_parameters()
    influxes = load_resource_influx(proliferation_plot)

    # overwrite the initial nutrient deposit to the influx in case of a growth curve experiment
    if proliferation_plot:
        resources_init_value = influxes['nutrient']

    threshold_dict = load_pathway_activation_thresholds()

    threshold_dicts = []

    for percentage_change in [-0.12, -0.08, -0.04, 0, 0.04, 0.08, 0.12]:
        new_threshold_dict = threshold_dict.copy()
        new_threshold_dict[threshold_type] = threshold_dict[threshold_type] + \
                                             threshold_dict[threshold_type] * percentage_change
        threshold_dicts.append(new_threshold_dict)

    fraction_types_cms4 = load_cell_fractions('cms4')

    CMS_specific_results_classification = np.zeros((len(threshold_dicts), n_repeats, n_steps + 1, 4))
    CMS_specific_results_num_cells = np.zeros((len(threshold_dicts), n_repeats, n_steps + 1, 5))
    CMS_specific_results_inflammation = np.zeros((len(threshold_dicts), n_repeats, n_steps + 1))
    CMS_specific_results_pathway_activation = np.zeros((len(threshold_dicts), n_repeats, n_steps + 1, 15))
    CMS_specific_results_signal_list = np.zeros((len(threshold_dicts), n_repeats, n_steps + 1, 14))

    for threshold_idx, threshold_dict_changed in enumerate(tqdm.tqdm(threshold_dicts, position=0, desc='threshold '
                                                                                                     'changed',
                                                             colour='blue', leave=True)):

        params_tumour, params_tcell, params_bcell, params_myeloid, params_stromal = load_cell_parameters(threshold_dict_changed)

        params_tumour, params_tcell, params_bcell, params_myeloid, params_stromal = load_inherent_signal_production(
            'cms4', params_tumour, params_tcell, params_bcell, params_myeloid, params_stromal)

        input = create_input_proliferation(n_repeats, n_steps, grid_size, n_cells, 'cms4', fraction_types_cms4,
                            params_tumour, params_tcell, params_bcell,
                            params_myeloid, params_stromal, resources_init_value,
                            oxygen_init_value, ros_init_value, signals_init_value,
                            influxes, False,
                            False, False, False, False,
                            proliferation_plot, proliferation_cell,
                            replenish_time, prediffusion_step, return_data, False, 1.5)

        output = run_sim_parallel(input)

        for repeat_idx in range(n_repeats):
            CMS_specific_results_classification[threshold_idx, repeat_idx, :, :] = \
                output[repeat_idx]['cms_classification_list'][:, :]
            CMS_specific_results_num_cells[threshold_idx, repeat_idx, :, :] = \
                output[repeat_idx]['num_cells_list'][:, :]
            CMS_specific_results_inflammation[threshold_idx, repeat_idx, :] = \
                output[repeat_idx]['inflammation_list'][:]
            CMS_specific_results_pathway_activation[threshold_idx, repeat_idx, :, :] = \
                output[repeat_idx]['pathway_activation_list'][:, :]
            CMS_specific_results_signal_list[threshold_idx, repeat_idx, :, :] = \
                output[repeat_idx]['average_signal_list'][:, :]

    print('saving data')
    with open('exp_results/pathway_activation_thresholds/pathway_activation_thresholds_classification_{}_{}.pkl'.format(threshold_type,sim_specification),
              'wb') as output:
        pickle.dump(CMS_specific_results_classification, output)
    with open('exp_results/pathway_activation_thresholds/pathway_activation_thresholds_num_cells_{}_{}.pkl'.format(
            threshold_type,sim_specification), 'wb') as output:
        pickle.dump(CMS_specific_results_num_cells, output)
    with open('exp_results/pathway_activation_thresholds/pathway_activation_thresholds_{}_{}.pkl'.format(threshold_type,sim_specification), 'wb') as output:
        pickle.dump(CMS_specific_results_inflammation, output)
    with open('exp_results/pathway_activation_thresholds/pathway_activation_thresholds_{}_{}.pkl'.format(threshold_type,sim_specification),
              'wb') as output:
        pickle.dump(CMS_specific_results_pathway_activation, output)
    with open('exp_results/pathway_activation_thresholds/pathway_activation_thresholds_{}_{}.pkl'.format(threshold_type,sim_specification), 'wb') as output:
        pickle.dump(CMS_specific_results_signal_list, output)

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

        plt.figure()
        for idx, (color, cms) in enumerate(list(zip(*(cms_colors, cms_names)))):
            plt.errorbar([threshold_dict[threshold_type] for threshold_dict in threshold_dicts], end_cms_classification_mean[:, idx],
                         end_cms_classification_std[:, idx],
                         linestyle='-',
                         marker='o',
                         label=cms,
                         markersize=1, color=color)

        plt.legend()
        plt.xlabel('{}'.format(threshold_type), fontsize=14)
        plt.ylabel('CMS classification (#)', fontsize=14)
        # plt.axvline(x=threshold_dicts[cell_type_idx], linestyle='--', color=cms_colors[3])
        # plt.axvline(x=threshold_dicts[cell_type_idx], linestyle='--', color=cms_colors[1])
        plt.savefig('exp_results/pathway_activation_thresholds/pathway_activation_thresholds_classification_{}_{}.pdf'.format(threshold_type,
                                                                                            sim_specification))
        plt.show()


if __name__ == '__main__':
    vary_pathway_activation_thresholds()
