"""
This file contains the functions for plotting
Author: Robin van den Berg
Contact: rvdb7345@gmail.com
"""

import matplotlib
from matplotlib import colors
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import os
import cv2
from cell_definitions import cell, cancercell, myeloid, tcell, bcell, stromalcell
import numpy as np


# get type ids of cells for visualisation
MAP_CLASSES_TO_IDS = np.vectorize(lambda x: 0 if x == 0 else (
    -1 if x.cell_state == 0 else (
        1 if isinstance(x, cancercell) else
        (2 if isinstance(x, tcell) else
         (3 if isinstance(x, bcell) else
          (4 if isinstance(x, myeloid) else
           (5 if isinstance(x, stromalcell) else 'not recognised'
            )))))))

# get the mass of all cells on grid
GET_MASS = np.vectorize(lambda x: 0. if x == 0. else float(x.mass))


CMS_COLOURS = ['orange', 'blue', 'pink', 'green']

def create_final_plots(init_time, num_cells_list, cms_classification_list, average_mass_list, inflammation_list,
                       pathway_list, pathway_activation_list, average_signal_list, all_signals, reasons_cell_death,
                       average_ros_list, n_isolated_tumour_list):
    """ This function creates and saves the plot depicting the metrics throughout the simulation """

    matplotlib.use('module://backend_interagg')

    cell_colors = ['brown', 'blue', 'pink', 'green', 'red']
    cell_names = ['Cancer', 'T cell', 'B cell', 'Myeloid', 'Stromal cells']

    plt.figure(figsize = (10,5))
    plt.title('Cells over time')
    for idx in range(len(num_cells_list[0,:])):
        plt.plot(num_cells_list[:, idx], color=cell_colors[idx], label=cell_names[idx])
    plt.ylabel('Number of cells', fontsize=16)
    plt.xlabel('Time (h)', fontsize=16)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('sim_results/sim_{}/numcellsovertime.pdf'.format(init_time))
    plt.show()

    plt.figure(figsize = (10,5))
    plt.title('Signals over time')

    linestyle_str = ['solid', 'dotted', 'dashed']
    distingishable_colors = ['#800000', 'blue', '#FFE119', '#DCBEFF', '#A9A9A9']

    all_signal_formatted = [r'NF$\kappa$B', 'Insulin', 'EGF', 'WNT', 'EMT sig', 'STAT3', 'IL2', r'TNF$\alpha$',
                            r'IFN$\gamma$', 'IL6', 'MYC', 'SHH', 'Meta sig', r'IFN$\alpha$', 'OPN']

    dist_color_list = distingishable_colors * 3
    for idx, signal in enumerate(all_signal_formatted):
        line_style_idx = int(np.floor(idx / len(distingishable_colors)))
        plt.plot(average_signal_list[:, idx], linestyle=linestyle_str[line_style_idx],
                 color=dist_color_list[idx], label=signal)
    plt.ylabel('Signal level', fontsize=16)
    plt.xlabel('Time (h)', fontsize=16)
    plt.legend(all_signal_formatted, loc='upper center', bbox_to_anchor=(0.5, -0.18), fontsize=14, ncol=6)
    plt.yscale('log')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig('sim_results/sim_{}/averagesignalovertime.pdf'.format(init_time))
    plt.show()

    plt.figure(figsize = (10,5))
    plt.title('Classification over time')
    plt.gca().set_prop_cycle(color=CMS_COLOURS)
    plt.plot(cms_classification_list)
    plt.ylabel('Regression values', fontsize=16)
    plt.xlabel('Time (h)', fontsize=16)
    plt.legend(['CMS1', 'CMS2', 'CMS3', 'CMS4'], fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig('sim_results/sim_{}/classificationovertime.pdf'.format(init_time))
    plt.show()

    plt.figure(figsize = (10,5))
    plt.plot(average_mass_list)
    plt.title('Average mass over time')
    plt.ylabel('Average mass', fontsize=16)
    plt.xlabel('Time (h)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig('sim_results/sim_{}/massovertime.pdf'.format(init_time))
    plt.show()

    plt.figure(figsize = (10,5))
    plt.plot(n_isolated_tumour_list)
    plt.title('Number of isolated tumour cells over time')
    plt.ylabel('Isolated tumour cells (#)', fontsize=16)
    plt.xlabel('Time (h)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig('sim_results/sim_{}/isotumourovertime.pdf'.format(init_time))
    plt.show()

    plt.figure(figsize = (10,5))
    plt.plot(average_ros_list)
    plt.title('Average ROS over time')
    plt.ylabel('Average ROS', fontsize=16)
    plt.xlabel('Time (h)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig('sim_results/sim_{}/rosovertime.pdf'.format(init_time))
    plt.show()

    plt.figure(figsize = (10,5))
    plt.plot(inflammation_list)
    plt.title('System inflammation over time')
    plt.ylabel('Inflammation of the system', fontsize=16)
    plt.xlabel('Time (h)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig('sim_results/sim_{}/inflammationovertime.pdf'.format(init_time))
    plt.show()

    linestyle_str = ['solid', 'dotted', 'dashed', 'dashdot']
    distingishable_colors = ['#800000', '#F58231', '#FFE119', '#DCBEFF', '#A9A9A9']

    pathway_list_formatted = ['P53', 'mTORC1', r'TNF$\alpha$viaNF$\kappa$B',
                    'Hypoxia', 'EMT', 'MYC',
                    'ROS', 'IL2_STAT5',
                    'Adipo', r'IFN$\gamma$', 'kras',
                    'IL6_JAK', 'Complement', r'IFN$\alpha$', 'PI3K']

    dist_color_list = distingishable_colors * 4
    plt.figure(figsize = (10,5))
    for idx, pathway in enumerate(pathway_list_formatted):
        line_style_idx = int(np.floor(idx / len(distingishable_colors)))
        plt.plot(pathway_activation_list[:, idx], linestyle=linestyle_str[line_style_idx],
                 color=dist_color_list[idx], label=pathway)
    plt.title('Average pathway activation over time')
    plt.ylabel('Average expression', fontsize=16)
    plt.xlabel('Time (h)', fontsize=16)
    plt.legend(pathway_list_formatted, loc='upper center', bbox_to_anchor=(0.5, -0.18), fontsize=14, ncol=5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig('sim_results/sim_{}/pathwaysovertime.pdf'.format(init_time))
    plt.show()

    # ['ducts', 'dead', 'empty', 'tumour', 'T cell', 'B cell', 'myeloid', 'stromal']
    reason_to_death_pd = pd.DataFrame(reasons_cell_death)
    reason_to_death_pd.rename({'cancercell': 'Cancer', 'tcell': 'T cell', 'bcell': 'B cell', 'myeloid': 'Myeloid',
                               'stromalcell': 'Stromal cell'}, inplace=True, axis=1)
    reason_to_death_pd.plot(kind='bar', stacked=True,
                            title='Events in simulation', color=cell_colors, fontsize=14)
    plt.savefig('sim_results/sim_{}/death_reasons.pdf'.format(init_time))
    plt.show()


def create_animation(init_time, n_steps):
    """ Create an animation from the snapshots that were made"""

    # check number of available frames
    dir = 'temp_animation/anim_{}/'.format(init_time)

    num_files = len([name for name in os.listdir(dir) if os.path.isfile(dir + name)])
    if num_files < n_steps:
        print('there are {} files while n_steps is {}'.format(num_files, n_steps))
        print('Simulation stopped prematurely, adjusting number of frames')
        n_steps = num_files

    img_array = []
    pbar = tqdm(['temp_animation/anim_{}/frame_{}.png'.format(init_time, i) for i in range(0, n_steps + 1)])
    for filename in pbar:
        pbar.set_description("Processing %s" % filename)
        img = cv2.imread(filename)
        # height, width, layers = img.shape
        # size = (width, height)

        scale_percent = 60  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        img_array.append(resized)

    out = cv2.VideoWriter('simulation_animations/sim_{}.mp4'.format(init_time), cv2.VideoWriter_fourcc(*'mp4v'), 5,
                          dim)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def update_figs(fig, ax0, ax1, ax2, ax3, ax4, cax5_list, ax6, ax7, env, int_env, mass_matrix, resources, oxygen, ros,
                cms_classification_list,
                pathway_activation_list, num_cell_list, cmap,
                do_animation,
                init_time,
                current_step, ducts):
    """ Update the figures  """

    int_env[ducts == 1] = -2
    cax = ax0.set_data(int_env)
    cax1 = ax1.set_data(resources)
    cax2 = ax2.set_data(oxygen)
    cax3 = ax3.set_data(mass_matrix)
    cax4 = ax4.set_data(ros)

    cms_classification_list = cms_classification_list[~np.all(cms_classification_list == 0, axis=1)]
    axes = fig.axes
    for idx, ax in enumerate(ax6):
        _ = ax.set_data(range(0, len(cms_classification_list[:, idx])), cms_classification_list[:, idx])
    axes[3].set_xlim(0, len(cms_classification_list[:, 0])-0.1)

    pathway_activation_list = pathway_activation_list[~np.all(pathway_activation_list == 0, axis=1)]
    for idx, cax5 in enumerate(cax5_list):
        _ = cax5[0].set_data(range(0, len(pathway_activation_list[:, idx])), pathway_activation_list[:, idx])
    axes[7].set_xlim(0, len(pathway_activation_list[:, 0])-0.1)


    num_cell_list = num_cell_list[~np.all(num_cell_list == 0, axis=1)]
    for idx, ax in enumerate(ax7):
        _ = ax.set_data(range(0, len(num_cell_list[:, idx])), num_cell_list[:, idx])
    axes[4].set_xlim(0, len(num_cell_list[:, 0])-0.1)
    axes[4].set_ylim(0, num_cell_list.max()+1.1)

    plt.pause(0.000000001)

    if do_animation:
        plt.savefig('temp_animation/anim_{}/frame_{}.png'.format(init_time, current_step), dpi=150)


def init_plots(env, resources, oxygen, ros, cms_classification, init_time, pathway_activation_list, pathway_list,
               num_cell_list, ducts,
               cmap=colors.ListedColormap(['grey', 'black', 'white', 'brown', 'blue', 'pink', 'green', 'red']),
               do_animation=False):
    """ Create all the handles and figures to show the matrices during the simulations """

    int_env = MAP_CLASSES_TO_IDS(env)
    int_env[ducts == 1] = -2

    fig, ((ax0, ax3, ax4, ax6, ax8), (ax1, ax2, ax5, ax7, ax9)) = plt.subplots(ncols=5, nrows=2)
    cax = ax0.matshow(int_env, cmap=cmap, vmin= -2 - 0.5, vmax=5+0.5)
    cax1 = ax1.matshow(resources, cmap='hot', vmin=0, vmax=10 ** (-16))
    cax2 = ax2.matshow(oxygen, cmap='hot', vmin=0, vmax=7.5 * 10 ** (-13))
    cax3 = ax3.matshow(GET_MASS(env), cmap='hot', vmin=0, vmax=2.1)
    cax4 = ax4.matshow(ros, cmap='hot', vmin=0, vmax=10)

    ax0.set_yticks([])
    ax0.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax3.set_yticks([])
    ax3.set_xticks([])
    ax4.set_yticks([])
    ax4.set_xticks([])

    # create colourbars the size of the plots itself
    aspect = 20
    pad_fraction = 0.5

    colorbars = []
    for axis, caxis in list(zip([ax0, ax1, ax2, ax3, ax4], [cax, cax1, cax2, cax3, cax4])):
        divider = make_axes_locatable(axis)
        width = axes_size.AxesY(axis, aspect=1. / aspect)
        pad = axes_size.Fraction(pad_fraction, width)
        clrax = divider.append_axes("right", size=width, pad=pad)
        cbar = plt.colorbar(caxis, cax=clrax)
        colorbars.append(cbar)

    # set the colour bar of the cells to have the names of the cells for clarity
    colorbars[0].ax.set_yticklabels(['vessels', 'dead', 'empty', 'tumour', 'T cell', 'B cell', 'myeloid', 'stromal'],
                             fontsize=8)

    # cbar = fig.colorbar(cax, ticks=[-1, 0, 1], orientation='horizontal')

    ax0.set_title('Cells')
    ax1.set_title('Nutrients')
    ax2.set_title('Oxygen')
    ax3.set_title('Mass')
    ax4.set_title('ROS')

    # fig.delaxes(ax7)
    fig.delaxes(ax9)

    linestyle_str = ['solid', 'dotted', 'dashed']
    distingishable_colors = ['#800000', '#F58231', '#FFE119', '#DCBEFF', '#A9A9A9']

    pathway_activation_list = pathway_activation_list[~np.all(pathway_activation_list == 0, axis=1)]

    dist_color_list = distingishable_colors * 3
    cax5_list = []
    for idx, pathway in enumerate(pathway_list):
        line_style_idx = int(np.floor(idx / len(distingishable_colors)))
        cax5_list.append(ax5.plot(pathway_activation_list[:, idx], linestyle=linestyle_str[line_style_idx],
                                  color=dist_color_list[idx], label=pathway.replace('_activation', '')))
    ax5.set_ylabel('Average expression')
    ax5.set_xlabel('Time (h)')
    ax5.set_ylim(-1.1, 1.1)
    pathway_list_formatted = ['P53', 'mTORC1', r'TNF$\alpha$viaNF$\kappa$B',
                    'Hypoxia', 'EMT', 'MYC',
                    'ROS', 'IL2_STAT5',
                    'Adipo', r'IFN$\gamma$', 'kras',
                    'IL6_JAK', 'Complement', r'IFN$\alpha$', 'PI3K']
    # ax5.legend(pathway_list_short, bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=6)

    # put the legend for the pathways in a separate plot to make it more clear
    ax7.legend([item[0] for item in cax5_list], pathway_list_formatted, loc='center left', fontsize=10)
    ax7.axis('off')

    cms_classification = cms_classification[~np.all(cms_classification == 0, axis=1)]
    ax6.set_prop_cycle(color=CMS_COLOURS)
    cax6 = ax6.plot(cms_classification)
    ax6.set_ylabel('Regression values')
    ax6.set_xlabel('Time (h)')
    ax6.set_ylim(0, 1.1)
    ax6.legend(['cms1', 'cms2', 'cms3', 'cms4'], fontsize=10, loc='upper right')
    plt.ion()

    num_cell_list = num_cell_list[~np.all(num_cell_list == 0, axis=1)]
    cax8 = ax8.plot(num_cell_list)
    ax8.set_ylabel('Cells (#)')
    ax8.set_xlabel('Time (h)')
    ax8.legend(['Cancer', 'T cell', 'B cell', 'Myeloid', 'Stromal'], fontsize=10, loc='lower right')
    plt.ion()


    # set the graphs to the max size to make them nice for the simulation video
    manager = plt.get_current_fig_manager()
    manager.resize(1920, 1080)

    fig.set_tight_layout(True)

    if do_animation:
        os.mkdir('temp_animation/anim_{}'.format(init_time))
        plt.savefig('temp_animation/anim_{}/frame_0.png'.format(init_time), dpi=300)



    return fig, cax, cax1, cax2, cax3, cax4, cax5_list, cax6, cax8
