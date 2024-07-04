"""
This file contains the functions for initiation of the simulation
Author: Robin van den Berg
Contact: rvdb7345@gmail.com
"""

import numpy as np
import pickle
from cell_definitions import cell, cancercell, myeloid, tcell, bcell, stromalcell
from load_initial_cell_setting import get_initial_coordinates, get_initial_coordinates_full_tumour
from load_resources_supply import get_resources_supplies, get_ducts, get_ducts_epith_guided
from utils import get_num_neighbours, get_moore
from numba import jit
from skimage.transform import resize
from PIL import Image
import matplotlib.pyplot as plt
import random
import time

np.random.seed(seed=int((time.time() % 1) * 10000000))


# @jit(nopython=True)
# def get_moore(y, x, grid_size, radius=1, randomise=True):
#     """ Get the coordinates of moore neighborhood """
#
#     possible_x_values = range(max(0, x - radius), min(grid_size - 1, x + radius) + 1)
#     possible_y_values = range(max(0, y - radius), min(grid_size - 1, y + radius) + 1)
#
#     moore_neighbors = [(i, j) for i in possible_y_values for j in possible_x_values]
#
#     # neumann_neighbors = [(min(y + 1, grid_size - 1), x),
#     #              (y, min(x + 1, grid_size - 1)),
#     #              (max(y - 1, 0), x),
#     #              (y, max(x - 1, 0))]
#
#     return moore_neighbors

def init_env(grid_size):
    """ Get matrix of zeros with correct dtype """
    env = np.zeros((grid_size, grid_size), dtype=cell)

    return env


def init_cells_random(env, n_cells, init_cms, fraction_types, params_tumour, params_tcell, params_bcell,
                      params_myeloid, params_stromal, cell_idx):
    """ Place cells randomly across the grid according to fractions """

    ''' Randomly initiate cells at random locations '''
    coors = random.sample(range(0, len(env[0]) ** 2), k=n_cells)

    # create the right proportions of all the cells we will be creating
    types = np.random.choice([0, 1, 2, 3, 4],
                             n_cells,
                             p=fraction_types)

    for idx, coor in enumerate(coors):
        if types[idx] == 0:
            env.flat[coor] = cancercell(**params_tumour)
        elif types[idx] == 1:
            env.flat[coor] = myeloid(**params_myeloid)
        elif types[idx] == 2:
            env.flat[coor] = tcell(**params_tcell)
        elif types[idx] == 3:
            env.flat[coor] = bcell(**params_bcell)
        elif types[idx] == 4:
            env.flat[coor] = stromalcell(**params_stromal)

    env = env.reshape(len(env[0]), len(env[0]))

    # assign the coordinates, ids and randomised mass to each cell
    for x in range(len(env)):
        for y in range(len(env)):
            if env[y][x] != 0:
                env[y][x].crs = (y, x)
                env[y][x].id = cell_idx

                # in the initial setting, cells should not all be initiated with the same masses
                env[y][x].mass = np.random.random() + 1
                env[y][x].time_lived = int(np.random.random() * (1 / env[y][x].apoptosis_rate))
                cell_idx += 1

                env[y][x].initialise_pathways(init_cms)
                env[y][x].pathway_response()

    return env, cell_idx


def init_cells_config(env, n_cells, init_cms, fraction_types, params_tumour, params_tcell, params_bcell,
                      params_myeloid, params_stromal, cell_idx, resources,
                      tumour_location_path, y_range, x_range):
    """ Place cells randomly across the grid according to fractions """

    ''' initiate cells at locations specified by image data'''

    if tumour_location_path == 'starting_configurations/epithelial_cells.jpg':
        coors = get_initial_coordinates(tumour_location_path, (60, 160), (0, 100))
    elif tumour_location_path == 'starting_configurations/tumour_cms4-2.jpg':
        coors = get_initial_coordinates_full_tumour(tumour_location_path, y_range, x_range)



    # create the right proportions of all the cells we will be creating
    if n_cells > len(coors):
        print('number of cells larger then available coordinates, adjusting to max')
        coors_tumour = coors
    else:
        coors_tumour = random.sample(coors, n_cells)

    for idx, coor in enumerate(coors_tumour):
        env[coor] = cancercell(**params_tumour)

    additional_cell_number = 2000

    available_coors_sep = np.where(env == 0)
    available_coors = list(zip(available_coors_sep[0], available_coors_sep[1]))
    random.shuffle(available_coors)
    available_coors = random.sample(available_coors, additional_cell_number)

    types = np.random.choice([1, 2, 3, 4],
                             additional_cell_number,
                             p=fraction_types[1:] / sum(fraction_types[1:]))

    for idx, coor in enumerate(available_coors):
        if types[idx] == 1:
            env[coor] = myeloid(**params_myeloid)
        elif types[idx] == 2:
            env[coor] = tcell(**params_tcell)
        elif types[idx] == 3:
            env[coor] = bcell(**params_bcell)
        elif types[idx] == 4:
            env[coor] = stromalcell(**params_stromal)

    # assign the coordinates, ids and randomised mass to each cell
    inflammations_list = []

    for x in range(len(env)):
        for y in range(len(env)):
            if env[y][x] != 0:
                env[y][x].crs = (y, x)
                env[y][x].id = cell_idx

                # in the initial setting, cells should not all be initiated with the same masses
                fraction_of_life_passed = np.random.random()
                env[y][x].mass = np.random.random() + 1
                env[y][x].time_lived = int(fraction_of_life_passed * (1 / env[y][x].apoptosis_rate)) * 0.1
                cell_idx += 1

                env[y][x].initialise_pathways(init_cms)
                system_inflammation_ind = env[y][x].pathway_response()
                inflammations_list.append(system_inflammation_ind)

                moore_neigbourhood_in = get_moore(y, x, len(env[0, :]), radius=1, randomise=False,
                                                  on_grid=True)
                num_neighbours = get_num_neighbours(env, moore_neigbourhood_in)
                env[y][x].DNA_damage = max(0, (num_neighbours - 5) * 0.2)
                env[y][x].ros_production_modulation(len(env[0, :]), env)

    system_inflammation = np.mean(inflammations_list)

    for x in range(len(env)):
        for y in range(len(env)):
            if env[y][x] != 0:
                env[y][x].update_signal_production(len(env[0, :]), env, system_inflammation, resources)

    return env, cell_idx, system_inflammation


def init_cells_prolif(env, proliferation_cell, init_cms, fraction_types, params_tumour, params_tcell,
                      params_bcell,
                      params_myeloid, params_stromal, cell_idx):
    """ Place one cell in the middle of the grid to see how the cell proliferates """

    middle_coor = (int(len(env[:, 0]) / 2), int(len(env[0, :]) / 2))

    coors = get_moore(middle_coor[0], middle_coor[1], len(env[:, 0]), radius=30)

    available_coors = random.sample(coors, 60)

    for coor in available_coors:
        if proliferation_cell == 'tumour':
            env[coor] = cancercell(**params_tumour)
        elif proliferation_cell == 'myeloid':
            env[coor] = myeloid(**params_myeloid)
        elif proliferation_cell == 'tcell':
            env[coor] = tcell(**params_tcell)
        elif proliferation_cell == 'bcell':
            env[coor] = bcell(**params_bcell)
        elif proliferation_cell == 'stromal':
            env[coor] = stromalcell(**params_stromal)

        # assign the coordinates, ids and randomised mass to each cell
        env[coor].crs = (coor[0], coor[1])
        env[coor].id = cell_idx

        # in the initial setting, cells should not all be initiated with the same masses
        env[coor].mass = 1 + random.random()
        env[coor].time_lived = int(np.random.random() * (1 / env[coor[0]][coor[1]].apoptosis_rate)) * 0.1
        cell_idx += 1

        env[coor].initialise_pathways(init_cms)
        env[coor].pathway_response()

    return env, cell_idx


def init_cells(env, n_cells, init_cms, fraction_types, params_tumour, params_tcell, params_bcell,
               params_myeloid, params_stromal, cell_idx, resources,tumour_location_path, y_range, x_range,
               type='random',
               proliferation_plot=False,
               proliferation_cell=None):
    system_inflammation = 1
    if not proliferation_plot:
        if type == 'random':
            env, cell_idx = init_cells_random(env, n_cells, init_cms, fraction_types, params_tumour, params_tcell,
                                              params_bcell,
                                              params_myeloid, params_stromal, cell_idx)
        elif type == 'config':
            env, cell_idx, system_inflammation = init_cells_config(env, n_cells, init_cms, fraction_types,
                                                                   params_tumour,
                                                                   params_tcell,
                                                                   params_bcell,
                                                                   params_myeloid, params_stromal, cell_idx, resources,
                                                                   tumour_location_path,
                                                                   y_range, x_range)
    else:
        env, cell_idx = init_cells_prolif(env, proliferation_cell, init_cms, fraction_types, params_tumour,
                                          params_tcell,
                                          params_bcell,
                                          params_myeloid, params_stromal, cell_idx)

    return env, cell_idx, system_inflammation


def init_resources_random(grid_size, init_value):
    return np.random.random((grid_size, grid_size)) * init_value


def init_resources_prolif(grid_size, medium_deposit):
    return np.ones((grid_size, grid_size)) * medium_deposit


def init_oxygen_prolif(grid_size):
    return np.ones((grid_size, grid_size)) * 10000000


def init_resources_config(init_cms):
    return np.loadtxt('starting_configurations/resources_start_{}.txt'.format(init_cms))


def init_oxygen_config(init_cms):
    return np.loadtxt('starting_configurations/oxygen_start_{}.txt'.format(init_cms))


def init_resources(grid_size, init_cms, init_value, type, get_diffused_grid, get_signalling_strengths,
                   proliferation_plot,
                   medium_deposit):
    """ Create a uniform distribution of resources """
    if (type == 'random' or get_diffused_grid) and not get_signalling_strengths:
        resources = init_resources_random(grid_size, init_value)
    elif type == 'config' or get_signalling_strengths:
        resources = init_resources_config(init_cms)
    if proliferation_plot:
        resources = init_resources_prolif(grid_size, medium_deposit)
    return resources


def init_oxygen(grid_size, init_cms, init_value, type, get_diffused_grid, get_signalling_strengths, proliferation_plot):
    """ Create a uniform distribution of resources """
    if (type == 'random' or get_diffused_grid) and not get_signalling_strengths:
        oxygen = init_oxygen_random(grid_size, init_value)
    elif type == 'config' or get_signalling_strengths:
        oxygen = init_oxygen_config(init_cms)
    if proliferation_plot:
        oxygen = init_oxygen_prolif(grid_size)

    return oxygen


def init_ros(grid_size, init_value):
    """ Create a uniform distribution of resources """
    return np.ones((grid_size, grid_size)) * init_value


def init_oxygen_random(grid_size, init_value):
    """ Create a uniform distribution of oxygen """
    return np.random.random((grid_size, grid_size)) * init_value


def initiate_simulation_grids(grid_size, resources_init_value, oxygen_init_value, ros_init_value, signals_init_values,
                              n_cells, init_cms,
                              fraction_types, params_tumour, params_tcell, params_bcell, params_myeloid,
                              params_stromal, all_signal_names, cell_idx, get_diffused_grid=False,
                              get_signalling_strengths=False, type='config',
                              proliferation_plot=False, proliferation_cell=None, medium_deposit=None,
                              tumour_image_path='starting_configurations/epithelial_cells.jpg', y_range=(60, 160),
                              x_range=(0, 100)):
    """
    This function initiates all the grids necessary for the simulation
    """

    env = init_env(grid_size)
    resources = init_resources(grid_size, init_cms, resources_init_value, type, get_diffused_grid,
                               get_signalling_strengths,
                               proliferation_plot,
                               medium_deposit)
    oxygen = init_oxygen(grid_size, init_cms, oxygen_init_value, type, get_diffused_grid, get_signalling_strengths,
                         proliferation_plot)
    env, cell_idx, system_inflammation = init_cells(env, n_cells, init_cms, fraction_types, params_tumour,
                                                    params_tcell, params_bcell,
                                                    params_myeloid, params_stromal, cell_idx, resources,
                                                    tumour_image_path, y_range, x_range, type,
                                                    proliferation_plot,
                                                    proliferation_cell)
    # p=1-0.0010

    duct_path = 'starting_configurations/duct_depository/ducts_{}_{}_{}.txt'\
        .format(tumour_image_path.replace('starting_configurations/', ''), y_range, x_range)
    if tumour_image_path == 'starting_configurations/epithelial_cells.jpg':
        down_size = True
    elif tumour_image_path == 'starting_configurations/tumour_cms4-2.jpg':
        down_size = False

    if get_diffused_grid and not get_signalling_strengths:
        # get constant starting signals
        signals = {}
        for signal in all_signal_names:
            signals[signal] = np.ones((grid_size, grid_size)) * signals_init_values[signal]

        # load the ducts from the image
        ducts = get_ducts_epith_guided(tumour_image_path, y_range, x_range,
                                       threshold=0.262, down_size=down_size)
        np.savetxt(duct_path, ducts)

        ros = init_ros(grid_size, ros_init_value)
    elif get_diffused_grid and get_signalling_strengths:
        signals = {}
        for signal in all_signal_names:
            signals[signal] = np.ones((grid_size, grid_size)) * signals_init_values[signal]

        # load the ducts if they are already available for this image, otherwise create them
        try:
            ducts = np.loadtxt(duct_path)
        except :
            # load the ducts from the image
            ducts = get_ducts_epith_guided(tumour_image_path, y_range, x_range,
                                           threshold=0.262, down_size=down_size)
            np.savetxt(duct_path, ducts)
        ros = np.loadtxt('starting_configurations/ros_start_{}.txt'.format(init_cms))
    else:
        with open('starting_configurations/prediffused_signals_{}.pkl'.format(init_cms), "rb") as fp:
            signals = pickle.load(fp)

        # load the ducts if they are already available for this image, otherwise create them
        try:
            ducts = np.loadtxt(duct_path)
        except :
            # load the ducts from the image
            ducts = get_ducts_epith_guided(tumour_image_path, y_range, x_range,
                                           threshold=0.262, down_size=down_size)
            np.savetxt(duct_path, ducts)

        ros = np.loadtxt('starting_configurations/ros_start_{}.txt'.format(init_cms))

    return env, resources, oxygen, ros, signals, ducts, cell_idx, system_inflammation
