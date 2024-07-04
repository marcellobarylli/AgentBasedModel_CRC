import numpy as np
import random
import os
from numba import jit, typeof
import bisect
from cell_definitions import cell, cancercell, myeloid, tcell, bcell, stromalcell

GET_PATHWAY_ACTIVATION_DICT = np.vectorize(lambda x: None if x == 0 else None if x.cell_state == 0 else (
    x.__dict__))

GET_DICT_ITEM = np.vectorize(lambda x, y: np.nan if x is None else float(x[y]))

# get type ids of cells for visualisation
MAP_CLASSES_TO_IDS = np.vectorize(lambda x: 0 if x == 0 else (
    -1 if x.cell_state == 0 else (
        1 if isinstance(x, cancercell) else
        (2 if isinstance(x, tcell) else
         (3 if isinstance(x, bcell) else
          (4 if isinstance(x, myeloid) else
           (5 if isinstance(x, stromalcell) else -2
            )))))))


# Main function that returns distinct count of islands in
# a given boolean 2D matrix
@jit(nopython=True)
def countDistinctIslands(grid):

    isolated_cell_count = 0
    for i in range(len(grid)):
        for j in range(len(grid)):
            # If a cell is not 1
            # no need to dfs
            if grid[i][j] != 1:
                continue
            else:
                # to hold coordinates
                # of this island
                possible_y_values = range(max(0, i - 1), min(len(grid) - 1, i + 1) + 1)
                possible_x_values = range(max(0, j - 1), min(len(grid) - 1, j + 1) + 1)

                moore_neighbors_in = [(i, j) for i in possible_y_values for j in possible_x_values]

                neighbors = 0

                for coor in moore_neighbors_in:
                    if grid[coor] == 1:
                        neighbors += 1
                    if neighbors > 1:
                        break

                print(grid[i][j], neighbors)
                if neighbors <= 1:
                    isolated_cell_count += 1

    return isolated_cell_count


def calc_metrics(env, int_env, mass_matrix, ros, pathway_list, signals, all_signal_names):
    """ This function is used to calculate all the metrics we want to track throughout the simulation """
    # mass_matrix[mass_matrix == 0] = np.nan
    pathway_activations = np.zeros((len(env), len(env)), dtype=dict)
    pathway_activations[:, :] = GET_PATHWAY_ACTIVATION_DICT(env)
    average_pathway_activation = np.zeros((len(pathway_list)))
    average_signals = np.zeros((len(all_signal_names)))

    for idx, pathway in enumerate(pathway_list):
        matrix = GET_DICT_ITEM(pathway_activations, pathway)
        average_pathway_activation[idx] = np.nanmean(matrix)

    for idx, signal in enumerate(all_signal_names):
        average_signals[idx] = signals[signal][int_env > 0].mean()

    isolated_tumour_cells = countDistinctIslands(int_env == 1)

    return np.count_nonzero(int_env == 1), np.count_nonzero(int_env == 2), np.count_nonzero(int_env == 3), \
           np.count_nonzero(int_env == 4), np.count_nonzero(int_env == 5), np.mean(mass_matrix[mass_matrix != 0]), \
           np.mean(ros), \
           average_pathway_activation, average_signals, isolated_tumour_cells


def get_num_neighbours(env, neighbourhood):
    """ this function gets the number of neighbours surrounding a cell """

    # neighbours_coors = get_moore(y, x, grid_size, randomise=False)

    # boolean_grid = (env[tuple(zip(*neighbourhood))] != 0) & (env[tuple(zip(*neighbourhood))] != -1)

    # num_neighbours = np.count_nonzero(env[tuple(zip(*neighbourhood))]) - 1
    neighbouring_env = env[tuple(zip(*neighbourhood))]
    return np.count_nonzero((neighbouring_env != 0) & (neighbouring_env != -1)) - 1


def choice(population, weights):
    x = random.random()
    weights = cdf(weights)
    idx = bisect.bisect(weights, x)
    return population[idx]


@jit(nopython=True)
def cdf(weights):
    total = np.sum(weights)
    result = []
    cumsum = 0
    for w in weights:
        cumsum += w
        result.append(cumsum / total)
    return result


def create_metric_lists(n_steps, pathway_list, signal_list):
    num_cells_list = np.zeros((n_steps + 1, 5))
    average_mass_list = np.zeros(n_steps + 1)
    average_ros_list = np.zeros(n_steps + 1)
    inflammation_list = np.zeros(n_steps + 1)
    pathway_activation_list = np.zeros((n_steps + 1, len(pathway_list)))
    cms_classification_list = np.zeros((n_steps + 1, 4))
    average_signal_list = np.zeros((n_steps + 1, len(signal_list)))
    n_isolated_tumour_list = np.zeros(n_steps + 1)

    reasons_cell_death = \
        {
            'cancercell': {'stress': 0, 'mitosis_supp': 0, 'old_age': 0, 'nut_ox_lack': 0, 'DNA_damage': 0,
            'mitosis': 0,
            'spawned': 0},
            'tcell': {'stress': 0, 'mitosis_supp': 0, 'old_age': 0, 'nut_ox_lack': 0, 'DNA_damage': 0, 'mitosis': 0,
                      'spawned': 0},
            'bcell': {'stress': 0, 'mitosis_supp': 0, 'old_age': 0, 'nut_ox_lack': 0, 'DNA_damage': 0, 'mitosis': 0,
                      'spawned': 0},
            'myeloid': {'stress': 0, 'mitosis_supp': 0, 'old_age': 0, 'nut_ox_lack': 0, 'DNA_damage': 0, 'mitosis':
                0, 'spawned': 0},
            'stromalcell': {'stress': 0, 'mitosis_supp': 0, 'old_age': 0, 'nut_ox_lack': 0, 'DNA_damage': 0,
                            'mitosis': 0, 'spawned': 0}
        }

    return num_cells_list, average_mass_list, average_ros_list, inflammation_list, pathway_activation_list, \
           cms_classification_list, average_signal_list, n_isolated_tumour_list, reasons_cell_death


def save_metrics(sim_step, env, int_env, mass_matrix, ros, pathway_list, num_cells_list, average_mass_list,
                 average_ros_list,
                 inflammation_list,
                 system_inflammation, pathway_activation_list, cms_classification_list, average_signal_list,
                 n_isolated_tumour_list, all_signal_names, signals, classifier_obj):
    num_cancer_cell, num_t_cell, num_b_cell, num_myeloid, num_stromal_cell, average_mass, average_ros, \
    average_pathway_activation, average_signals, isolated_tumour_cells\
        = calc_metrics(env, int_env, mass_matrix, ros, pathway_list, signals, all_signal_names)
    num_cells_list[sim_step, 0] = num_cancer_cell
    num_cells_list[sim_step, 1] = num_t_cell
    num_cells_list[sim_step, 2] = num_b_cell
    num_cells_list[sim_step, 3] = num_myeloid
    num_cells_list[sim_step, 4] = num_stromal_cell
    average_mass_list[sim_step] = average_mass
    average_ros_list[sim_step] = average_ros
    inflammation_list[sim_step] = system_inflammation
    pathway_activation_list[sim_step, :] = average_pathway_activation
    cms_classification_list[sim_step, :] = classifier_obj.classify_sample(
        dict(zip(pathway_list, average_pathway_activation)))
    average_signal_list[sim_step, :] = average_signals
    n_isolated_tumour_list[sim_step] = isolated_tumour_cells
    return num_cells_list, average_mass_list, average_ros_list, inflammation_list, pathway_activation_list, \
           cms_classification_list, average_signal_list, n_isolated_tumour_list


def trim_lists(average_mass_list, average_ros_list, inflammation_list, num_cells_list, pathway_activation_list,
               cms_classification_list):
    """ This function trims the lists in case the simulation is over quicker then expected """
    average_mass_list = np.trim_zeros(average_mass_list, 'b')
    average_ros_list = np.trim_zeros(average_ros_list, 'b')
    inflammation_list = np.trim_zeros(inflammation_list, 'b')
    num_cells_list = num_cells_list[~np.all(num_cells_list == 0, axis=1)]
    pathway_activation_list = pathway_activation_list[~np.all(pathway_activation_list == 0, axis=1)]
    cms_classification_list = cms_classification_list[~np.all(cms_classification_list == 0, axis=1)]

    return average_mass_list, average_ros_list, inflammation_list, num_cells_list, pathway_activation_list, \
           cms_classification_list


def save_raw_data(init_time, num_cells_list, average_mass_list, inflammation_list, pathway_activation_list,
                  cms_classification_list):
    """ This function saves the raw data from the simulation """

    os.mkdir('sim_results/sim_{}'.format(init_time))
    os.mkdir('sim_results/sim_{}/raw_data'.format(init_time))

    np.savetxt('sim_results/sim_{}/raw_data/numcellsovertime.txt'.format(init_time), num_cells_list)
    np.savetxt('sim_results/sim_{}/raw_data/massovertime.txt'.format(init_time), average_mass_list)
    np.savetxt('sim_results/sim_{}/raw_data/inflammationovertime.txt'.format(init_time), inflammation_list)
    np.savetxt('sim_results/sim_{}/raw_data/pathwaysovertime.txt'.format(init_time), pathway_activation_list)
    np.savetxt('sim_results/sim_{}/raw_data/classificationovertime.txt'.format(init_time), cms_classification_list)


def get_moore(y, x, grid_size, radius=1, randomise=True, on_grid=True):
    """ Get the coordinates of moore neighborhood """

    if on_grid:
        possible_x_values = range(max(0, x - radius), min(grid_size - 1, x + radius) + 1)
        possible_y_values = range(max(0, y - radius), min(grid_size - 1, y + radius) + 1)

        moore_neighbors_in = [(i, j) for i in possible_y_values for j in possible_x_values]
        return moore_neighbors_in

    else:

        possible_x_values_in = range(max(0, x - radius), min(grid_size - 1, x + radius) + 1)
        possible_y_values_in = range(max(0, y - radius), min(grid_size - 1, y + radius) + 1)
        moore_neighbors_in = [(i, j) for i in possible_y_values_in for j in possible_x_values_in]

        possible_x_values = range((x - radius), (x + radius) + 1)
        possible_y_values = range((y - radius), (y + radius) + 1)
        moore_neighbors = [(i, j) for i in possible_y_values for j in possible_x_values]

        return moore_neighbors_in, moore_neighbors

    # neumann_neighbors = [(min(y + 1, grid_size - 1), x),
    #              (y, min(x + 1, grid_size - 1)),
    #              (max(y - 1, 0), x),
    #              (y, max(x - 1, 0))]

