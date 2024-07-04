"""
This file contains framework of the project modelling colorectal cancer as an agent-based model
Author: Robin van den Berg
Contact: rvdb7345@gmail.com
"""

import tqdm
import time
import random
import matplotlib.pyplot as plt
import sys
import pickle
import numpy as np
import matplotlib
from matplotlib import colors
from utils import calc_metrics, get_num_neighbours, choice, cdf, create_metric_lists, save_metrics, trim_lists, \
    save_raw_data, get_moore
from load_parameters import load_cell_parameters, load_cell_fractions, load_biological_parameters, \
    load_initial_grid_parameters, load_inherent_signal_production, load_resource_influx, \
    load_pathway_activation_thresholds, load_base_signal_production
from simulation_initiation import initiate_simulation_grids
from pathway_cms_classification import sample_classifier
from diffusion_scripts.diffusion_glutamine import diffusion_nutrients
from diffusion_scripts.diffusion_oxygen import diffusion_oxygen
from diffusion_scripts.diffusion_ROS import diffusion_ROS
from diffusion_scripts.diffusion_signal import diffusion_signal
from plotting import create_final_plots, create_animation, update_figs, init_plots
from cell_definitions import cell, cancercell, myeloid, tcell, bcell, stromalcell

matplotlib.use('TkAgg')
np.set_printoptions(threshold=sys.maxsize)

# get nutrient consumption of the different cells
TRANSLATE_IDX_NUTRIENT = lambda x: float(x.nutrient_consumption) if not x == 0 or x == -1 else 0
TRANSLATE_IDX_OXYGEN = lambda x: 0 if x == 0 or x == -1 else float(x.oxygen_consumption)
TRANSLATE_IDX_PROLIFERATION = lambda x: 0 if x == 0 or x == -1 else float(x.proliferation_rate)
TRANSLATE_IDX_TIMEDEATH = lambda x: -1 if x == 0 else float(x.time_dead)
TRANSLATE_IDX_PROLIFERATION_subtype = lambda x: 0 if x == 0 or x == -1 else float(x.proliferation_rate)
TRANSLATE_IDX_ROS_PRODUCTION = lambda x: 0 if x == 0 or x == -1 else float(x.ROS_production)

# define vectorised function to consume resources
GET_METABOLIC_MAINTENANCE = np.vectorize(TRANSLATE_IDX_NUTRIENT, otypes=[np.float])
GET_OXYGEN_CONSUMPTION = np.vectorize(TRANSLATE_IDX_OXYGEN, otypes=[np.float])
GET_PROLIFERATION_RATE = np.vectorize(TRANSLATE_IDX_PROLIFERATION, otypes=[np.float])
GET_TIME_DEATH = np.vectorize(TRANSLATE_IDX_TIMEDEATH, otypes=[int])
GET_ROS_PRODUCTION = np.vectorize(TRANSLATE_IDX_ROS_PRODUCTION, otypes=[np.float])

# get type ids of cells
MAP_CLASSES_TO_IDS = np.vectorize(lambda x: 0 if x == 0 else (
    -1 if x.cell_state == 0 else (
        1 if isinstance(x, cancercell) else
        (2 if isinstance(x, tcell) else
         (3 if isinstance(x, bcell) else
          (4 if isinstance(x, myeloid) else
           (5 if isinstance(x, stromalcell) else -2
            )))))), otypes=[int])

GET_PATHWAY_ACTIVATION = np.vectorize(lambda x: np.zeros(20) if x == 0 else (np.zeros(20) if x.cell_state == 0 else ([
    x.oxidative_phosphorylation_activation, x.glycolysis_activation,
    x.P53_activation, x.mTORC1_activation, x.TNFaviaNFkB_activation,
    x.unfolded_protein_response_activation, x.hypoxia_activation,
    x.EMT_activation, x.myogenesis_activation, x.MYC_target_v1_activation,
    x.ROS_pathway_activation, x.IL2_STAT5_signalling_activation,
    x.peroxisome_activation, x.adipogenesis_activation,
    x.IFN_gamma_activation, x.kras_signalling_activation,
    x.IL6_JAK_activation, x.complement_activation,
    x.interferon_a_pathway_activation, x.PI3K_activation])), otypes=[np.float])

GET_SIGNAL_RELEASE = np.vectorize(lambda x, signal: 0 if isinstance(x, int) else x.get_signal(signal),
                                  otypes=[np.float])

GET_PATHWAY_ACTIVATION_DICT = np.vectorize(lambda x: None if x == 0 else None if x.cell_state == 0 else (
    x.__dict__), otypes=[dict])

GET_DICT_CELL_TYPE_SPEC = np.vectorize(lambda x, y: None if x == 0 else None if x.cell_state == 0 else None if
x.cell_type != y else (
    x.__dict__), otypes=[dict])

GET_DICT_ITEM = np.vectorize(lambda x, y: np.nan if x is None else float(x[y]), otypes=[np.float])

# get the mass of all cells on grid
GET_MASS = np.vectorize(lambda x: 0. if x == 0. else float(x.mass), otypes=[np.float])


def do_mitosis(env, y, x, grid_size, ducts, cell_idx,
               params_tumour, params_tcell, params_bcell,
               params_myeloid, params_stromal):
    """ Make a cell do mitosis and place it around the mother cell """

    _, possible_coor = get_moore(y, x, grid_size, radius=2, on_grid=False)
    random.shuffle(possible_coor)
    found = False
    idx = 0
    while not found and not idx == len(possible_coor):
        # cells daughter cells can be placed outside the grid
        if possible_coor[idx][0] < 0 or possible_coor[idx][0] >= grid_size or possible_coor[idx][1] < 0 or \
                possible_coor[idx][1] >= grid_size:
            env[y][x].mass = 1
            env[y][x].time_lived = 0
            found = True

        elif env[possible_coor[idx][0]][possible_coor[idx][1]] == 0:
            class_id = MAP_CLASSES_TO_IDS(env[y][x])

            # choose which cell to
            if class_id == 1:
                env[possible_coor[idx][0]][possible_coor[idx][1]] = cancercell(**params_tumour)
            elif class_id == 2:
                env[possible_coor[idx][0]][possible_coor[idx][1]] = tcell(**params_tcell)
            elif class_id == 3:
                env[possible_coor[idx][0]][possible_coor[idx][1]] = bcell(**params_bcell)
            elif class_id == 4:
                env[possible_coor[idx][0]][possible_coor[idx][1]] = myeloid(**params_myeloid)
            elif class_id == 5:
                env[possible_coor[idx][0]][possible_coor[idx][1]] = stromalcell(**params_stromal)

            found = True

            # set sell id for newly created cell
            env[possible_coor[idx][0]][possible_coor[idx][1]].crs = \
                (possible_coor[idx][0], possible_coor[idx][1])
            env[possible_coor[idx][0]][possible_coor[idx][1]].id = cell_idx
            cell_idx += 1
            env[possible_coor[idx][0]][possible_coor[idx][1]].copy_pathways(env[y][x])
            env[possible_coor[idx][0]][possible_coor[idx][1]].ROS_production = env[y][x].ROS_production
            # reset mass of mother cell to initial state
            env[y][x].mass = 1
            env[y][x].time_lived = 0

            if env[y][x].cell_type == 'tcell':
                env[possible_coor[idx][0]][possible_coor[idx][1]].activated = env[y][x].activated

        else:
            idx += 1

    return env, found, cell_idx


def cell_migrate(y, x, grid_size, resources, oxygen, env, ducts, ROS):
    """ This function migrates a cell with a higher chance of migrating towards an area with more preferable
    circumstances """

    moore_neigbourhood_in, moore_neigbourhood = get_moore(y, x, grid_size, radius=1, randomise=False, on_grid=False)
    num_neighbours = get_num_neighbours(env, moore_neigbourhood_in)

    selec_resources = resources[tuple(zip(*moore_neigbourhood_in))]
    selec_oxygen = oxygen[tuple(zip(*moore_neigbourhood_in))]

    average_resources = np.mean(selec_resources)
    average_oxygen = np.mean(selec_oxygen)

    selec_resources += (len(moore_neigbourhood) - len(moore_neigbourhood_in)) * average_resources
    selec_oxygen += (len(moore_neigbourhood) - len(moore_neigbourhood_in)) * average_oxygen

    # calculate the relative circumstances of each grid space
    percentage_diff_resources = 1 + (selec_resources - average_resources) / \
                                average_resources
    percentage_diff_oxygen = 1 + (selec_oxygen - average_oxygen) / \
                             average_oxygen

    bias_list = (percentage_diff_oxygen + percentage_diff_resources) / 2

    chosen_coordinates = choice(moore_neigbourhood, bias_list / len(bias_list))

    # allow cells to migrate out of the grid then removing them from the system
    if chosen_coordinates[0] > grid_size - 1 or chosen_coordinates[0] < 0 or chosen_coordinates[1] > grid_size - 1 or \
            chosen_coordinates[1] < 0:
        env[y][x] = 0

    # relocate the cell if the grid space is empty, the adhesiveness doesn't prevent it and if it isn't in a duct.
    elif env[chosen_coordinates[0]][chosen_coordinates[1]] == 0 and \
            random.random() <= (1 - 2 * num_neighbours / 2 * env[y][x].adhesiveness):
        env[chosen_coordinates[0]][chosen_coordinates[1]] = env[y][x]
        env[chosen_coordinates[0]][chosen_coordinates[1]].migrate(chosen_coordinates[0], chosen_coordinates[1])
        env[y][x] = 0

        temp = ROS[chosen_coordinates[0]][chosen_coordinates[1]]
        ROS[chosen_coordinates[0]][chosen_coordinates[1]] = ROS[y][x]
        ROS[y][x] = temp

    return env


def diffusion_wrapper(grid_size, current_step, resources, mass_cell, mass_matrix, proliferation_rates,
                      metabolic_maintenance_matrix,
                      nutrient_diffusion_constant,
                      S_nutrient, ducts, oxygen, oxygen_consumption_matrix, oxygen_diffusion_constant, ros,
                      cell_locations, ros_diffusion_constant, signals, all_signal_names, env,
                      signal_diffusion_constant, influxes, proliferation_plot, replenish_time, ox_modulation_value):
    """" This function wraps all the diffusion functions to keep the main function clean """

    print('start diffusion')

    # if we are not trying to make a proliferation plot, oxygen is diffused and nutrients come from a duct
    if not proliferation_plot:
        resources_supply = ducts
        oxygen = diffusion_oxygen(grid_size, grid_size, oxygen, oxygen_consumption_matrix,
                                  diffusion_constant=oxygen_diffusion_constant, resources_supply=resources_supply,
                                  influx_value=influxes['oxygen'])

        average_signal_productions = {}
        for signal in list(set(all_signal_names) - set(['insulin', 'Shh', 'metabolistic_signalling',
                                                        'IFNalpha'])):
            signal_productions = GET_SIGNAL_RELEASE(env, signal + '_production')
            signals[signal], _ = diffusion_signal(grid_size, grid_size, signals[signal], cell_locations,
                                                  signal_productions,
                                                  diffusion_constant=signal_diffusion_constant, signal_name=signal)
            average_signal_productions[signal] = signal_productions[signal_productions > 0].mean()

        print('The average signalling productions: ', average_signal_productions)

        ros = diffusion_ROS(grid_size, grid_size, ros, cell_locations, ox_modulation_value, GET_ROS_PRODUCTION(env),
                            diffusion_constant=ros_diffusion_constant)

    else:
        # just like in experiments, the medium is replaced every x number 
        if current_step % replenish_time == 0:
            resources_supply = np.ones((grid_size, grid_size))
            resources = np.zeros((grid_size, grid_size))
        else:
            resources_supply = np.zeros((grid_size, grid_size))

    resources, _ = diffusion_nutrients(grid_size, grid_size, mass_cell, resources, mass_matrix, proliferation_rates,
                                       metabolic_maintenance_matrix,
                                       diffusion_constant=nutrient_diffusion_constant, S=S_nutrient,
                                       resources_supply=resources_supply,
                                       influx_value=influxes['nutrient'])

    print('diffusion ended')
    return resources, oxygen, ros, signals


def spawn_immune_cells(cell_idx, env, ducts, n_cells_to_place, params_tcell, init_cms):
    """ Place a T cell randomly on the grid """

    coordinates = list(zip(*np.nonzero(env == 0)))
    # available_spots = [coor for coor in coordinates if coor not in ducts]
    random.shuffle(coordinates)

    for coor_idx in range(int(n_cells_to_place)):
        env[coordinates[coor_idx][0]][coordinates[coor_idx][1]] = tcell(**params_tcell)
        env[coordinates[coor_idx][0]][coordinates[coor_idx][1]].initialise_pathways(init_cms)
        env[coordinates[coor_idx][0]][coordinates[coor_idx][1]].crs = (
            coordinates[coor_idx][0], coordinates[coor_idx][1])
        env[coordinates[coor_idx][0]][coordinates[coor_idx][1]].id = cell_idx
        cell_idx += 1

    return env, cell_idx


def run_sim(n_steps, grid_size, n_cells, init_cms, fraction_types, params_tumour, params_tcell, params_bcell,
            params_myeloid, params_stromal, resources_init_value, oxygen_init_value, ros_init_value,
            signals_init_value, influxes, plot=True, get_signalling_strengths=False,
            do_animation=False, save_data=True, get_diffused_grids=False, proliferation_plot=False,
            proliferation_cell=None, replenish_time=24, prediffision_step=1, return_data=False,
            fixed_inflammation=False, ox_modulation_value=1.5,
            tumour_image_path='starting_configurations/epithelial_cells.jpg', tumour_image_y_range=(60, 160),
            tumour_image_x_range=(0, 100)):
    init_time = str(time.time())[0:9]

    matplotlib.use('TkAgg')

    """ Main function that walks through the iterations """
    cell_idx = 0

    # create the object that can classify a sample based on the pathway activation
    classifier_obj = sample_classifier('test', 'test')

    pathway_list = ['P53_activation', 'mTORC1_activation', 'TNFaviaNFkB_activation',
                    'hypoxia_activation', 'EMT_activation', 'MYC_target_v1_activation',
                    'ROS_pathway_activation', 'IL2_STAT5_signalling_activation',
                    'adipogenesis_activation', 'IFN_gamma_activation', 'kras_signalling_activation',
                    'IL6_JAK_activation', 'complement_activation', 'interferon_a_pathway_activation', 'PI3K_activation']

    # create an environment, cells and the signal grids
    all_signal_names = ['NFkB', 'insulin', 'EGF', 'WNT', 'EMT_signalling', 'STAT3', 'IL2', 'TNFalpha', 'IFNgamma',
                        'IL6', 'MYC', 'Shh', 'metabolistic_signalling', 'IFNalpha', 'OPN']

    env, resources, oxygen, ros, signals, ducts, cell_idx, system_inflammation = \
        initiate_simulation_grids(grid_size,
                                  resources_init_value,
                                  oxygen_init_value, ros_init_value,
                                  signals_init_value,
                                  n_cells,
                                  init_cms,
                                  fraction_types, params_tumour,
                                  params_tcell,
                                  params_bcell, params_myeloid,
                                  params_stromal, all_signal_names,
                                  cell_idx,
                                  get_diffused_grids,
                                  get_signalling_strengths,
                                  type='config',
                                  proliferation_plot=proliferation_plot,
                                  proliferation_cell=proliferation_cell,
                                  medium_deposit=influxes['nutrient'],
                                  tumour_image_path=tumour_image_path,
                                  y_range=tumour_image_y_range,
                                  x_range=tumour_image_x_range)

    dicts_of_cancercell = GET_DICT_CELL_TYPE_SPEC(env, 'cancercell')
    cancer_cell_prolif_rate = GET_DICT_ITEM(dicts_of_cancercell, 'proliferation_rate')
    # print('the average cancer cell proliferation rate:', np.nanmean(cancer_cell_prolif_rate))

    # create lists to save metrics in
    num_cells_list, average_mass_list, average_ros_list, inflammation_list, pathway_activation_list, \
    cms_classification_list, average_signal_list, n_isolated_tumour_list, reasons_cell_death = create_metric_lists(
        n_steps, pathway_list, all_signal_names)

    # simulation parameters
    system_inflammation = 0.0
    current_step = 0

    # save starting metrics of the simulation
    int_env = MAP_CLASSES_TO_IDS(env)
    mass_matrix = GET_MASS(env)

    num_cells_list, average_mass_list, average_ros_list, inflammation_list, pathway_activation_list, \
    cms_classification_list, average_signal_list, n_isolated_tumour_list = save_metrics(current_step, env, int_env,
                                                                                        mass_matrix, ros,
                                                                                        pathway_list,
                                                                                        num_cells_list,
                                                                                        average_mass_list,
                                                                                        average_ros_list,
                                                                                        inflammation_list,
                                                                                        system_inflammation,
                                                                                        pathway_activation_list,
                                                                                        cms_classification_list,
                                                                                        average_signal_list,
                                                                                        n_isolated_tumour_list,
                                                                                        all_signal_names, signals,
                                                                                        classifier_obj)
    pathway_activation_list_trunc = pathway_activation_list[~np.all(pathway_activation_list == 0, axis=1)]
    # print(dict(zip(pathway_list, pathway_activation_list_trunc[-1, :])))

    # display the initial situation in a figure
    if plot:
        cmap = colors.ListedColormap(['grey', 'black', 'white', 'brown', 'blue', 'yellow', 'green', 'red'])
        fig, ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7 = init_plots(env, resources, oxygen, ros, cms_classification_list,
                                                                 init_time, pathway_activation_list, pathway_list,
                                                                 num_cells_list, ducts,
                                                                 cmap=cmap, do_animation=do_animation)

    # biological parameters
    S_nutrient, stress_buildup, cell_stress_threshold, time_dormant_threshold, mass_mitosis_threshold, \
    mitosis_suppression_threshold, dead_cell_removal_time, nutrient_diffusion_constant, oxygen_diffusion_constant, \
    ros_diffusion_constant, signal_diffusion_constant, mass_cell, migration_fraction, DNA_damage_buildup, \
    DNA_damage_threshold, metabolic_maintenance \
        = load_biological_parameters()

    Vmax = 6.25 * 10 ** (-17) * 3600
    km = 1.33 * 10 ** (-6) * 3375 * 10 ** -15 * 3600
    ox_threshold = 0.15 * 10 ** (-6) * 3375 * 10 ** -15 * 3600

    print('this is the oxygen threshold: ox_threshold: ', ox_threshold)

    for current_step in tqdm.tqdm(range(1, n_steps + 1), disable=return_data):

        # allows to set the inflammation of the system to a fixed value.
        if fixed_inflammation != False:
            system_inflammation = fixed_inflammation

        system_inflammation_values = []

        if not get_diffused_grids:
            cell_indices = np.nonzero(env)
            cell_indices_list = list(zip(cell_indices[0], cell_indices[1]))
            random.shuffle(cell_indices_list)

            list_of_cells_we_already_had = []

            for y, x in cell_indices_list:
                if env[y][x].DNA_damage > DNA_damage_threshold and not env[y][x].cell_state == 0:
                    reasons_cell_death[env[y][x].cell_type]['DNA_damage'] += 1
                    env[y][x].death()
                elif not env[y][x] == 0 and not env[y][x].cell_state == 0:

                    # check whether we aren't iterating over cells twice during a run
                    if env[y][x].id in list_of_cells_we_already_had:
                        print(env[y][x].id)
                        print('we already did this cell')
                    list_of_cells_we_already_had.append(env[y][x].id)

                    # change the pathway activations based on the circumstances
                    if not proliferation_plot:
                        env[y][x].pathway_activation(grid_size, env, oxygen, resources, ros, signals)

                    # increase the time lived by 1h
                    env[y][x].time_lived += 1

                    # print(mass_cell * env[y][x].mass * 2.0e-8 * 3600 * resources[y][x] / (resources[y][x] +
                    #                                                                       S_nutrient), \
                    #         resources[y][x])

                    # cell lacks both oxygen and nutrients --> cell goes dormant and dies if completely phagotosed
                    if mass_cell * env[y][x].mass * metabolic_maintenance * \
                            resources[y][x] / (resources[y][x] + S_nutrient) > \
                            resources[y][x] and ox_threshold > oxygen[y][x]:
                        env[y][x].time_dormant += 1  # first dormant, if # steps dormant > threshold ---> die

                        if env[y][x].time_dormant > time_dormant_threshold:
                            reasons_cell_death[env[y][x].cell_type]['nut_ox_lack'] += 1
                            env[y][x].death()

                    # if a cell lacks either oxygen or nutrient it goes dormant but builds up stress and persistent
                    # stress, the cell dies

                    # if sufficient resources --> continue normal business
                    else:

                        if ox_threshold > oxygen[y][x]:
                            env[y][x].DNA_damage += stress_buildup

                            if env[y][x].DNA_damage > DNA_damage_threshold:
                                reasons_cell_death[env[y][x].cell_type]['DNA_damage'] += 1
                                env[y][x].death()

                        # build up DNA damage if the ROS is too high
                        if ros[y][x] > 2.5:
                            env[y][x].DNA_damage += DNA_damage_buildup
                        else:

                            # repair the DNA if the circumstances allow it
                            env[y][x].DNA_damage = max(0, env[y][x].DNA_damage - DNA_damage_buildup)

                        # multiply if we have grown enough
                        if env[y][x].mass >= mass_mitosis_threshold:
                            env, succeeded, cell_idx = do_mitosis(env, y, x, grid_size, ducts, cell_idx,
                                                                  params_tumour, params_tcell, params_bcell,
                                                                  params_myeloid, params_stromal)
                            # if the cell is not able to proliferate due to lack of space
                            if not succeeded:
                                env[y][x].mitosis_suppressed += 1
                                env[y][x].DNA_damage += DNA_damage_buildup

                                # cell dies if it is not able to differentiate for too long
                                if env[y][x].mitosis_suppressed > mitosis_suppression_threshold:
                                    reasons_cell_death[env[y][x].cell_type]['mitosis_supp'] += 1
                                    env[y][x].death()

                            # reset some of the mother parameters if it has divided
                            else:
                                reasons_cell_death[env[y][x].cell_type]['mitosis'] += 1
                                env[y][x].mitosis_suppressed = 0
                                env[y][x].DNA_damage = 0

                        # if a cell is not yet ready for mitosis
                        else:
                            env[y][x].mass += env[y][x].mass * env[y][x].proliferation_rate * resources[y][x] / (
                                    resources[y][x] + S_nutrient)

                        # chances of the cell dying increase as the cell approaches its average lifespan.
                        if env[y][x].calc_p_necrosis():
                            reasons_cell_death[env[y][x].cell_type]['old_age'] += 1
                            env[y][x].death()

                        # if the cell is not yet dead, do the following
                        if not env[y][x].cell_state == 0:

                            # activate pathways based on environmental circumstances
                            if not proliferation_plot:
                                system_inflammation_ind = env[y][x].pathway_response(system_inflammation)
                                system_inflammation_values.append(system_inflammation_ind)

                            # update signal production
                            env[y][x].update_signal_production(grid_size, env, system_inflammation, resources, signals)

                            # releases ROS
                            env[y][x].ros_production_modulation(grid_size, env)

                            # move the cell in a direction with a bias towards high nutrient and oxygen
                            if np.random.random() < migration_fraction:
                                env = cell_migrate(y, x, grid_size, resources, oxygen, env, ducts, ros)


                # keep track of dead time remove dead cells that have been
                elif env[y][x].cell_state == 0:
                    if env[y][x].time_dead > dead_cell_removal_time:
                        env[y][x] = 0
                    else:
                        env[y][x].time_dead += 1

        mass_matrix = GET_MASS(env)
        oxygen_consumption_matrix = mass_matrix
        metabolic_maintenance_matrix = mass_matrix * metabolic_maintenance

        # nutrient_consumption_matrix = GET_MASS(env)
        # oxygen_consumption_matrix = GET_MASS(env)
        system_inflammation = np.mean(system_inflammation_values)
        if system_inflammation > 1.2:
            n_cells_to_place = system_inflammation // 0.1
            reasons_cell_death['tcell']['spawned'] += n_cells_to_place
            env, cell_idx = spawn_immune_cells(cell_idx, env, ducts, n_cells_to_place, params_tcell, init_cms)

        if sum(sum(mass_matrix)) == 0:
            if not return_data:
                average_mass_list, average_ros_list, inflammation_list, num_cells_list, pathway_activation_list, \
                cms_classification_list = trim_lists(average_mass_list, average_ros_list, inflammation_list,
                                                     num_cells_list, pathway_activation_list, cms_classification_list)
            if save_data:
                save_raw_data(init_time, num_cells_list, average_mass_list, inflammation_list, pathway_activation_list,
                              cms_classification_list)
            if plot:
                create_final_plots(init_time, num_cells_list, cms_classification_list, average_mass_list,
                                   inflammation_list,
                                   pathway_list, pathway_activation_list, average_signal_list, all_signal_names,
                                   reasons_cell_death, average_ros_list, n_isolated_tumour_list)

            if do_animation:
                create_animation(init_time, current_step)

            print('all cells are dead')
            break

        int_env = MAP_CLASSES_TO_IDS(env)

        # save metrics of the simulation
        num_cells_list, average_mass_list, average_ros_list, inflammation_list, pathway_activation_list, \
        cms_classification_list, average_signal_list, n_isolated_tumour_list = save_metrics(current_step, env,
                                                                                            int_env, mass_matrix, ros,
                                                                                            pathway_list,
                                                                                            num_cells_list,
                                                                                            average_mass_list,
                                                                                            average_ros_list,
                                                                                            inflammation_list,
                                                                                            system_inflammation,
                                                                                            pathway_activation_list,
                                                                                            cms_classification_list,
                                                                                            average_signal_list,
                                                                                            n_isolated_tumour_list,
                                                                                            all_signal_names, signals,
                                                                                            classifier_obj)

        dicts_of_cancercell = GET_DICT_CELL_TYPE_SPEC(env, 'cancercell')
        cancer_cell_prolif_rate = GET_DICT_ITEM(dicts_of_cancercell, 'proliferation_rate')

        if plot:
            pathway_activation_list_trunc = pathway_activation_list[~np.all(pathway_activation_list == 0, axis=1)]
            average_signal_list_trunc = average_signal_list[~np.all(average_signal_list == 0, axis=1)]
            num_cells_list_trunc = num_cells_list[~np.all(num_cells_list == 0, axis=1)]

            update_figs(fig, ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, env, int_env, mass_matrix, resources, oxygen, ros,
                        cms_classification_list, pathway_activation_list, num_cells_list_trunc, cmap,
                        do_animation, init_time,
                        current_step, ducts)

        # do diffusion on the necessary substances
        resources, oxygen, ros, signals = diffusion_wrapper(grid_size, current_step, resources, mass_cell, mass_matrix,
                                                            GET_PROLIFERATION_RATE(env),
                                                            metabolic_maintenance_matrix, nutrient_diffusion_constant,
                                                            S_nutrient, ducts, oxygen, oxygen_consumption_matrix,
                                                            oxygen_diffusion_constant, ros, (env != 0).astype(int),
                                                            ros_diffusion_constant, signals, all_signal_names, env,
                                                            signal_diffusion_constant, influxes, proliferation_plot,
                                                            replenish_time, ox_modulation_value)

    if do_animation:
        create_animation(init_time, n_steps)

    if save_data:
        save_raw_data(init_time, num_cells_list, average_mass_list, inflammation_list, pathway_activation_list,
                      cms_classification_list)

    if plot:
        create_final_plots(init_time, num_cells_list, cms_classification_list, average_mass_list, inflammation_list,
                           pathway_list, pathway_activation_list, average_signal_list, all_signal_names,
                           reasons_cell_death, average_ros_list, n_isolated_tumour_list)

    if get_diffused_grids:
        np.savetxt('starting_configurations/resources_start_{}.txt'.format(init_cms), resources)
        np.savetxt('starting_configurations/oxygen_start_{}.txt'.format(init_cms), oxygen)
        np.savetxt('starting_configurations/ros_start_{}.txt'.format(init_cms), ros)

        if prediffision_step == 2:
            with open('starting_configurations/prediffused_signals_{}.pkl'.format(init_cms), 'wb') as output:
                pickle.dump(signals, output)
        else:
            run_sim(n_steps=n_steps, grid_size=grid_size, n_cells=n_cells, init_cms=init_cms,
                    fraction_types=fraction_types,
                    params_tumour=params_tumour, params_tcell=params_tcell, params_bcell=params_bcell,
                    params_myeloid=params_myeloid, params_stromal=params_stromal,
                    resources_init_value=resources_init_value,
                    oxygen_init_value=oxygen_init_value, ros_init_value=ros_init_value,
                    signals_init_value=signals_init_value,
                    influxes=influxes,
                    plot=False, do_animation=False, save_data=False, get_diffused_grids=True,
                    get_signalling_strengths=True,
                    proliferation_plot=proliferation_plot, proliferation_cell=proliferation_cell,
                    replenish_time=replenish_time, prediffision_step=2)

    if return_data:
        return {'init_time': init_time, 'num_cells_list': num_cells_list, 'average_mass_list': average_mass_list,
                'inflammation_list': inflammation_list, 'pathway_activation_list': pathway_activation_list,
                'cms_classification_list': cms_classification_list, 'average_signal_list': average_signal_list}


if __name__ == '__main__':

    # set variables
    grid_size = 100
    n_steps = 300
    n_cells = 20000
    init_cms = 'cms2'

    # set settings for the proliferation plots
    proliferation_plot = False
    proliferation_cell = 'tumour'
    replenish_time = 24

    # cancer cell, myeloid, tcell, bcell, stromal
    fraction_types = load_cell_fractions(init_cms)
    resources_init_value, oxygen_init_value, ros_init_value, signals_init_value = load_initial_grid_parameters()
    influxes = load_resource_influx(proliferation_plot)
    # fraction_types = np.array([0.04347826, 0.35431545, 0.40967724, 0.13286829, 0.10313901])

    # tumour_image_path = 'starting_configurations/tumour_cms4-2.jpg'
    tumour_image_path = 'starting_configurations/epithelial_cells.jpg'

    # y_range, x_range = (450, 550), (100, 200)
    y_range, x_range = (60, 160), (0, 100)

    # overwrite the initial nutrient deposit to the influx in case of a growth curve experiment
    if proliferation_plot:
        resources_init_value = influxes['nutrient']

    threshold_dict = load_pathway_activation_thresholds()
    base_signal_production_per_cell_type = load_base_signal_production()

    params_tumour, params_tcell, params_bcell, params_myeloid, params_stromal = \
        load_cell_parameters(threshold_dict, base_signal_production_per_cell_type)
    base_signal_production_per_cell_type = \
        load_inherent_signal_production(init_cms, base_signal_production_per_cell_type)

    run_sim(n_steps=n_steps, grid_size=grid_size, n_cells=n_cells, init_cms=init_cms, fraction_types=fraction_types,
            params_tumour=params_tumour, params_tcell=params_tcell, params_bcell=params_bcell,
            params_myeloid=params_myeloid, params_stromal=params_stromal, resources_init_value=resources_init_value,
            oxygen_init_value=oxygen_init_value, ros_init_value=ros_init_value, signals_init_value=signals_init_value,
            influxes=influxes,
            plot=True, do_animation=True, save_data=True, get_diffused_grids=False,
            proliferation_plot=proliferation_plot, proliferation_cell=proliferation_cell,
            replenish_time=replenish_time, tumour_image_path=tumour_image_path, tumour_image_y_range=y_range,
            tumour_image_x_range=x_range)
