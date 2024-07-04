"""
This file contains the descriptions of the cells that are present on the grid
Author: Robin van den Berg
Contact: rvdb7345@gmail.com
"""

import numpy as np
# from main import get_moore
from numba import jit
from load_parameters import load_pathway_activation_thresholds
import json

clip = lambda x, l, u: l if x < l else u if x > u else x


@jit(nopython=True)
def get_moore(y, x, grid_size, radius=1, randomise=True):
    """ Get the coordinates of moore neighborhood """

    possible_x_values = range(max(0, x - radius), min(grid_size - 1, x + radius) + 1)
    possible_y_values = range(max(0, y - radius), min(grid_size - 1, y + radius) + 1)

    moore_neighbors = [(i, j) for i in possible_y_values for j in possible_x_values]

    # neumann_neighbors = [(min(y + 1, grid_size - 1), x),
    #              (y, min(x + 1, grid_size - 1)),
    #              (max(y - 1, 0), x),
    #              (y, max(x - 1, 0))]

    return moore_neighbors


MAP_CLASSES_TO_IDS = np.vectorize(lambda x: 0 if x == 0 else (
    -1 if x.cell_state == 0 else (
        1 if isinstance(x, cancercell) else
        (2 if isinstance(x, tcell) else
         (3 if isinstance(x, bcell) else
          (4 if isinstance(x, myeloid) else
           (5 if isinstance(x, stromalcell) else -10
            )))))), otypes=[int])

MAP_CELL_STATE = np.vectorize(lambda x: 0 if x == 0 else (
    -1 if x.cell_state == 0 else 1), otypes=[int])

MAP_TUMOUR_CELLS = np.vectorize(lambda x: 0 if x == 0 else (
    -1 if x.cell_state == 0 else (
        1 if isinstance(x, cancercell) else 2
    )), otypes=[int])

# global parameters regulating the strength of the pathways
oxidative_phosphorylation_strength = 1
glycolysis_strength = 0.5
P53_strength = 0.5
mTORC1_strength = 0.5
TNFaviaNFkB_strength = 0.5
unfolded_protein_response_strength = 0.5
hypoxia_strength = 0.5
EMT_strength = 0.5
myogenesis_strength = 0.5
MYC_target_v1_strength = 0.5
ROS_pathway_strength = 0.5
IL2_STAT5_signalling_strength = 0.5
peroxisome_strength = 0.5
adipogenesis_strength = 0.5
IFN_gamma_strength = 0.5
kras_signalling_strength = 0.5
IL6_JAK_strength = 0.5
complement_strength = 0.5
interferon_a_pathway_strength = 0.5
PI3K_strength = 0.5

# standard parameters
surrounding_cells_stress_threshold = 5
growth_factors_threshold = 0.5


# insulin_threshold = 0.5
# interferon_a_threshold = 0.5
# triiodothyronine_threshold = 0.5
# peroxisome_threshold = 0.5
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

# triiodothyronine_threshold, peroxisome_threshold, interferon_a_threshold, insulin_threshold, \
# ROS_threshold, NFkB_threshold, EGF_threshold, WNT_threshold, EMT_signalling_threshold, STAT3_threshold, \
# IL2_threshold, TNFalpha_threshold, IFN_gamma_threshold, IL6_threshold, MYC_threshold, Shh_threshold, \
# metabolistic_signalling_threshold = load_pathway_activation_thresholds()

# surrounding_cells_stress_threshold = 5
# growth_factors_threshold = 0.5
# insulin_threshold = 0.5
# ROS_threshold = 0.5
# interferon_a_threshold = 0.5
# IL6_threshold = 2.5225042029070512
# STAT3_threshold = 6.021750775898224
# NFkB_threshold = 5.498282142550543
# IFN_gamma_threshold = 0.07739197698113828
# WNT_threshold = 0.1527710277560626
# triiodothyronine_threshold = 0.5
# peroxisome_threshold = 0.5
# Shh_threshold = 0.5
# EGF_threshold = 0.16313237858984517
# TNFalpha_threshold = 0.06606538846720411
# MYC_threshold = 0.18806930948452763
# IL2_threshold = 0.08200484218995123
# EMT_signalling_threshold = 1.5829397950792234
# metabolistic_signalling_threshold = 0.5

def thres_func(value, threshold):
    return (value - threshold) / threshold

class cell():
    """ Defines a general cell without cell type specific functions

    Pathways system works in two-fold:
        - functions for adjusting the activation per pathway
        - one for carrying out the effects of the activation level
    """

    def __init__(self, id, proliferation_rate, apoptosis_rate, nutrient_consumption,
                 oxygen_consumption,
                 nutrient_threshold,
                 oxygen_threshold, base_adhesiveness, pathway_thresholds, base_signal_production):

        # defines the cell type specific parameters
        self.proliferation_rate = proliferation_rate
        self.base_proliferation_rate = proliferation_rate
        self.original_proliferation_rate = proliferation_rate
        self.apoptosis_rate = apoptosis_rate
        self.base_apoptosis_rate = apoptosis_rate
        self.nutrient_consumption = nutrient_consumption
        self.base_nutrient_consumption = nutrient_consumption
        self.oxygen_consumption = oxygen_consumption
        self.nutrient_threshold = nutrient_threshold
        self.oxygen_threshold = oxygen_threshold
        self.base_adhesiveness = base_adhesiveness
        self.ROS_production = 1
        self.crs = (0, 0)

        # defines simulation parameters
        self.time_lived = 0
        self.time_dormant = 0
        self.DNA_damage = 0
        self.time_dead = 0
        self.cell_stress = 0
        self.mass = 1
        self.cell_state = 1  # 1 represents alive while 0 represents necrotic

        self.id = id
        self.mitosis_suppressed = 0

        self.base_signal_production = base_signal_production

        # communally expression signals
        self.signals = {
            'NFkB_production': base_signal_production['NFkB_production'],
            'STAT3_production': base_signal_production['STAT3_production']
        }

        self.path_thres = pathway_thresholds

    def get_signal(self, signal):
        try:
            return self.signals[signal]
        except:
            return 0

    def initialise_pathways(self, cms_init):
        '''
        This function loads the initial activation values from a json file
        :param cms_init: the cms with which the cell should be initiated
        :return:
        '''

        with open('init_pathway_activation.json') as json_file:
            data = json.load(json_file)

        # pathway activation indicators
        # self.oxidative_phosphorylation_activation = data[cms_init][self.cell_type][
        #     'oxidative_phosphorylation_activation']
        # self.glycolysis_activation = data[cms_init][self.cell_type]['glycolysis_activation']
        self.P53_activation = data[cms_init][self.cell_type]['P53_activation']
        self.mTORC1_activation = data[cms_init][self.cell_type]['mTORC1_activation']
        self.TNFaviaNFkB_activation = data[cms_init][self.cell_type]['TNFaviaNFkB_activation']
        # self.unfolded_protein_response_activation = data[cms_init][self.cell_type][
        #     'unfolded_protein_response_activation']
        self.hypoxia_activation = data[cms_init][self.cell_type]['hypoxia_activation']
        self.EMT_activation = data[cms_init][self.cell_type]['EMT_activation']
        # self.myogenesis_activation = data[cms_init][self.cell_type]['myogenesis_activation']
        self.MYC_target_v1_activation = data[cms_init][self.cell_type]['MYC_target_v1_activation']
        self.ROS_pathway_activation = data[cms_init][self.cell_type]['ROS_pathway_activation']
        self.IL2_STAT5_signalling_activation = data[cms_init][self.cell_type]['IL2_STAT5_signalling_activation']
        # self.peroxisome_activation = data[cms_init][self.cell_type]['peroxisome_activation']
        self.adipogenesis_activation = data[cms_init][self.cell_type]['adipogenesis_activation']
        self.IFN_gamma_activation = data[cms_init][self.cell_type]['IFN_gamma_activation']
        self.kras_signalling_activation = data[cms_init][self.cell_type]['kras_signalling_activation']
        self.IL6_JAK_activation = data[cms_init][self.cell_type]['IL6_JAK_activation']
        self.complement_activation = data[cms_init][self.cell_type]['complement_activation']
        self.interferon_a_pathway_activation = data[cms_init][self.cell_type]['interferon_a_pathway_activation']
        self.PI3K_activation = data[cms_init][self.cell_type]['PI3K_activation']

    def add_pathways(self):
        '''
        This function adds the variables to save the activation values in
        :return:
        '''
        # pathway activation indicators
        # self.oxidative_phosphorylation_activation = 0
        # self.glycolysis_activation = 0
        self.P53_activation = 0
        self.mTORC1_activation = 0
        self.TNFaviaNFkB_activation = 0
        # self.unfolded_protein_response_activation = 0
        self.hypoxia_activation = 0
        self.EMT_activation = 0
        # self.myogenesis_activation = 0
        self.MYC_target_v1_activation = 0
        self.ROS_pathway_activation = 0
        self.IL2_STAT5_signalling_activation = 0
        # self.peroxisome_activation = 0
        self.adipogenesis_activation = 0
        self.IFN_gamma_activation = 0
        self.kras_signalling_activation = 0
        self.IL6_JAK_activation = 0
        self.complement_activation = 0
        self.interferon_a_pathway_activation = 0
        self.PI3K_activation = 0

    def copy_pathways(self, mother_cell):
        '''
        This function copies the pathway activations from another given cell, in mitosis the mother cell
        :param mother_cell: the cell from which to copy the pathway activations
        :return:
        '''

        # pathway activation indicators
        # self.oxidative_phosphorylation_activation = mother_cell.oxidative_phosphorylation_activation
        # self.glycolysis_activation = mother_cell.glycolysis_activation
        self.P53_activation = mother_cell.P53_activation
        self.mTORC1_activation = mother_cell.mTORC1_activation
        self.TNFaviaNFkB_activation = mother_cell.TNFaviaNFkB_activation
        # self.unfolded_protein_response_activation = mother_cell.unfolded_protein_response_activation
        self.hypoxia_activation = mother_cell.hypoxia_activation
        self.EMT_activation = mother_cell.EMT_activation
        # self.myogenesis_activation = mother_cell.myogenesis_activation
        self.MYC_target_v1_activation = mother_cell.MYC_target_v1_activation
        self.ROS_pathway_activation = mother_cell.ROS_pathway_activation
        self.IL2_STAT5_signalling_activation = mother_cell.IL2_STAT5_signalling_activation
        # self.peroxisome_activation = mother_cell.peroxisome_activation
        self.adipogenesis_activation = mother_cell.adipogenesis_activation
        self.IFN_gamma_activation = mother_cell.IFN_gamma_activation
        self.kras_signalling_activation = mother_cell.kras_signalling_activation
        self.IL6_JAK_activation = mother_cell.IL6_JAK_activation
        self.complement_activation = mother_cell.complement_activation
        self.interferon_a_pathway_activation = mother_cell.interferon_a_pathway_activation
        self.PI3K_activation = mother_cell.PI3K_activation

    def oxidative_phosphorylation_exp_reg(self, oxygen):

        if oxygen > self.oxygen_threshold:
            # self.oxidative_phosphorylation_activation = (oxygen - self.oxygen_threshold) / self.oxygen_threshold
            self.oxidative_phosphorylation_activation = -0.0879

        else:
            self.oxidative_phosphorylation_activation = 0
        # self.oxidative_phosphorylation_activation = 10000

    def ros_production_modulation(self, grid_size, env):
        '''
        This function calculates the ROS produced by a certain cell
        :param grid_size: the size of the grid
        :param env: the cell layer
        :return:
        '''

        surrounding_cells_coor = get_moore(self.crs[0], self.crs[1], grid_size, radius=1)
        num_surrounding_cells = np.count_nonzero((env[tuple(zip(*surrounding_cells_coor))]) != 0)
        self.ROS_production = 1.5 + clip(0.15 * (num_surrounding_cells - surrounding_cells_stress_threshold) / \
                                         surrounding_cells_stress_threshold, 0, 0.5)

    def glycolysis(self, nutrient):
        nutrient -= self.nutrient_consumption

    def death(self):
        self.cell_state = 0
        self.mass = 0.

        return

    def calc_p_necrosis(self):
        """
        This function returns True if it is time for a cell to commit apoptosis
        returns: boolean dictating whether to commit apoptosis
        """
        # print(self.time_lived , np.random.normal(1 / self.apoptosis_rate, (1 / self.apoptosis_rate) * 0.03))
        if self.time_lived > np.random.normal(1 / self.apoptosis_rate, (1 / self.apoptosis_rate) * 0.03):
            return True
        else:
            return False

    def migrate(self, y, x):
        '''
        This function adjusts the saved coordinates of the cell when it moves
        :param y: new y coordinate
        :param x: new x coordinate
        :return:
        '''

        self.crs = (y, x)

        return

    def cap_activation_values(self):
        '''
        This function caps the activation values so that the values remain between -1 and 1
        :return:
        '''

        # self.oxidative_phosphorylation_activation = max(-1, min(self.oxidative_phosphorylation_activation, 1))
        # self.glycolysis_activation = max(-1, min(self.glycolysis_activation, 1))
        self.P53_activation = max(-1, min(self.P53_activation, 1))
        self.mTORC1_activation = max(-1, min(self.mTORC1_activation, 1))
        self.TNFaviaNFkB_activation = max(-1, min(self.TNFaviaNFkB_activation, 1))
        # self.unfolded_protein_response_activation = max(-1, min(self.unfolded_protein_response_activation, 1))
        self.hypoxia_activation = max(-1, min(self.hypoxia_activation, 1))
        self.EMT_activation = max(-1, min(self.EMT_activation, 1))
        # self.myogenesis_activation = max(-1, min(self.myogenesis_activation, 1))
        self.MYC_target_v1_activation = max(-1, min(self.MYC_target_v1_activation, 1))
        self.ROS_pathway_activation = max(-1, min(self.ROS_pathway_activation, 1))
        self.IL2_STAT5_signalling_activation = max(-1, min(self.IL2_STAT5_signalling_activation, 1))
        # self.peroxisome_activation = max(-1, min(self.peroxisome_activation, 1))
        self.adipogenesis_activation = max(-1, min(self.adipogenesis_activation, 1))
        self.IFN_gamma_activation = max(-1, min(self.IFN_gamma_activation, 1))
        self.kras_signalling_activation = max(-1, min(self.kras_signalling_activation, 1))
        self.IL6_JAK_activation = max(-1, min(self.IL6_JAK_activation, 1))
        self.complement_activation = max(-1, min(self.complement_activation, 1))
        self.interferon_a_pathway_activation = max(-1, min(self.interferon_a_pathway_activation, 1))
        self.PI3K_activation = max(-1, min(self.PI3K_activation, 1))

    def pathway_activation(self, grid_size, env, oxygen, nutrients, ros, signals):
        '''
        This wrapper functions calls all the activation functions so that they are correctly set
        :param grid_size: size of the grid we use
        :param env: the cell layer object
        :param oxygen: the grid with the oxygen concentration
        :param nutrients: the grid with the nutrient concentration
        :param ros: the grid with the ros concentration
        :param signals: a dict with the grids with the signal concentration
        :return:
        '''
        # self.oxidative_phosphorylation_exp_reg(oxygen[self.crs[0]][self.crs[1]])
        self.P53_exp_reg(oxygen[self.crs[0]][self.crs[1]], signals)
        self.mTORC1_exp_reg(oxygen[self.crs[0]][self.crs[1]], nutrients[self.crs[0]][
            self.crs[1]], ros[self.crs[0]][self.crs[1]], signals)
        self.TNFaviaNFkB_exp_reg(ros[self.crs[0]][self.crs[1]], signals)
        # self.unfolded_protein_response_exp_reg()
        self.hypoxia_exp_reg(oxygen[self.crs[0]][self.crs[1]])

        self.EMT_exp_reg(oxygen[self.crs[0]][self.crs[1]], signals)
        self.ROS_pathway_exp_reg(ros[self.crs[0]][self.crs[1]])
        # self.peroxisome_exp_reg(signals)
        self.complement_exp_reg(env, grid_size)
        self.PI3K_exp_reg(oxygen[self.crs[0]][self.crs[1]], nutrients[self.crs[0]][
            self.crs[1]], ros[self.crs[0]][self.crs[1]], signals)

        self.MYC_target_v1_exp_reg(signals)
        self.IL2_STAT5_signalling_exp_reg(signals)
        self.adipogenesis_exp_reg(signals)
        self.IFN_gamma_exp_reg(signals)
        self.IL6_JAK_exp_reg(signals)
        self.interferon_a_pathway_exp_reg(signals)

        self.cap_activation_values()

        # self.myogenesis(signals) --> activation not clear yet
        # self.kras_signalling() --> through mutations? --> perhaps fix value at beginning fo simulation

    def pathway_response(self, system_inflammation=None):
        """
        This function applies the pathway activation values to the cells qualities
        :param system_inflammation: The inflammation coefficient of the current system
        :return: system inflammation after adjusted by the pathway activation of the cell
        """

        # apply pathway effects to the apoptosis rate
        self.apoptosis_rate = self.base_apoptosis_rate * (1 + P53_strength * self.P53_activation -
                                                          MYC_target_v1_strength * self.MYC_target_v1_activation +
                                                          ROS_pathway_strength * self.ROS_pathway_activation)

        # build up general proliferation activation factor
        activation_proliferation_factor = -P53_strength * self.P53_activation + \
                                          mTORC1_strength * self.mTORC1_activation + \
                                          MYC_target_v1_strength * self.MYC_target_v1_activation + \
                                          kras_signalling_strength * self.kras_signalling_activation - \
                                          ROS_pathway_strength * self.ROS_pathway_activation + \
                                          PI3K_strength * self.PI3K_activation

        # add cell type specific proliferation effects to the activation factor
        if self.cell_type == 'bcell' or self.cell_type == 'tcell':
            if self.IL2_STAT5_signalling_activation > 0.7:
                self.activated += 1
                self.base_proliferation_rate = self.activated_proliferation_rate
            elif self.IL2_STAT5_signalling_activation < 0.2:
                self.base_proliferation_rate = self.original_proliferation_rate
                self.activated = max(self.activated - 1, 0)
            else:
                self.base_proliferation_rate = self.original_proliferation_rate

        if self.cell_type == 'cancercell':
            activation_proliferation_factor += adipogenesis_strength * self.adipogenesis_activation + \
                                               IL6_JAK_strength * self.IL6_JAK_activation

        self.proliferation_rate = self.base_proliferation_rate * (1 + activation_proliferation_factor)

        # apply effects of pathways on the nutrient consumption
        self.nutrient_consumption = (1 + mTORC1_strength * self.mTORC1_activation + PI3K_strength *
                                     self.PI3K_activation)

        # apply effects of pathways on the system inflammation
        system_inflammation = TNFaviaNFkB_strength * self.TNFaviaNFkB_activation + IL6_JAK_strength * \
                              self.IL6_JAK_activation + complement_strength * \
                              self.complement_activation

        # apply effect of EMT on the adhesiveness of the cells
        self.adhesiveness = self.base_adhesiveness - self.EMT_activation

        return system_inflammation

    def P53_exp_reg(self, oxygen, signals):
        """ Expression because of DNA damage, oxidation stress (free radicals), osmotic values out of balance,
        hypoxia or overexpression of MYC """

        # self.P53_activation = self.DNA_damage - \
        #                       clip((oxygen - self.oxygen_threshold) / self.oxygen_threshold, -0.33, 0.0) + \
        #                       clip((signals['MYC'][self.crs[0]][self.crs[1]] - self.path_thres['MYC_threshold'])
        #                            / \
        #                            self.path_thres['MYC_threshold'], -0.33, 0.33)

        self.P53_activation = self.DNA_damage - \
                              clip(thres_func(oxygen, self.oxygen_threshold), -0.33, 0.0) + \
                              clip(thres_func(signals['MYC'][self.crs[0]][self.crs[1]], self.path_thres[
                                  'MYC_threshold']), -0.33, 0.33)

    def mTORC1_exp_reg(self, oxygen, nutrients, ROS, signals):
        self.mTORC1_activation = clip(thres_func(signals['insulin'][self.crs[0]][self.crs[1]],
                                                 self.path_thres['insulin_threshold']), -0.5, 0.5) + \
                                 clip(thres_func(ROS, self.path_thres['ROS_threshold']), -0.5, 0.5)

    def TNFaviaNFkB_exp_reg(self, ROS, signals):

        self.TNFaviaNFkB_activation = clip(thres_func(ROS, self.path_thres['ROS_threshold']), -0.5, 0.5) + \
                                      clip(thres_func(signals['TNFalpha'][self.crs[0]][self.crs[1]],
                                                      self.path_thres['TNFalpha_threshold']), -0.5, 0.5)

    def unfolded_protein_response_exp_reg(self):
        self.unfolded_protein_response_activation = self.cell_stress

    def hypoxia_exp_reg(self, oxygen):
        self.hypoxia_activation = clip(-thres_func(oxygen, self.oxygen_threshold), 0, 1)

    def EMT_exp_reg(self, oxygen, signals):
        self.EMT_activation = -clip(thres_func(oxygen, self.oxygen_threshold), -0.25, 0) + \
                              clip(thres_func(signals['WNT'][self.crs[0]][self.crs[1]],
                                              self.path_thres['WNT_threshold']), -0.25, 0.25) + \
                              clip(thres_func(signals['EMT_signalling'][self.crs[0]][self.crs[1]],
                                              self.path_thres['EMT_signalling_threshold']), -0.75, 0.75) + \
                              clip(thres_func(signals['OPN'][self.crs[0]][self.crs[1]],
                                              self.path_thres['OPN_threshold']), -0.5, 0.5)

    def MYC_target_v1_exp_reg(self, signals):
        self.MYC_target_v1_activation = clip(thres_func(signals['WNT'][self.crs[0]][self.crs[1]],
                                                        self.path_thres['WNT_threshold']), -0.33, 0.33) + \
                                        clip(thres_func(signals['Shh'][self.crs[0]][self.crs[1]],
                                                        self.path_thres['Shh_threshold']), -0.33, 0.33) + \
                                        clip(thres_func(signals['EGF'][self.crs[0]][self.crs[1]],
                                                        self.path_thres['EGF_threshold']), -0.33, 0.33)

    def ROS_pathway_exp_reg(self, ROS):
        self.ROS_pathway_activation = thres_func(ROS, self.path_thres['ROS_threshold'])

    def IL2_STAT5_signalling_exp_reg(self, signals):

        self.IL2_STAT5_signalling_activation = thres_func(signals['IL2'][self.crs[0]][self.crs[1]],
                                                          self.path_thres['IL2_threshold'])

    def adipogenesis_exp_reg(self, signals):
        self.adipogenesis_activation = -clip(thres_func(signals['WNT'][self.crs[0]][self.crs[1]],
                                                         self.path_thres['WNT_threshold']), -0.33, 0.33) + \
                                       clip(thres_func(signals['insulin'][self.crs[0]][self.crs[1]],
                                                       self.path_thres['insulin_threshold']), -0.33, 0.33) + \
                                       clip(thres_func(signals['metabolistic_signalling'][self.crs[0]][self.crs[1]],
                                                       self.path_thres['metabolistic_signalling_threshold']), -0.33,
                                            0.33)

    def IFN_gamma_exp_reg(self, signals):

        self.IFN_gamma_activation = thres_func(signals['IFNgamma'][self.crs[0]][self.crs[1]],
                                               self.path_thres['IFN_gamma_threshold'])

    def kras_signalling_exp_reg(self):
        mutations = False
        if mutations:
            self.kras_signalling_activation *= 1.01

    def IL6_JAK_exp_reg(self, signals):

        self.IL6_JAK_activation = clip(thres_func(signals['IL6'][self.crs[0]][self.crs[1]],
                                                  self.path_thres['IL6_threshold']), -0.33, 0.33) + \
                                  clip(thres_func(signals['STAT3'][self.crs[0]][self.crs[1]],
                                                  self.path_thres['STAT3_threshold']), -0.33, 0.33) + \
                                  clip(thres_func(signals['NFkB'][self.crs[0]][self.crs[1]],
                                        self.path_thres['NFkB_threshold']), -0.33, 0.33)

    def complement_exp_reg(self, cells, grid_size):

        surrounding_cells_coor = get_moore(self.crs[0], self.crs[1], grid_size, radius=5)
        num_dead_neighbours = np.count_nonzero(MAP_CELL_STATE(cells[tuple(zip(*surrounding_cells_coor))]) == -1)

        self.complement_activation = 0.1 * num_dead_neighbours

    def interferon_a_pathway_exp_reg(self, signals):
        self.interferon_a_pathway_activation = thres_func(signals['IFNalpha'][self.crs[0]][self.crs[1]],
                                                          self.path_thres['interferon_a_threshold'])

    def interferon_a_pathway_eff(self, signals):
        signals['STAT5'] *= 1 + interferon_a_pathway_strength * self.interferon_a_pathway_activation
        signals['STAT3'] *= 1 + interferon_a_pathway_strength * self.interferon_a_pathway_activation
        signals['MAPK'] *= 1 + interferon_a_pathway_strength * self.interferon_a_pathway_activation
        signals['PI3K'] *= 1 + interferon_a_pathway_strength * self.interferon_a_pathway_activation
        signals['NFkB'] *= 1 + interferon_a_pathway_strength * self.interferon_a_pathway_activation

    def PI3K_exp_reg(self, oxygen, nutrients, ROS, signals):

        self.PI3K_activation = clip(thres_func(signals['insulin'][self.crs[0]][self.crs[1]],
                                               self.path_thres['insulin_threshold']), -0.5, 0.5) + \
                               clip(thres_func(ROS, self.path_thres['ROS_threshold']), -0.5, 0.5)


class cancercell(cell):
    def __init__(self, id, proliferation_rate, apoptosis_rate, nutrient_consumption,
                 oxygen_consumption,
                 nutrient_threshold,
                 oxygen_threshold, base_adhesiveness, pathway_thresholds, base_signal_production):
        super().__init__(id, proliferation_rate, apoptosis_rate, nutrient_consumption,
                         oxygen_consumption,
                         nutrient_threshold,
                         oxygen_threshold, base_adhesiveness, pathway_thresholds, base_signal_production)
        self.cell_type = 'cancercell'
        self.signals["EGF_production"] = base_signal_production['EGF_production']
        self.signals["MYC_production"] = base_signal_production['MYC_production']

    def update_signal_production(self, grid_size, cells, inflammation, nutrients, signals=None):
        surrounding_cells_coor = get_moore(self.crs[0], self.crs[1], grid_size, radius=7)
        num_tumour_cells = np.count_nonzero(MAP_TUMOUR_CELLS(cells[tuple(zip(*surrounding_cells_coor))]) == 1)
        self.signals["EGF_production"] = self.base_signal_production['EGF_production'] + 0.01 * num_tumour_cells
        self.signals["MYC_production"] = self.base_signal_production['MYC_production'] - 0.05 * (
                nutrients[self.crs[0]][self.crs[1]] -
                self.nutrient_threshold) / self.nutrient_threshold
        self.signals["NFkB_production"] = self.base_signal_production['NFkB_production'] + 0.05 * self.DNA_damage
        self.signals["STAT3_production"] = self.base_signal_production['STAT3_production'] + 0.01 * num_tumour_cells


class myeloid(cell):
    def __init__(self, id, proliferation_rate, apoptosis_rate, nutrient_consumption,
                 oxygen_consumption,
                 nutrient_threshold,
                 oxygen_threshold, base_adhesiveness, pathway_thresholds, base_signal_production):
        super().__init__(id, proliferation_rate, apoptosis_rate, nutrient_consumption,
                         oxygen_consumption,
                         nutrient_threshold,
                         oxygen_threshold, base_adhesiveness, pathway_thresholds, base_signal_production)
        self.cell_type = 'myeloid'
        self.signals["IL6_production"] = base_signal_production['IL6_production']
        self.signals["TNFalpha_production"] = base_signal_production['TNFalpha_production']
        self.signals["EMT_signalling_production"] = base_signal_production['EMT_signalling_production']
        self.signals["OPN_production"] = base_signal_production['OPN_production']

    def update_signal_production(self, grid_size, cells, inflammation, nutrients, signals=None):
        surrounding_cells_coor = get_moore(self.crs[0], self.crs[1], grid_size, radius=7)
        num_tumour_cells = np.count_nonzero(MAP_TUMOUR_CELLS(cells[tuple(zip(*surrounding_cells_coor))]) == 1)
        self.signals["NFkB_production"] = self.base_signal_production['NFkB_production'] + 0.05 * self.DNA_damage
        self.signals["STAT3_production"] = self.base_signal_production['STAT3_production'] + 0.01 * num_tumour_cells
        self.signals["IL6_production"] = self.base_signal_production['IL6_production'] + 0.5 * inflammation
        self.signals["TNFalpha_production"] = self.base_signal_production[
                                                  'TNFalpha_production'] + 0.05 * self.hypoxia_activation

        if signals:
            opn_multiplier = max(1, signals['OPN'][self.crs[0]][self.crs[1]])
        else:
            opn_multiplier = 1
        self.signals["EMT_signalling_production"] = (self.base_signal_production[
                                                         'EMT_signalling_production'] + 0.01 * num_tumour_cells) * opn_multiplier


class tcell(cell):
    def __init__(self, id, proliferation_rate, activated_proliferation_rate, apoptosis_rate, nutrient_consumption,
                 oxygen_consumption,
                 nutrient_threshold,
                 oxygen_threshold, base_adhesiveness, pathway_thresholds, base_signal_production):
        super().__init__(id, proliferation_rate, apoptosis_rate, nutrient_consumption,
                         oxygen_consumption,
                         nutrient_threshold,
                         oxygen_threshold, base_adhesiveness, pathway_thresholds, base_signal_production)
        self.activated_proliferation_rate = activated_proliferation_rate
        self.cell_type = 'tcell'
        self.activated = 0
        self.signals["IL2_production"] = base_signal_production['IL2_production']
        self.signals["TNFalpha_production"] = base_signal_production["TNFalpha_production"]
        self.signals["EMT_signalling_production"] = base_signal_production["EMT_signalling_production"]
        self.signals["IFNgamma_production"] = base_signal_production["IFNgamma_production"]

    def update_signal_production(self, grid_size, cells, inflammation, nutrients, signals=None):
        surrounding_cells_coor = get_moore(self.crs[0], self.crs[1], grid_size, radius=7)
        num_tumour_cells = np.count_nonzero(MAP_TUMOUR_CELLS(cells[tuple(zip(*surrounding_cells_coor))]) == 1)
        if self.activated > 8:
            print('IL2 production de-activated')
            self.signals["IL2_production"] = 0
        else:
            self.signals["IL2_production"] = self.base_signal_production["IL2_production"] + 0.15 * (inflammation)
        self.signals["TNFalpha_production"] = self.base_signal_production[
                                                  "TNFalpha_production"] + 0.05 * self.hypoxia_activation
        if signals:
            opn_multiplier = max(1, signals['OPN'][self.crs[0]][self.crs[1]])
        else:
            opn_multiplier = 1
        self.signals["EMT_signalling_production"] = (self.base_signal_production[
                                                         "EMT_signalling_production"] + 0.01 * num_tumour_cells) * \
                                                    opn_multiplier
        self.signals["IFNgamma_production"] = self.base_signal_production[
                                                  "IFNgamma_production"] + 0.05 * self.IFN_gamma_activation
        self.signals["NFkB_production"] = self.base_signal_production["NFkB_production"] + 0.05 * self.DNA_damage
        self.signals["STAT3_production"] = self.base_signal_production["STAT3_production"] + 0.01 * num_tumour_cells


class bcell(cell):
    def __init__(self, id, proliferation_rate, activated_proliferation_rate, apoptosis_rate, nutrient_consumption,
                 oxygen_consumption,
                 nutrient_threshold,
                 oxygen_threshold, base_adhesiveness, pathway_thresholds, base_signal_production):
        super().__init__(id, proliferation_rate, apoptosis_rate, nutrient_consumption,
                         oxygen_consumption,
                         nutrient_threshold,
                         oxygen_threshold, base_adhesiveness, pathway_thresholds, base_signal_production)
        self.cell_type = 'bcell'
        self.activated_proliferation_rate = activated_proliferation_rate
        self.activated = 0

        self.signals["MYC_production"] = base_signal_production['MYC_production']
        self.signals["EMT_signalling_production"] = base_signal_production['EMT_signalling_production']

    def update_signal_production(self, grid_size, cells, inflammation, nutrients, signals=None):
        surrounding_cells_coor = get_moore(self.crs[0], self.crs[1], grid_size, radius=7)
        num_tumour_cells = np.count_nonzero(MAP_TUMOUR_CELLS(cells[tuple(zip(*surrounding_cells_coor))]) == 1)
        self.signals["NFkB_production"] = self.base_signal_production['NFkB_production'] + 0.05 * self.DNA_damage
        self.signals["STAT3_production"] = self.base_signal_production['STAT3_production'] + 0.01 * num_tumour_cells
        self.signals["MYC_production"] = self.base_signal_production['MYC_production'] - 0.05 * (
                nutrients[self.crs[0]][self.crs[1]] -
                self.nutrient_threshold) / self.nutrient_threshold

        if signals:
            opn_multiplier = max(1, signals['OPN'][self.crs[0]][self.crs[1]])
        else:
            opn_multiplier = 1
        self.signals["EMT_signalling_production"] = (self.base_signal_production[
                                                         'EMT_signalling_production'] + 0.01 * num_tumour_cells) \
                                                    * opn_multiplier


class stromalcell(cell):
    def __init__(self, id, proliferation_rate, apoptosis_rate, nutrient_consumption,
                 oxygen_consumption,
                 nutrient_threshold,
                 oxygen_threshold, base_adhesiveness, pathway_thresholds, base_signal_production):
        super().__init__(id, proliferation_rate, apoptosis_rate, nutrient_consumption,
                         oxygen_consumption,
                         nutrient_threshold,
                         oxygen_threshold, base_adhesiveness, pathway_thresholds, base_signal_production)

        self.cell_type = 'stromalcell'
        self.signals["EMT_signalling_production"] = base_signal_production['EMT_signalling_production']
        self.signals["IL6_production"] = base_signal_production['IL6_production']
        self.signals["WNT_production"] = base_signal_production['WNT_production']
        self.signals["MYC_production"] = base_signal_production['MYC_production']

    def update_signal_production(self, grid_size, cells, inflammation, nutrients, signals=None):
        surrounding_cells_coor = get_moore(self.crs[0], self.crs[1], grid_size, radius=7)
        num_tumour_cells = np.count_nonzero(MAP_TUMOUR_CELLS(cells[tuple(zip(*surrounding_cells_coor))]) == 1)

        if signals:
            opn_multiplier = max(1, signals['OPN'][self.crs[0]][self.crs[1]])
        else:
            opn_multiplier = 1
        self.signals["EMT_signalling_production"] = (self.base_signal_production['EMT_signalling_production'] + 0.3 * \
                                                     num_tumour_cells) * opn_multiplier
        self.signals["IL6_production"] = self.base_signal_production['IL6_production'] + 0.5 * inflammation
        self.signals["NFkB_production"] = self.base_signal_production['NFkB_production'] + 0.05 * self.DNA_damage
        self.signals["STAT3_production"] = self.base_signal_production['STAT3_production'] + 0.01 * num_tumour_cells
        self.signals["MYC_production"] = self.base_signal_production['MYC_production'] - 0.05 * (
                nutrients[self.crs[0]][self.crs[1]] -
                self.nutrient_threshold) / self.nutrient_threshold
