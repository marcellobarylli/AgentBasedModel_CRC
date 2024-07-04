"""
This file contains the parameters of the simulation which are loaded through separate functions
Author: Robin van den Berg
Contact: rvdb7345@gmail.com
"""

import numpy as np


def load_cell_parameters(threshold_dict, base_signal_production_per_cell):
    params_tumour = {'id': 0,
                     'proliferation_rate': 1 / 35,
                     'apoptosis_rate': 1 / 8000,
                     'nutrient_consumption': 1.06 * 10 ** (-16) * 3600,
                     'oxygen_consumption': 1.11 * 10 ** (-16) * 3600,
                     'nutrient_threshold': 0.6 * 3375 * 10 ** (-18) * 3600,
                     'oxygen_threshold': 0.15 * 10 ** (-6) * 3375 * 10 ** -15 * 3600,
                     'base_adhesiveness': 1,
                     'pathway_thresholds': threshold_dict,
                     'base_signal_production': base_signal_production_per_cell['tumour']}
    params_tcell = {'id': 0,
                    'proliferation_rate': 1 / 65.5,
                    'activated_proliferation_rate': 1 / 24,
                    'apoptosis_rate': 1 / 8000,
                    'nutrient_consumption': 1.06 * 10 ** (-16) * 3600,
                    'oxygen_consumption': 1.11 * 10 ** (-16) * 3600,
                    'nutrient_threshold': 0.6 * 3375 * 10 ** (-18) * 3600,
                    'oxygen_threshold': 0.15 * 10 ** (-6) * 3375 * 10 ** -15 * 3600,
                    'base_adhesiveness': 0.0,
                    'pathway_thresholds': threshold_dict,
                    'base_signal_production': base_signal_production_per_cell['tcell']}
    params_bcell = {'id': 0,
                    'activated_proliferation_rate': 1 / 36,
                    'proliferation_rate': 1 / 65.5,
                    'apoptosis_rate': 1 / 8000,
                    'nutrient_consumption': 1.06 * 10 ** (-16) * 3600,
                    'oxygen_consumption': 1.11 * 10 ** (-16) * 3600,
                    'nutrient_threshold': 0.6 * 3375 * 10 ** (-18) * 3600,
                    'oxygen_threshold': 0.15 * 10 ** (-6) * 3375 * 10 ** -15 * 3600,
                    'base_adhesiveness': 0.0,
                    'pathway_thresholds': threshold_dict,
                    'base_signal_production': base_signal_production_per_cell['bcell']}
    params_myeloid = {'id': 0,
                      'proliferation_rate': 1 / 34,
                      'apoptosis_rate': 1 / (300 * 24),
                      'nutrient_consumption': 1.06 * 10 ** (-16) * 3600,
                      'oxygen_consumption': 1.11 * 10 ** (-16) * 3600,
                      'nutrient_threshold': 0.6 * 3375 * 10 ** (-18) * 3600,
                      'oxygen_threshold': 0.15 * 10 ** (-6) * 3375 * 10 ** -15 * 3600,
                      'base_adhesiveness': 1,
                      'pathway_thresholds': threshold_dict,
                      'base_signal_production': base_signal_production_per_cell['myeloid']}
    params_stromal = {'id': 0,
                      'proliferation_rate': 1 / 37,
                      'apoptosis_rate': 1 / (570 * 24),
                      'nutrient_consumption': 1.06 * 10 ** (-16) * 3600,
                      'oxygen_consumption': 1.11 * 10 ** (-16) * 3600,
                      'nutrient_threshold': 0.6 * 3375 * 10 ** (-18) * 3600,
                      'oxygen_threshold': 0.15 * 10 ** (-6) * 3375 * 10 ** -15 * 3600,
                      'base_adhesiveness': 1,
                      'pathway_thresholds': threshold_dict,
                      'base_signal_production': base_signal_production_per_cell['stromal']}

    return params_tumour, params_tcell, params_bcell, params_myeloid, params_stromal


def load_base_signal_production():
    base_signal_production_per_cell = {
        'tumour': {
            'NFkB_production': 1.5,
            'STAT3_production': 1.5,
            'EGF_production': 1.5,
            'MYC_production': 1.5
        },
        'tcell': {
            'NFkB_production': 1.5,
            'STAT3_production': 1.5,
            'IL2_production': 1.0,
            'TNFalpha_production': 1.5,
            'EMT_signalling_production': 0.25,
            'IFNgamma_production': 1.5
        },
        'bcell': {
            'NFkB_production': 1.5,
            'STAT3_production': 1.5,
            'MYC_production': 1.5,
            'EMT_signalling_production': 0.25
        },
        'myeloid': {
            'NFkB_production': 1.5,
            'STAT3_production': 1.5,
            'IL6_production': 1.5,
            'TNFalpha_production': 1.5,
            'EMT_signalling_production': 1.5,
            'OPN_production': 1.5
        },
        'stromal': {
            'NFkB_production': 1.5,
            'STAT3_production': 1.5,
            'EMT_signalling_production': 1.5,
            'IL6_production': 1.2,
            'WNT_production': 1.5,
            'MYC_production': 1.5
        }
    }

    return base_signal_production_per_cell


def load_cell_fractions(cms):
    # cancer cell, myeloid, tcell, bcell, stromal

    if cms == 'cms1':
        fraction = np.array([0.025, 0.115, 0.48, 0.24, 0.13])
    elif cms == 'cms2':
        fraction = np.array([0.04, 0.12, 0.67, 0.15, 0.04])
    elif cms == 'cms3':
        fraction = np.array([0.04, 0.08, 0.59, 0.19, 0.12])
    elif cms == 'cms4':
        fraction = np.array([0.045, 0.32, 0.37, 0.12, 0.18])

    return fraction / sum(fraction)


# class pathway_activation_threshold_class():
#     def __init__(self, triiodothyronine_threshold, peroxisome_threshold, interferon_a_threshold, insulin_threshold, \
#            ROS_threshold, NFkB_threshold, EGF_threshold, WNT_threshold, EMT_signalling_threshold, STAT3_threshold, \
#            IL2_threshold, TNFalpha_threshold, IFN_gamma_threshold, IL6_threshold, MYC_threshold, Shh_threshold, \
#            metabolistic_signalling_threshold):
#
#         self.triiodothyronine_threshold = triiodothyronine_threshold
#         self.peroxisome_threshold = peroxisome_threshold
#         self.interferon_a_threshold = interferon_a_threshold
#         self.insulin_threshold = insulin_threshold
#
#         self.ROS_threshold = ROS_threshold
#         self.NFkB_threshold = NFkB_threshold
#         self.EGF_threshold = EGF_threshold
#         self.WNT_threshold = WNT_threshold
#         self.EMT_signalling_threshold = EMT_signalling_threshold
#         self.STAT3_threshold = STAT3_threshold
#         self.IL2_threshold = IL2_threshold
#         self.TNFalpha_threshold = TNFalpha_threshold
#         self.IFN_gamma_threshold = IFN_gamma_threshold
#         self.IL6_threshold = IL6_threshold
#         self.MYC_threshold = MYC_threshold
#         self.Shh_threshold = Shh_threshold
#         self.metabolistic_signalling_threshold = metabolistic_signalling_threshold

def load_pathway_activation_thresholds():
    triiodothyronine_threshold = 0.5
    peroxisome_threshold = 0.5
    interferon_a_threshold = 0.5
    insulin_threshold = 0.5

    ROS_threshold = 0.6783560785429451
    NFkB_threshold = 7.6174780940379843
    EGF_threshold = 5.5905972268825545
    WNT_threshold = 0.60949653605233884
    EMT_signalling_threshold = 5.51932083042646
    STAT3_threshold = 8.2840830976556106
    IL2_threshold = 1.724001092101165
    TNFalpha_threshold = 1.17011720874256378
    IFN_gamma_threshold = 1.18181407267048355
    IL6_threshold = 2.70074466018311812
    MYC_threshold = 2.3467913004504808
    Shh_threshold = 1.5000000011567264
    metabolistic_signalling_threshold = 1.5000000034023739
    OPN_threshold = 0.5

    return {'triiodothyronine_threshold': triiodothyronine_threshold, 'peroxisome_threshold': peroxisome_threshold,
            'interferon_a_threshold': interferon_a_threshold, 'insulin_threshold': insulin_threshold,
            'ROS_threshold': ROS_threshold, 'NFkB_threshold': NFkB_threshold, 'EGF_threshold': EGF_threshold,
            'WNT_threshold': WNT_threshold, 'EMT_signalling_threshold': EMT_signalling_threshold,
            'STAT3_threshold': STAT3_threshold,
            'IL2_threshold': IL2_threshold, 'TNFalpha_threshold': TNFalpha_threshold,
            'IFN_gamma_threshold': IFN_gamma_threshold,
            'IL6_threshold': IL6_threshold, 'MYC_threshold': MYC_threshold,
            'Shh_threshold': Shh_threshold,
            'metabolistic_signalling_threshold': metabolistic_signalling_threshold,
            'OPN_threshold': OPN_threshold}


def load_biological_parameters():
    S_nutrient = 0.1 * 3375 * 10 ** (-18)  # half saturation coefficient mol/gridspace
    stress_buildup = 1 / 24  # h-1
    DNA_damage_buildup = 1 / 10
    cell_stress_threshold = 1
    DNA_damage_threshold = 1
    time_dormant_threshold = 10
    mass_mitosis_threshold = 2
    mitosis_suppression_threshold = 9  # h
    dead_cell_removal_time = 4
    nutrient_diffusion_constant = 1600
    oxygen_diffusion_constant = 16000
    ros_diffusion_constant = 16000
    # signal_diffusion_constant = 24000
    signal_diffusion_constant = 160
    mass_cell = 27 * 10 ** (-12)
    migration_fraction = 0.9
    metabolic_maintenance = 2.0e-8 * 3600

    return S_nutrient, stress_buildup, cell_stress_threshold, time_dormant_threshold, mass_mitosis_threshold, \
           mitosis_suppression_threshold, \
           dead_cell_removal_time, nutrient_diffusion_constant, oxygen_diffusion_constant, ros_diffusion_constant, \
           signal_diffusion_constant, mass_cell, migration_fraction, DNA_damage_buildup, DNA_damage_threshold, \
           metabolic_maintenance


def load_resource_influx(proliferation_plot):
    """ This function specifies the amount of resources released on to the grid"""

    # parameters for standard simulations
    if not proliferation_plot:
        vessel_influxes = {
            'oxygen': 100 * 10 ** -13,
            'nutrient': 1 * 10 ** -15
        }
        return vessel_influxes

    # parameters for generating growth curves of the different cell types
    else:
        exp_deposits = {
            'oxygen': 5 * 10 ** -10,
            'nutrient': 10 ** -14
        }
        return exp_deposits


def load_initial_grid_parameters():
    resources_init_value = 2.0 * 3375 * 10 ** (-18)
    oxygen_init_value = 7.5 * 10 ** -13
    ros_init_value = 0.5
    signals_init_value = {
        'NFkB': 0.51086449,
        'insulin': 0.49684003,
        'EGF': 0.54201485,
        'WNT': 0.51042211,
        'EMT_signalling': 0.67947765,
        'STAT3': 0.51086449,
        'IL2': 0.56995903,
        'TNFalpha': 0.64299711,
        'IFNgamma': 0.60371808,
        'IL6': 0.51086449,
        'MYC': 0.44752881,
        'Shh': 0.3625372,
        'metabolistic_signalling': 0.53159275,
        'triiodothyronine': 0.5,
        'IFNalpha': 0.5,
        'OPN': 1.5
    }

    # signals_init_value = {
    #     'NFkB': 0.31086449,
    #     'insulin': 0.39684003,
    #     'EGF': 0.34201485,
    #     'WNT': 0.31042211,
    #     'EMT_signalling': 0.37947765,
    #     'STAT3': 0.31086449,
    #     'IL2': 0.36995903,
    #     'TNFalpha': 0.34299711,
    #     'IFNgamma': 0.30371808,
    #     'IL6': 0.31086449,
    #     'MYC': 0.34752881,
    #     'Shh': 0.3625372,
    #     'metabolistic_signalling': 0.33159275,
    #     'triiodothyronine': 0.3,
    #     'IFNalpha': 0.3
    # }

    return resources_init_value, oxygen_init_value, ros_init_value, signals_init_value


def load_inherent_signal_production(init_cms, base_signal_production_per_cell_type):
    if init_cms == "cms1":
        base_signal_production_per_cell_type['tumour'].update({'WNT_production': 0.51})
        base_signal_production_per_cell_type['stromal'].update({'WNT_production': 1.51})
        base_signal_production_per_cell_type['myeloid'].update({'OPN_production': 2.5})

    if init_cms == "cms2":
        base_signal_production_per_cell_type['tumour'].update({'WNT_production': 3.51})
        base_signal_production_per_cell_type['stromal'].update({'WNT_production': 3.51})
        base_signal_production_per_cell_type['myeloid'].update({'OPN_production': 0.2})

    if init_cms == "cms3":
        base_signal_production_per_cell_type['tumour'].update({'WNT_production': 0.51})
        base_signal_production_per_cell_type['stromal'].update({'WNT_production': 1.51})
        base_signal_production_per_cell_type['myeloid'].update({'OPN_production': 0.25})

    if init_cms == "cms4":
        base_signal_production_per_cell_type['tumour'].update({'WNT_production': 1.51})
        base_signal_production_per_cell_type['stromal'].update({'WNT_production': 1.51})
        base_signal_production_per_cell_type['myeloid'].update({'OPN_production': 1.19})

    return base_signal_production_per_cell_type
