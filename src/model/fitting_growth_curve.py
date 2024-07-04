import curveball
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
# open the data file


# time_date = 162808441 # CMS1
# path = 'sim_results/sim_{}/raw_data/numcellsovertime.txt'.format(time_date)
#
# num_cells_over_time = np.loadtxt(path)
# time = np.arange(0, len(num_cells_over_time[:, 0]))
#
# data_points_time = [0, 24, 48, 72]
# data_points_cells_SW480 = np.array([7.4525745257452485, 12.466124661246624, 39.29539295392952, 80.35230352303523]) # CMS4
# data_points_cells_HCT116 = np.array([6.25, 25.5, 68.75, 175]) # CMS1 #
# data_points_cells_RKO = np.array([2.9166666666666714, 8.333333333333329, 24.861111111111114, 76.80555555555554]) # CMS1 #
# # data_points_cells_DLD1 = np.array([7.94979079497908, 15.620641562064165, 37.51743375174338, 85.63458856345886]) # NA #
#
# scaled_sim_x = (num_cells_over_time[:, 0]) / num_cells_over_time[0, 0]
# # scaled_data_points_cells = data_points_cells / data_points_cells[0]
# data_points_cells_SW480_scaled = data_points_cells_SW480 / data_points_cells_SW480[0]
# data_points_cells_HCT116_scaled = data_points_cells_HCT116 / data_points_cells_HCT116[0]
# data_points_cells_RKO_scaled = data_points_cells_RKO / data_points_cells_RKO[0]
#
# fig,ax = plt.subplots()
# plt.plot(time[0:], scaled_sim_x, "o", label='Sim-CMS1')
# plt.plot(data_points_time, data_points_cells_HCT116_scaled, label='HCT116-CMS1')
# plt.plot(data_points_time, data_points_cells_RKO_scaled, label='RKO-CMS1')
# plt.grid()
# plt.legend(fontsize=14)
# plt.ylabel('Cell count scaled (a.u.)', fontsize=14)
# plt.xlabel('Time (h)', fontsize=14)
# plt.show()
#
#
# time_date = 162808100 # CMS4
#
# path = 'sim_results/sim_{}/raw_data/numcellsovertime.txt'.format(time_date)
#
# num_cells_over_time = np.loadtxt(path)
# time = np.arange(0, len(num_cells_over_time[:, 0]))
#
# data_points_time = [0, 24, 48, 72]
# data_points_cells_SW480 = np.array([7.4525745257452485, 12.466124661246624, 39.29539295392952, 80.35230352303523]) # CMS4
# # data_points_cells_DLD1 = np.array([7.94979079497908, 15.620641562064165, 37.51743375174338, 85.63458856345886]) # NA #
#
# scaled_sim_x = (num_cells_over_time[:, 0]) / num_cells_over_time[0, 0]
# # scaled_data_points_cells = data_points_cells / data_points_cells[0]
# data_points_cells_SW480_scaled = data_points_cells_SW480 / data_points_cells_SW480[0]
#
#
# fig,ax = plt.subplots()
# plt.plot(time[0:], scaled_sim_x, "o", label='Sim-CMS4')
# plt.plot(data_points_time, data_points_cells_SW480_scaled, label='HCT116-CMS4')
# plt.grid()
# plt.legend(fontsize=14)
# plt.ylabel('Cell count scaled (a.u.)', fontsize=14)
# plt.xlabel('Time (h)', fontsize=14)
# plt.show()

''' Growth curve comparison '''
time_date = '162825291_CMS1'
path = 'sim_results/sim_{}/raw_data/numcellsovertime.txt'.format(time_date)
num_cells_over_time_CMS1 = np.loadtxt(path)
time = np.arange(0, len(num_cells_over_time_CMS1[:, 0]))
time_date = '162825417_CMS2'
path = 'sim_results/sim_{}/raw_data/numcellsovertime.txt'.format(time_date)
num_cells_over_time_CMS2 = np.loadtxt(path)
time_date = '162825428_CMS3'
path = 'sim_results/sim_{}/raw_data/numcellsovertime.txt'.format(time_date)
num_cells_over_time_CMS3 = np.loadtxt(path)
time_date = '162825444_CMS4'
path = 'sim_results/sim_{}/raw_data/numcellsovertime.txt'.format(time_date)
num_cells_over_time_CMS4 = np.loadtxt(path)

fig,ax = plt.subplots()
plt.plot(time[0:], num_cells_over_time_CMS1[:, 0], "-", label='CMS1')
plt.plot(time[0:], num_cells_over_time_CMS2[:, 0], "-", label='CMS2')
plt.plot(time[0:], num_cells_over_time_CMS3[:, 0], "-", label='CMS3')
plt.plot(time[0:], num_cells_over_time_CMS4[:, 0], "-", label='CMS4')
plt.grid()
plt.legend(fontsize=14)
plt.ylabel('Cell count scaled (a.u.)', fontsize=14)
plt.xlabel('Time (h)', fontsize=14)
plt.show()


#
# cancer_data_df = pd.DataFrame(np.log(num_cells_over_time[:, 0]), columns=['OD'])
# cancer_data_df['Time'] = time
# print(cancer_data_df)
#
# models, fig, ax = curveball.models.fit_model(cancer_data_df, PLOT=True, PRINT=False)
# # models, fig, ax = curveball.models.fit_model(cancer_data_df, PLOT=True, PRINT=False, param_max={'y0': 9})
#
# plt.show()
#
#
# # do logarithmic fit
# def objective(x, l, m, n):
# 	return l * np.exp(m * x) + n
#
# popt, _ = curve_fit(objective, time[0:300], num_cells_over_time[0:300, 0])
# l, m, n = popt
# print(l, m, n)
#
# y_new = objective(time[0:300], l, m, n)
#
# plt.figure()
# plt.plot(time[0:300], num_cells_over_time[0:300, 0], "o")
# plt.plot(time[0:300], y_new)
# plt.show()



#
# from croissance import process_curve
# result = process_curve(cancer_data_df['OD'])
# print(result.growth_phases)