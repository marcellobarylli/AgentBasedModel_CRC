import sys
import os
import matplotlib.pyplot as plt

# print(os.environ['PATH'])
# sys.path.append('/usr/local/lib/python3.7/site-packages')
#
# from mpi4py import MPI
#
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# import gc

import time



start = time.time()
import numpy as np
from decimal import Decimal

# from fipy.solvers.trilinos.preconditioners import MultilevelSAPreconditioner

from fipy.variables.variable import Variable

# from fipy.solvers.pysparse import PysparseSolver
# print('This would be the default sovlers: ', DefaultSolver)
from fipy import Grid2D, TransientTerm, ImplicitSourceTerm, parallelComm, TSVViewer, FaceVariable, LinearGMRESSolver, \
    dump, ImplicitSourceTerm, CellVariable, DiffusionTerm, LinearLUSolver, LinearPCGSolver, Viewer

def diffusion_oxygen(nx, ny, u1, oxygen_consumption, diffusion_constant, resources_supply, influx_value):

    u1 = u1.flatten(order='F')
    oxygen_consumption = oxygen_consumption.flatten(order='F')
    # nx = 657
    # ny = 527
    # prefixfolder = "/media/data/cc3dcode/gghcistax/Simulation/"
    # massofpix = 3.63e-7
    # dx = 1.
    # dy = dx
    # largeValue = 1000000000000000000000000000000000000.
    # Kcg = 8.75e-6
    # m = 2.0e-8
    # Y = 100.0
    # S = 2.75e-14  # 0.5
    # L = dx * nx

    mesh = Grid2D(dx=1., dy=1., nx=nx, ny=ny)
    # namer = "uglutamine.dat"
    # u1 = np.loadtxt(namer)
    # namer = "cellpresent.dat"
    # cellpresent1 = np.loadtxt(namer)

    # X, Y = mesh.cellCenters

    oxygen_consumption_cv = CellVariable(mesh=mesh, value=oxygen_consumption)
    oxygen = CellVariable(mesh=mesh, value=u1, hasOld=True)

    diffusion_constant_matrix = oxygen_consumption.copy()
    diffusion_constant_matrix[diffusion_constant_matrix > 0] = diffusion_constant / 3
    diffusion_constant_matrix[diffusion_constant_matrix == 0] = diffusion_constant
    Dg = CellVariable(mesh=mesh, value=diffusion_constant_matrix)

    influx = resources_supply
    influx = influx.flatten(order='F')


    valuetopg = influx_value
    oxygen.constrain(valuetopg, influx)

    res = 1.
    # print(uglutamine)
    # print(present)

    # print('these are the cells: \n', cellpresent1)
    # print('these are the glutamine: \n', u1)

    Vmax = 6.25 * 10 ** (-17) * 3600
    km = 1.33 * 10 ** (-3) * 3375 * 10**-18
    ox_array = oxygen_consumption * Vmax / (km + u1) * u1
    print('max oxygen consumption: ', ox_array.max())
    print('mean oxygen consumption: ', ox_array.mean())
    print('mean oxygen on grid: ', u1.mean())
    print('sum oxygen consumption: ', ox_array.sum())
    print('sum of influx: ', (valuetopg*influx).sum())

    eq = TransientTerm(var=oxygen) == DiffusionTerm(coeff=Dg, var=oxygen) - ImplicitSourceTerm(
        (oxygen_consumption_cv * Vmax / (km + oxygen)), var=oxygen)

    i = 0
    while res > 1.65e-13:  # change and see
        # gc.collect()
        # print(eq)
        # print(mysolver)
        eq.solve(dt=1, solver=LinearLUSolver())
        endt = time.time()
        diff2 = oxygen.old - oxygen
        diff2 = np.absolute(diff2)
        res = 100
        # res = max(comm.allgather(res))
        minig = min(oxygen)
        # minig = min(comm.allgather(minig))

        if minig < 0.:
            oxygen.setValue(0.0, where=oxygen < 0.)
            oxygen.updateOld()
        else:
            oxygen.updateOld()

        i+=1
        if i > 0:
            break

        # if i % 100 == 0:
        #     vi.plot()
        #     vii.plot()
        #
        # i+=1
        # if rank == 1:
        #     print("Resolution", res, "minimum g", minig, "time", endt - start)

    # namer = "uglutamine.dat"
    # dump.write(uglutamine, namer)
    # uglutamine = dump.read(namer)
    # namer = "present.dat"
    # dump.write(present, namer)
    # present = dump.read(namer)
    return np.reshape(oxygen, (ny, nx), order='F')
