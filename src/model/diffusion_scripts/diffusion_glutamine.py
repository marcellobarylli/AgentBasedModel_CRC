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

from fipy import Grid2D, TransientTerm, ImplicitSourceTerm, parallelComm, TSVViewer, FaceVariable, LinearGMRESSolver, \
    dump, ImplicitSourceTerm, CellVariable, DiffusionTerm, LinearLUSolver, LinearPCGSolver, Viewer

def diffusion_nutrients(nx, ny, mass_cell, u1, cellpresent1, growth_rate, metabolic_maintenance_matrix,
                        diffusion_constant, S, resources_supply, \
                                   influx_value):
    """ This function calculates the diffusion of nutrients as well as processes the consumption of them.

    """

    u1 = u1.flatten(order='F')
    cellpresent1 = cellpresent1.flatten(order='F')
    growth_rate = growth_rate.flatten(order='F')
    metabolic_maintenance = metabolic_maintenance_matrix.flatten(order='F')

    Yield = 100.0  # yield coefficient

    mesh = Grid2D(dx=1., dy=1., nx=nx, ny=ny)


    present = CellVariable(mesh=mesh, value=cellpresent1)
    growth_rate_cv = CellVariable(mesh=mesh, value=growth_rate)
    metabolic_maintenance_cv = CellVariable(mesh=mesh, value=metabolic_maintenance)
    uglutamine = CellVariable(mesh=mesh, value=u1, hasOld=True)

    diffusion_constant_matrix = cellpresent1.copy()
    diffusion_constant_matrix[diffusion_constant_matrix > 1] = diffusion_constant / 3
    diffusion_constant_matrix[diffusion_constant_matrix == 0] = diffusion_constant
    Dg = CellVariable(mesh=mesh, value=diffusion_constant_matrix)

    influx = resources_supply
    influx = influx.flatten(order='F')

    valuetopg = influx_value
    uglutamine.constrain(valuetopg, influx)

    res = 1.


    eq = TransientTerm(var=uglutamine) == DiffusionTerm(coeff=Dg, var=uglutamine) - ImplicitSourceTerm(
        mass_cell * ((growth_rate_cv / Yield) * present / (S + uglutamine) + metabolic_maintenance_cv * present / (S + uglutamine)),
        var=uglutamine)

    i = 0
    while res > 1.65e-13:  # change and see
        # gc.collect()
        # print(eq)
        # print(mysolver)
        eq.solve(dt=1, solver=LinearLUSolver())
        diff2 = uglutamine.old - uglutamine
        diff2 = np.absolute(diff2)
        res = max(diff2)
        # res = max(comm.allgather(res))
        minig = min(uglutamine)

        if minig < 0.:
            uglutamine.setValue(0.0, where=uglutamine < 0)
            uglutamine.updateOld()
            # print('there is a glut value below 0')
            # uglutamine[:] = uglutamine.old
        else:
            # print('This is the minimal glutamine: ', minig)
            uglutamine.updateOld()

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
    return np.reshape(uglutamine, (ny, nx), order='F'), present
