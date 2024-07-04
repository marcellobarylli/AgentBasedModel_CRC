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


def diffusion_signal(nx, ny, u1, cellpresent1, signal_production, diffusion_constant, signal_name):
    u1 = u1.flatten(order='F')
    cellpresent1 = cellpresent1.flatten(order='F')
    cellnotpresent = 1 - cellpresent1

    signal_production = signal_production.flatten(order='F')

    deterioration = (np.ones((ny, nx)) * 0.087).flatten(order='F')

    mesh = Grid2D(dx=1., dy=1., nx=nx, ny=ny)

    present = CellVariable(mesh=mesh, value=cellpresent1)
    cellnotpresent = CellVariable(mesh=mesh, value=cellnotpresent)
    signal = CellVariable(mesh=mesh, value=u1, hasOld=True)
    signal_production = CellVariable(mesh=mesh, value=signal_production, hasOld=True)
    deterioration = CellVariable(mesh=mesh, value=deterioration, hasOld=True)

    diffusion_constant_matrix = cellpresent1
    diffusion_constant_matrix[diffusion_constant_matrix > 1] = diffusion_constant / 3
    diffusion_constant_matrix[diffusion_constant_matrix == 0] = diffusion_constant
    Dg = CellVariable(mesh=mesh, value=diffusion_constant_matrix)

    res = 1.

    eq = TransientTerm(var=signal) == DiffusionTerm(coeff=Dg, var=signal) + signal_production - ImplicitSourceTerm(
        deterioration * present, var=signal) - ImplicitSourceTerm(cellnotpresent * deterioration, var=signal)

    i = 0
    while res > 1.65e-6:  # change and see
        eq.solve(dt=1, solver=LinearLUSolver())
        diff2 = signal.old - signal
        diff2 = np.absolute(diff2)
        res = max(diff2)

        minig = min(signal)
        # minig = min(comm.allgather(minig))

        if minig < 0.:
            if signal_name == 'IL2':
                signal.setValue(1.724001092101165*2, where=signal > 1.724001092101165 * 2)

            signal.setValue(0.0, where=signal < 0.)
            signal.updateOld()
        else:
            if signal_name == 'IL2':
                signal.setValue(1.724001092101165*2, where=signal > 1.724001092101165 * 2)
            signal.updateOld()

        i += 1
        if i > 0:
            break

    return np.reshape(signal, (ny, nx), order='F'), present
