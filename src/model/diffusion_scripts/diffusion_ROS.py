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

def diffusion_ROS(nx, ny, u1, cell_locations, ros_neutralisation, ros_production, diffusion_constant):

    u1 = u1.flatten(order='F')
    cell_locations = cell_locations.flatten(order='F').astype(float)
    ros_production = ros_production.flatten(order='F')
    cellnotpresent = 1 - cell_locations

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

    cell_locations = CellVariable(mesh=mesh, value=cell_locations)
    cellnotpresent = CellVariable(mesh=mesh, value=cellnotpresent)

    ROS = CellVariable(mesh=mesh, value=u1, hasOld=True)
    # ROS.constrain(0, where=mesh.exteriorFaces)
    ros_neutralisation = CellVariable(mesh=mesh, value=ros_neutralisation)
    ros_production = CellVariable(mesh=mesh, value=ros_production)

    diffusion_constant_matrix = cell_locations.copy()
    diffusion_constant_matrix[diffusion_constant_matrix > 0] = 0.1
    diffusion_constant_matrix[diffusion_constant_matrix == 0] = diffusion_constant
    Dg = CellVariable(mesh=mesh, value=diffusion_constant_matrix)
    # Dg = Dg_cell.getFaceGrad().dot((1,0,0))

    # Dg = FaceVariable(mesh=mesh, value=diffusion_constant)  # 0.01
    # fX, fY = mesh.faceCenters
    # Entry = mesh.facesTop
    # print(Entry)
    # influx = np.zeros((ny, nx), dtype=bool)
    # influx[int(ny/2) - 5:int(ny/2) + 5, -1] = True
    # influx = influx.flatten(order='F')

    # valuetopg = 1.06 * 10**(-13)
    # ROS.constrain(valuetopg, influx)
    # vi = Viewer(uglutamine)
    # vii = Viewer(present)
    
    # mysolver = PysparseSolver()
    res = 1.
    # print(uglutamine)
    # print(present)

    # print('these are the cells: \n', cellpresent1)
    # print('these are the glutamine: \n', u1)
    eq = TransientTerm(var=ROS) == DiffusionTerm(coeff=Dg, var=ROS) - cell_locations * ros_neutralisation + \
         ros_production - ImplicitSourceTerm(cellnotpresent * 0.087, var=ROS)

    i = 0
    while res > 1.65e-6:  # change and see
        eq.solve(dt=1, solver=LinearLUSolver())
        endt = time.time()
        diff2 = ROS.old - ROS
        diff2 = np.absolute(diff2)
        res = max(diff2)

        minig = min(ROS)
        # minig = min(comm.allgather(minig))

        if minig < 0.:
            ROS.setValue(0.0, where=ROS < 0.)
            ROS.updateOld()
        else:
            ROS.updateOld()

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
    return np.reshape(ROS, (ny, nx), order='F')
