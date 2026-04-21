'''
   MFEM example 0 (converted from ex0.cpp)

   See c++ version in the MFEM library for more detail

   How to run:
      python <arguments>

   Example of arguments:
      ex1.py -m star.mesh
      ex1.py -m fichera.mesh -o 2

   Description: This example code demonstrates the most basic usage of MFEM to
                define a simple finite element discretization of the Laplace
                problem -Delta u = 1 with zero Dirichlet boundary conditions.
                General 2D/3D mesh files and finite element polynomial degrees
                can be specified by command line options.

'''
import os
from os.path import expanduser, join
import numpy as np

import mfem.par as mfem
from mpi4py import MPI

num_procs = MPI.COMM_WORLD.size
myid = MPI.COMM_WORLD.rank
smyid = '{:0>6d}'.format(myid)


def get_full_assembly_level():
    enum_type = getattr(mfem, 'AssemblyLevel', None)
    if enum_type is not None and hasattr(enum_type, 'FULL'):
        return enum_type.FULL
    if hasattr(mfem, 'AssemblyLevel_FULL'):
        return mfem.AssemblyLevel_FULL
    raise AttributeError('FULL assembly level is not available in this PyMFEM build.')


def run(order=1, meshfile='', visualization=False, use_cpardiso=False,
    use_mumps=False, use_full_assembly=False, refinement_levels=1):
    '''
    run ex0
    '''

    #  2. Read the mesh from the given mesh file and refine uniformly.
    serial_mesh = mfem.Mesh(meshfile)
    mesh = mfem.ParMesh(MPI.COMM_WORLD, serial_mesh)
    for _ in range(refinement_levels):
        mesh.UniformRefinement()

    if use_cpardiso and use_full_assembly:
        if myid == 0:
            print('CPardiso with FULL assembly is disabled in this example; '
                  'falling back to LEGACY assembly to avoid a PyMFEM/MFEM crash.')
        use_full_assembly = False

    # 3. Define a finite element space on the mesh. Here we use H1 continuous
    #    high-order Lagrange finite elements of the given order.
    fec = mfem.H1_FECollection(order,  mesh.Dimension())
    fespace = mfem.ParFiniteElementSpace(mesh, fec)
    gtdof = fespace.GlobalTrueVSize()

    if myid == 0:
        print('Number of finite element unknowns: ' + str(gtdof))

    # 4. Extract the list of all the boundary DOFs. These will be marked as
    #    Dirichlet in order to enforce zero boundary conditions.
    boundary_dofs = mfem.intArray()
    fespace.GetBoundaryTrueDofs(boundary_dofs)

    # 5. Define the solution x as a finite element grid function in fespace. Set
    #    the initial guess to zero, which also sets the boundary conditions.
    x = mfem.ParGridFunction(fespace)
    x.Assign(0.0)

    # 6. Set up the linear form b(.) corresponding to the right-hand side.
    one = mfem.ConstantCoefficient(1.0)
    b = mfem.ParLinearForm(fespace)
    b.AddDomainIntegrator(mfem.DomainLFIntegrator(one))
    b.Assemble()

    # 7. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
    a = mfem.ParBilinearForm(fespace)
    a.AddDomainIntegrator(mfem.DiffusionIntegrator(one))
    if use_full_assembly:
        a.SetAssemblyLevel(get_full_assembly_level())
    a.Assemble()

    # 8. Form the linear system A X = B. This includes eliminating boundary
    #    conditions, applying AMR constraints, and other transformations.
    A = mfem.HypreParMatrix()
    B = mfem.Vector()
    X = mfem.Vector()
    a.FormLinearSystem(boundary_dofs, x, b, A, X, B)

    # 9. Solve the system using CPardiso, MUMPS, or PCG with BoomerAMG.
    if use_cpardiso:
        try:
            from mfem._par.cpardiso import CPardisoSolver
            start_time = MPI.Wtime()
            cpardiso = CPardisoSolver(MPI.COMM_WORLD)
            cpardiso.SetMatrixType(CPardisoSolver.REAL_NONSYMMETRIC)
            cpardiso.SetPrintLevel(1 if myid == 0 else 0)
            cpardiso.SetOperator(A)
            if myid == 0:
                print('CPardiso is available, running with Cluster Pardiso direct solver')
            cpardiso.Mult(B, X)
            if myid == 0:
                print('CPardiso solve completed in', MPI.Wtime() - start_time, 'seconds')
        except Exception as e:
            if myid == 0:
                print('CPardiso not available, falling back to PCG+BoomerAMG:', e)
            use_cpardiso = False

    if use_mumps and not use_cpardiso:
        try:
            from mfem._par.mumps import MUMPSSolver
            start_time = MPI.Wtime()
            mumps = MUMPSSolver(MPI.COMM_WORLD)
            mumps.SetMatrixSymType(MUMPSSolver.SYMMETRIC_POSITIVE_DEFINITE)
            mumps.SetPrintLevel(1 if myid == 0 else 0)
            mumps.SetOperator(A)
            if myid == 0:
                print('MUMPS is available, running with MUMPS direct solver')
            mumps.Mult(B, X)
            if myid == 0:
                print('MUMPS solve completed in', MPI.Wtime() - start_time, 'seconds')
        except Exception as e:
            if myid == 0:
                print('MUMPS not available, falling back to PCG+BoomerAMG:', e)
            use_mumps = False

    if not use_cpardiso and not use_mumps:
        start_time = MPI.Wtime()
        M = mfem.HypreBoomerAMG(A)
        cg = mfem.CGSolver(MPI.COMM_WORLD)
        cg.SetRelTol(1e-12)
        cg.SetMaxIter(2000)
        cg.SetPrintLevel(0)
        cg.SetPreconditioner(M)
        cg.SetOperator(A)
        cg.Mult(B, X)
        if myid == 0:
            print('PCG+BoomerAMG solve completed in', MPI.Wtime() - start_time, 'seconds')
    # 10. Recover the solution x as a grid function and save to file. The output
    #     can be viewed using GLVis as follows: "glvis -np <np> -m mesh -g sol"
    a.RecoverFEMSolution(X, b, x)
    x.Save('sol.'+smyid)
    mesh.Print('mesh.'+smyid)

    # 11. Send the solution by socket to a GLVis server.
    if visualization:
        sol_sock = mfem.socketstream("localhost", 19916)
        sol_sock.send_text("parallel " + str(num_procs) + " " + str(myid))
        sol_sock.precision(8)
        sol_sock.send_solution(mesh, x)


if __name__ == "__main__":
    from mfem.common.arg_parser import ArgParser

    parser = ArgParser(description='Ex1 (Laplace Problem)')
    parser.add_argument('-m', '--mesh',
                        default='inline-hex.mesh',
                        action='store', type=str,
                        help='Mesh file to use.')
    parser.add_argument('-o', '--order',
                        action='store', default=1, type=int,
                        help="Finite element order (polynomial degree) or -1 for isoparametric space.")
    parser.add_argument('-r', '--refinement-levels',
                        action='store', default=1, type=int,
                        help='Number of uniform mesh refinement levels.')
    parser.add_argument('-vis', '--visualization',
                        action='store_true',
                        help='Enable GLVis visualization')
    parser.add_argument('--use-cpardiso',
                        action='store_true', default=False,
                        help='Use Intel MKL Cluster Pardiso direct solver instead of PCG+BoomerAMG.')
    parser.add_argument('-mumps', '--use-mumps',
                        action='store_true', default=False,
                        help='Use MUMPS direct solver instead of PCG+BoomerAMG.')
    parser.add_argument('--full-assembly',
                        action='store_true', default=False,
                        help='Use MFEM FULL assembly before forming the parallel linear system.')

    args = parser.parse_args()
    if myid == 0:
        parser.print_options(args)

    order = args.order
    meshfile = expanduser(
        join(os.path.dirname(__file__), '..', 'data', args.mesh))

    run(order=order,
        meshfile=meshfile,
        visualization=args.visualization,
        use_cpardiso=args.use_cpardiso,
        use_mumps=args.use_mumps,
        use_full_assembly=args.full_assembly,
        refinement_levels=args.refinement_levels)
