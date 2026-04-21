'''
   MFEM example 0 (converted from ex0.cpp)

   See c++ version in the MFEM library for more detail

   How to run:
      python <arguments>

   Example of arguments:
      ex0.py -m star.mesh
      ex0.py -m fichera.mesh -o 2
      ex0.py -m star.mesh --use-pardiso

   Description: This example code demonstrates the most basic usage of MFEM to
                define a simple finite element discretization of the Laplace
                problem -Delta u = 1 with zero Dirichlet boundary conditions.
                General 2D/3D mesh files and finite element polynomial degrees
                can be specified by command line options.

'''
import os
from os.path import expanduser, join
import numpy as np

import mfem.ser as mfem


def get_full_assembly_level():
    enum_type = getattr(mfem, 'AssemblyLevel', None)
    if enum_type is not None and hasattr(enum_type, 'FULL'):
        return enum_type.FULL
    if hasattr(mfem, 'AssemblyLevel_FULL'):
        return mfem.AssemblyLevel_FULL
    raise AttributeError('FULL assembly level is not available in this PyMFEM build.')


def run(order=1, meshfile='', use_pardiso=False, use_full_assembly=False,
    refinement_levels=1):
    '''
    run ex0
    '''

    #  2. Read the mesh from the given mesh file and refine uniformly.
    mesh = mfem.Mesh(meshfile, 1, 1)
    for _ in range(refinement_levels):
        mesh.UniformRefinement()

    # 3. Define a finite element space on the mesh. Here we use H1 continuous
    #    high-order Lagrange finite elements of the given order.
    fec = mfem.H1_FECollection(order,  mesh.Dimension())
    fespace = mfem.FiniteElementSpace(mesh, fec)
    print('Number of finite element unknowns: ' +
          str(fespace.GetTrueVSize()))

    # 4. Extract the list of all the boundary DOFs. These will be marked as
    #    Dirichlet in order to enforce zero boundary conditions.
    boundary_dofs = mfem.intArray()
    fespace.GetBoundaryTrueDofs(boundary_dofs)

    # 5. Define the solution x as a finite element grid function in fespace. Set
    #    the initial guess to zero, which also sets the boundary conditions.
    x = mfem.GridFunction(fespace)
    x.Assign(0.0)

    # 6. Set up the linear form b(.) corresponding to the right-hand side.
    one = mfem.ConstantCoefficient(1.0)
    b = mfem.LinearForm(fespace)
    b.AddDomainIntegrator(mfem.DomainLFIntegrator(one))
    b.Assemble()

    # 7. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
    a = mfem.BilinearForm(fespace)
    a.AddDomainIntegrator(mfem.DiffusionIntegrator(one))
    if use_full_assembly:
        a.SetAssemblyLevel(get_full_assembly_level())
    a.Assemble()

    # 8. Form the linear system A X = B. This includes eliminating boundary
    #    conditions, applying AMR constraints, and other transformations.
    A = mfem.SparseMatrix()
    B = mfem.Vector()
    X = mfem.Vector()
    a.FormLinearSystem(boundary_dofs, x, b, A, X, B)
    print("Size of linear system: " + str(A.Height()))

    # 9. Solve the system.
    if use_pardiso:
        # Requires MFEM built with MFEM_USE_MKL_PARDISO=YES (-C"with-mkl-pardiso=Yes")
        from mfem._ser.pardiso import PardisoSolver
        import time
        t_start = time.time()
        solver = PardisoSolver()
        solver.SetPrintLevel(0)
        solver.SetOperator(A)
        solver.Mult(B, X)
        elapsed = time.time() - t_start
        print(f"PardisoSolver elapsed time: {elapsed:.6f} seconds")
    else:
        # Default: PCG with symmetric Gauss-Seidel preconditioner.
        M = mfem.GSSmoother(A)
        mfem.PCG(A, M, B, X, 1, 200, 1e-12, 0.0)

    # 10. Recover the solution x as a grid function and save to file. The output
    #     can be viewed using GLVis as follows: "glvis -m mesh.mesh -g sol.gf"
    a.RecoverFEMSolution(X, b, x)
    x.Save('sol.gf')
    mesh.Save('mesh.mesh')


if __name__ == "__main__":
    from mfem.common.arg_parser import ArgParser

    parser = ArgParser(description='Ex0 (Laplace Problem)')
    parser.add_argument('-m', '--mesh',
                        default='star.mesh',
                        action='store', type=str,
                        help='Mesh file to use.')
    parser.add_argument('-o', '--order',
                        action='store', default=1, type=int,
                        help="Finite element order (polynomial degree) or -1 for isoparametric space.")
    parser.add_argument('-r', '--refinement-levels',
                        action='store', default=1, type=int,
                        help='Number of uniform mesh refinement levels.')
    parser.add_argument('--use-pardiso',
                        action='store_true', default=False,
                        help='Use Intel MKL Pardiso direct solver (requires MFEM_USE_MKL_PARDISO).')
    parser.add_argument('--full-assembly',
                        action='store_true', default=True,
                        help='Use MFEM FULL assembly before forming the sparse linear system.')

    args = parser.parse_args()
    parser.print_options(args)

    order = args.order
    meshfile = expanduser(
        join(os.path.dirname(__file__), '..', 'data', args.mesh))

    run(order=order,
        meshfile=meshfile,
        use_pardiso=args.use_pardiso,
        use_full_assembly=args.full_assembly,
        refinement_levels=args.refinement_levels)
