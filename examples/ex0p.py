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
import time
import pytest

import mfem.par as mfem
from mpi4py import MPI

num_procs = MPI.COMM_WORLD.size
myid = MPI.COMM_WORLD.rank
smyid = '{:0>6d}'.format(myid)

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
_MESH_DIR = expanduser(join(_EXAMPLES_DIR, '..', 'data'))


def _mesh(name):
    """Return the absolute path to a mesh file in the data directory."""
    return join(_MESH_DIR, name)


def get_full_assembly_level():
    enum_type = getattr(mfem, 'AssemblyLevel', None)
    if enum_type is not None and hasattr(enum_type, 'FULL'):
        return enum_type.FULL
    if hasattr(mfem, 'AssemblyLevel_FULL'):
        return mfem.AssemblyLevel_FULL
    raise AttributeError('FULL assembly level is not available in this PyMFEM build.')

def _is_complex_mumps_available():
    """Return True only when both the Python binding and libzmumps exist."""
    try:
        from mfem._par.mumps import ComplexMUMPSSolver  # noqa: F401
        return True
    except (ImportError, AttributeError):
        return False
    

def _assemble_complex_system(meshfile, order=1, epsilon=1e-3):
    """
    Assemble a complex Helmholtz-like system and return
    (op_complex, B_r, B_i, fespace, a_r) where op_complex is a
    ComplexHypreParMatrix, B_r/B_i are real/imaginary RHS vectors and
    a_r is the real bilinear form (needed for RecoverFEMSolution).
    """
    import mfem.par as mfem
    from mpi4py import MPI
    from mfem._par.complex_operator import ComplexHypreParMatrix

    serial_mesh = mfem.Mesh(meshfile)
    mesh = mfem.ParMesh(MPI.COMM_WORLD, serial_mesh)

    fec = mfem.H1_FECollection(order, mesh.Dimension())
    fespace = mfem.ParFiniteElementSpace(mesh, fec)

    boundary_dofs = mfem.intArray()
    fespace.GetBoundaryTrueDofs(boundary_dofs)

    one = mfem.ConstantCoefficient(1.0)
    eps = mfem.ConstantCoefficient(epsilon)

    # Real part: -Δ
    a_r = mfem.ParBilinearForm(fespace)
    a_r.AddDomainIntegrator(mfem.DiffusionIntegrator(one))
    a_r.Assemble()

    # Imaginary part: epsilon * M  (small mass matrix perturbation)
    a_i = mfem.ParBilinearForm(fespace)
    a_i.AddDomainIntegrator(mfem.MassIntegrator(eps))
    a_i.Assemble()

    # RHS
    b = mfem.ParLinearForm(fespace)
    b.AddDomainIntegrator(mfem.DomainLFIntegrator(one))
    b.Assemble()

    x_r = mfem.ParGridFunction(fespace)
    x_r.Assign(0.0)
    x_i = mfem.ParGridFunction(fespace)
    x_i.Assign(0.0)

    # Form real and imaginary parallel system matrices
    A_r = mfem.HypreParMatrix()
    A_i = mfem.HypreParMatrix()
    B_r = mfem.Vector()
    B_i = mfem.Vector()
    X_r = mfem.Vector()
    X_i = mfem.Vector()

    a_r.FormLinearSystem(boundary_dofs, x_r, b, A_r, X_r, B_r)
    a_i.FormLinearSystem(boundary_dofs, x_i, b, A_i, X_i, B_i)

    # Wrap into a ComplexHypreParMatrix (BLOCK_SYMMETRIC convention)
    from mfem._par.complex_operator import ComplexOperator
    op_complex = ComplexHypreParMatrix(
        A_r, A_i,
        False, False,  # do not take ownership
        ComplexOperator.BLOCK_SYMMETRIC)

    return op_complex, B_r, B_i, X_r, X_i, fespace, a_r, b, x_r


def test_complex_mumps_available():
    """Skip the remaining complex tests gracefully when zmumps is absent."""
    if not _is_complex_mumps_available():
        pytest.skip('ComplexMUMPSSolver not available (zmumps/cmumps missing)')


@pytest.mark.skipif(not _is_complex_mumps_available(),
                    reason='ComplexMUMPSSolver not available')
def test_complex_mumps_solves():
    """ComplexMUMPSSolver produces a finite solution on star.mesh."""
    import mfem.par as mfem
    from mpi4py import MPI
    from mfem._par.mumps import ComplexMUMPSSolver

    (op, B_r, B_i, X_r, X_i,
     fespace, a_r, b, x_r) = _assemble_complex_system(_mesh('star.mesh'))

    solver = ComplexMUMPSSolver(MPI.COMM_WORLD)
    solver.SetMatrixSymType(ComplexMUMPSSolver.UNSYMMETRIC)
    solver.SetPrintLevel(0)
    solver.SetOperator(op)

    solver.Mult(B_r, B_i, X_r, X_i)

    # Recover and check the solution is finite (no NaN/Inf).
    import numpy as np
    xr_arr = np.array(X_r.GetDataArray())
    xi_arr = np.array(X_i.GetDataArray())
    assert np.all(np.isfinite(xr_arr)), "Real solution part contains non-finite values"
    assert np.all(np.isfinite(xi_arr)), "Imaginary solution part contains non-finite values"


@pytest.mark.skipif(not _is_complex_mumps_available(),
                    reason='ComplexMUMPSSolver not available')
def test_complex_mumps_matches_real_solve():
    """With epsilon→0, real part of complex solution must match pure-real solve."""
    import mfem.par as mfem
    from mpi4py import MPI
    from mfem._par.mumps import MUMPSSolver, ComplexMUMPSSolver
    import numpy as np

    epsilon = 1e-6
    meshfile = _mesh('star.mesh')

    # ---- pure-real reference solve ----------------------------------------
    serial_mesh = mfem.Mesh(meshfile)
    mesh = mfem.ParMesh(MPI.COMM_WORLD, serial_mesh)
    fec = mfem.H1_FECollection(1, mesh.Dimension())
    fespace = mfem.ParFiniteElementSpace(mesh, fec)
    bdofs = mfem.intArray()
    fespace.GetBoundaryTrueDofs(bdofs)
    one = mfem.ConstantCoefficient(1.0)
    a_ref = mfem.ParBilinearForm(fespace)
    a_ref.AddDomainIntegrator(mfem.DiffusionIntegrator(one))
    a_ref.Assemble()
    b_ref = mfem.ParLinearForm(fespace)
    b_ref.AddDomainIntegrator(mfem.DomainLFIntegrator(one))
    b_ref.Assemble()
    x_ref = mfem.ParGridFunction(fespace)
    x_ref.Assign(0.0)
    A_ref = mfem.HypreParMatrix()
    B_ref = mfem.Vector()
    X_ref = mfem.Vector()
    a_ref.FormLinearSystem(bdofs, x_ref, b_ref, A_ref, X_ref, B_ref)

    mumps_ref = MUMPSSolver(MPI.COMM_WORLD)
    mumps_ref.SetMatrixSymType(MUMPSSolver.SYMMETRIC_POSITIVE_DEFINITE)
    mumps_ref.SetPrintLevel(0)
    mumps_ref.SetOperator(A_ref)
    mumps_ref.Mult(B_ref, X_ref)
    x_real_ref = np.array(X_ref.GetDataArray()).copy()

    # ---- complex solve with tiny imaginary perturbation -------------------
    (op, B_r, B_i, X_r, X_i,
     fespace2, a_r2, b2, xr2) = _assemble_complex_system(meshfile,
                                                          epsilon=epsilon)

    solver = ComplexMUMPSSolver(MPI.COMM_WORLD)
    solver.SetMatrixSymType(ComplexMUMPSSolver.UNSYMMETRIC)
    solver.SetPrintLevel(0)
    solver.SetOperator(op)
    solver.Mult(B_r, B_i, X_r, X_i)

    x_real_cplx = np.array(X_r.GetDataArray())

    # Real parts must agree to within O(epsilon).
    norm_ref = np.linalg.norm(x_real_ref)
    rel_err = np.linalg.norm(x_real_cplx - x_real_ref) / max(norm_ref, 1e-14)
    print(f"Relative error real(complex_solve) vs real_solve: {rel_err:.3e}")
    assert rel_err < 1e-4, (
        f"ComplexMUMPS real part deviates from real solve: rel_err={rel_err:.3e}")


@pytest.mark.skipif(not _is_complex_mumps_available(),
                    reason='ComplexMUMPSSolver not available')
def test_complex_mumps_fichera():
    """ComplexMUMPSSolver on fichera.mesh (3-D)."""
    import mfem.par as mfem
    from mpi4py import MPI
    from mfem._par.mumps import ComplexMUMPSSolver
    import numpy as np

    (op, B_r, B_i, X_r, X_i,
     fespace, a_r, b, x_r) = _assemble_complex_system(_mesh('fichera.mesh'))

    solver = ComplexMUMPSSolver(MPI.COMM_WORLD)
    solver.SetMatrixSymType(ComplexMUMPSSolver.UNSYMMETRIC)
    solver.SetPrintLevel(0)
    solver.SetOperator(op)
    solver.Mult(B_r, B_i, X_r, X_i)

    xr_arr = np.array(X_r.GetDataArray())
    assert np.all(np.isfinite(xr_arr))


@pytest.mark.skipif(not _is_complex_mumps_available(),
                    reason='ComplexMUMPSSolver not available')
def test_complex_mumps_elapsed_time():
    """ComplexMUMPSSolver must complete in under 60 s on star.mesh."""
    from mpi4py import MPI
    from mfem._par.mumps import ComplexMUMPSSolver

    (op, B_r, B_i, X_r, X_i,
     fespace, a_r, b, x_r) = _assemble_complex_system(_mesh('star.mesh'))

    solver = ComplexMUMPSSolver(MPI.COMM_WORLD)
    solver.SetMatrixSymType(ComplexMUMPSSolver.UNSYMMETRIC)
    solver.SetPrintLevel(0)
    solver.SetOperator(op)

    start = time.time()
    solver.Mult(B_r, B_i, X_r, X_i)
    elapsed = time.time() - start
    print(f"ComplexMUMPS solve elapsed: {elapsed:.4f}s")
    assert elapsed < 60, f"Solver too slow: {elapsed:.2f}s"

def run(order=1, meshfile='', visualization=False, use_cpardiso=False,
    use_mumps=False, use_complex_mumps=False, use_full_assembly=False,
    refinement_levels=1):
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

    if use_complex_mumps and not use_cpardiso and not use_mumps:
        # 9c. Solve with ComplexMUMPSSolver.
        # The real system -Delta u = 1 is solved as a complex system with a
        # tiny imaginary mass perturbation epsilon*M so the complex driver is
        # exercised.  The real part of the solution equals the pure-real result
        # to O(epsilon).
        try:
            from mfem._par.mumps import ComplexMUMPSSolver
            from mfem._par.complex_operator import ComplexHypreParMatrix, ComplexOperator
            epsilon = 1e-6
            eps_coeff = mfem.ConstantCoefficient(epsilon)
            a_i = mfem.ParBilinearForm(fespace)
            a_i.AddDomainIntegrator(mfem.MassIntegrator(eps_coeff))
            a_i.Assemble()
            x_i_gf = mfem.ParGridFunction(fespace)
            x_i_gf.Assign(0.0)
            A_i = mfem.HypreParMatrix()
            B_i = mfem.Vector()
            X_i = mfem.Vector()
            a_i.FormLinearSystem(boundary_dofs, x_i_gf, b, A_i, X_i, B_i)
            op_c = ComplexHypreParMatrix(
                A, A_i, False, False, ComplexOperator.BLOCK_SYMMETRIC)
            start_time = MPI.Wtime()
            csolver = ComplexMUMPSSolver(MPI.COMM_WORLD)
            csolver.SetMatrixSymType(ComplexMUMPSSolver.UNSYMMETRIC)
            csolver.SetPrintLevel(1 if myid == 0 else 0)
            csolver.SetOperator(op_c)
            if myid == 0:
                print('ComplexMUMPS is available, running with ComplexMUMPS direct solver')
            csolver.Mult(B, B_i, X, X_i)
            if myid == 0:
                print('ComplexMUMPS solve completed in',
                      MPI.Wtime() - start_time, 'seconds')
            use_complex_mumps = True
        except Exception as e:
            if myid == 0:
                print('ComplexMUMPS not available, falling back to PCG+BoomerAMG:', e)
            use_complex_mumps = False

    if not use_cpardiso and not use_mumps and not use_complex_mumps:
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
    parser.add_argument('-cmumps', '--use-complex-mumps',
                        action='store_true', default=False,
                        help='Use ComplexMUMPS direct solver (requires zmumps/cmumps).')
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
        use_complex_mumps=args.use_complex_mumps,
        use_full_assembly=args.full_assembly,
        refinement_levels=args.refinement_levels)
