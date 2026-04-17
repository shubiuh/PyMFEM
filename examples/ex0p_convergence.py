'''
   MFEM example 0p - Convergence Study

   This script tests h-refinement convergence for the Laplace problem
   using a manufactured solution.

   How to run:
      mpirun -n <np> python ex0p_convergence.py

   Example:
      mpirun --allow-run-as-root -n 4 python ex0p_convergence.py

   Description: Solves -Delta u = f with manufactured solution to verify
                convergence rates under h-refinement.
'''
import os
from os.path import expanduser, join
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for parallel environments
import matplotlib.pyplot as plt

import mfem.par as mfem
from mpi4py import MPI
from math import pi, sqrt

num_procs = MPI.COMM_WORLD.size
myid = MPI.COMM_WORLD.rank


class ManufacturedSolution(mfem.PyCoefficient):
    """Exact solution: u = sin(pi*x)*sin(pi*y) for 2D
                    or u = sin(pi*x)*sin(pi*y)*sin(pi*z) for 3D"""
    def __init__(self, dim):
        super(ManufacturedSolution, self).__init__()
        self.dim = dim
        
    def Eval(self, T, ip):
        x = T.Transform(ip)
        if self.dim == 2:
            return np.sin(pi * x[0]) * np.sin(pi * x[1])
        else:  # dim == 3
            return np.sin(pi * x[0]) * np.sin(pi * x[1]) * np.sin(pi * x[2])


class RHSCoefficient(mfem.PyCoefficient):
    """RHS: f = -Delta u for the manufactured solution"""
    def __init__(self, dim):
        super(RHSCoefficient, self).__init__()
        self.dim = dim
        
    def Eval(self, T, ip):
        x = T.Transform(ip)
        if self.dim == 2:
            # -Delta(sin(pi*x)*sin(pi*y)) = 2*pi^2*sin(pi*x)*sin(pi*y)
            return 2.0 * pi**2 * np.sin(pi * x[0]) * np.sin(pi * x[1])
        else:  # dim == 3
            # -Delta(sin(pi*x)*sin(pi*y)*sin(pi*z)) = 3*pi^2*sin(pi*x)*sin(pi*y)*sin(pi*z)
            return 3.0 * pi**2 * np.sin(pi * x[0]) * np.sin(pi * x[1]) * np.sin(pi * x[2])


def solve_at_refinement(order, base_mesh, ref_level, exact_sol):
    """Solve the problem at a given refinement level"""
    
    # Create mesh copy and refine
    mesh = mfem.ParMesh(MPI.COMM_WORLD, base_mesh)
    for _ in range(ref_level):
        mesh.UniformRefinement()
    
    dim = mesh.Dimension()
    h = 1.0 / (2**ref_level * (base_mesh.GetNE()**(1.0/dim)))  # Approximate element size
    
    # Define finite element space
    fec = mfem.H1_FECollection(order, dim)
    fespace = mfem.ParFiniteElementSpace(mesh, fec)
    gtdof = fespace.GlobalTrueVSize()
    
    if myid == 0:
        print(f'  Refinement level {ref_level}: h = {h:.4e}, DOFs = {gtdof}')
    
    # Get boundary DOFs
    boundary_dofs = mfem.intArray()
    fespace.GetBoundaryTrueDofs(boundary_dofs)
    
    # Initialize solution
    x = mfem.ParGridFunction(fespace)
    x.Assign(0.0)
    
    # Set up RHS
    rhs_coef = RHSCoefficient(dim)
    b = mfem.ParLinearForm(fespace)
    b.AddDomainIntegrator(mfem.DomainLFIntegrator(rhs_coef))
    b.Assemble()
    
    # Set up bilinear form
    one = mfem.ConstantCoefficient(1.0)
    a = mfem.ParBilinearForm(fespace)
    a.AddDomainIntegrator(mfem.DiffusionIntegrator(one))
    a.Assemble()
    
    # Form linear system
    A = mfem.HypreParMatrix()
    B = mfem.Vector()
    X = mfem.Vector()
    a.FormLinearSystem(boundary_dofs, x, b, A, X, B)
    
    # Solve using PCG with AMG preconditioner
    M = mfem.HypreBoomerAMG(A)
    cg = mfem.CGSolver(MPI.COMM_WORLD)
    cg.SetRelTol(1e-12)
    cg.SetMaxIter(2000)
    cg.SetPrintLevel(-1)  # Quiet
    cg.SetPreconditioner(M)
    cg.SetOperator(A)
    cg.Mult(B, X)
    
    # Recover solution
    a.RecoverFEMSolution(X, b, x)
    
    # Compute L2 error
    error = x.ComputeL2Error(exact_sol)
    
    return h, error, gtdof


def run_convergence_study(order=1, meshfile='', max_ref_levels=5):
    """Run convergence study with multiple refinement levels"""
    
    if myid == 0:
        print('='*70)
        print(f'H-Refinement Convergence Study')
        print(f'Polynomial order: {order}')
        print(f'Mesh file: {meshfile}')
        print('='*70)
    
    # Load base mesh
    serial_mesh = mfem.Mesh(meshfile)
    dim = serial_mesh.Dimension()
    
    # Create manufactured solution
    exact_sol = ManufacturedSolution(dim)
    
    # Storage for results
    h_values = []
    errors = []
    dofs = []
    
    # Run convergence study
    if myid == 0:
        print('\nSolving at different refinement levels:')
    
    for ref_level in range(max_ref_levels):
        h, error, ndofs = solve_at_refinement(order, serial_mesh, ref_level, exact_sol)
        
        h_values.append(h)
        errors.append(error)
        dofs.append(ndofs)
    
    # Compute convergence rates
    if myid == 0:
        print('\n' + '='*70)
        print('Convergence Results:')
        print('='*70)
        print(f'{"Level":<8} {"h":<12} {"DOFs":<12} {"L2 Error":<15} {"Rate":<10}')
        print('-'*70)
        
        for i in range(len(h_values)):
            if i == 0:
                rate_str = "---"
            else:
                rate = np.log(errors[i]/errors[i-1]) / np.log(h_values[i]/h_values[i-1])
                rate_str = f"{rate:.3f}"
            
            print(f'{i:<8} {h_values[i]:<12.4e} {dofs[i]:<12} {errors[i]:<15.4e} {rate_str:<10}')
        
        # Expected rate
        expected_rate = order + 1
        print('-'*70)
        print(f'Expected convergence rate: O(h^{expected_rate}) = {expected_rate}')
        print('='*70)
        
        # Plot convergence
        plot_convergence(h_values, errors, order, meshfile)


def plot_convergence(h_values, errors, order, meshfile):
    """Plot L2 error vs mesh size"""
    
    h_array = np.array(h_values)
    error_array = np.array(errors)
    
    plt.figure(figsize=(10, 7))
    
    # Plot errors
    plt.loglog(h_array, error_array, 'o-', linewidth=2, markersize=8, label='Computed Error')
    
    # Plot reference lines
    expected_rate = order + 1
    ref_error = error_array[0] * (h_array / h_array[0])**expected_rate
    plt.loglog(h_array, ref_error, '--', linewidth=2, 
               label=f'O(h^{expected_rate}) Reference', alpha=0.7)
    
    plt.xlabel('Mesh Size (h)', fontsize=12)
    plt.ylabel('L2 Error', fontsize=12)
    plt.title(f'H-Refinement Convergence (Order p={order})\n{os.path.basename(meshfile)}', 
              fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add convergence rate annotation
    if len(h_values) > 1:
        rate = np.log(errors[-1]/errors[0]) / np.log(h_values[-1]/h_values[0])
        plt.text(0.05, 0.95, f'Measured Rate: {rate:.3f}',
                transform=plt.gca().transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    output_file = f'convergence_p{order}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f'\nConvergence plot saved to: {output_file}')
    plt.close()


if __name__ == "__main__":
    from mfem.common.arg_parser import ArgParser

    parser = ArgParser(description='Ex0p Convergence Study')
    parser.add_argument('-m', '--mesh',
                        default='inline-quad.mesh',
                        action='store', type=str,
                        help='Mesh file to use.')
    parser.add_argument('-o', '--order',
                        action='store', default=1, type=int,
                        help="Finite element polynomial order.")
    parser.add_argument('-r', '--refinements',
                        action='store', default=5, type=int,
                        help="Number of refinement levels to test.")

    args = parser.parse_args()
    
    if myid == 0:
        parser.print_options(args)

    meshfile = expanduser(join(os.path.dirname(__file__), '..', 'data', args.mesh))

    run_convergence_study(order=args.order,
                         meshfile=meshfile,
                         max_ref_levels=args.refinements)
