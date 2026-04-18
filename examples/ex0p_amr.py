'''
   MFEM example 0p - Adaptive Mesh Refinement (AMR)

   This example demonstrates adaptive h-refinement for the Laplace problem
   using the Zienkiewicz-Zhu error estimator.

   How to run:
      mpirun -n <np> python ex0p_amr.py [options]

   Example of arguments:
      mpirun --allow-run-as-root -n 4 python ex0p_amr.py -m star.mesh
      mpirun --allow-run-as-root -n 4 python ex0p_amr.py -m fichera.mesh -o 2
      mpirun --allow-run-as-root -n 4 python ex0p_amr.py -vis
      mpirun --allow-run-as-root -n 4 python ex0p_amr.py -et 0.3 -ml 4
      mpirun --allow-run-as-root -n 4 python ex0p_amr.py -o 2 -et 0.5 -ml 3

   Description: Solves the Laplace problem -Delta u = f with zero Dirichlet
                boundary conditions using adaptive mesh refinement, where f is
                a sharp Gaussian peak. Elements with high error are automatically
                refined based on the Zienkiewicz-Zhu error estimator.
'''
import os
from os.path import expanduser, join
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

import mfem.par as mfem
from mpi4py import MPI
from math import pi, exp

num_procs = MPI.COMM_WORLD.size
myid = MPI.COMM_WORLD.rank


class PeakRHS(mfem.PyCoefficient):
    """RHS with a sharp peak to trigger adaptive refinement"""
    def __init__(self, dim):
        super(PeakRHS, self).__init__()
        self.dim = dim
        
    def Eval(self, T, ip):
        x = T.Transform(ip)
        # Create a sharp peak at the center
        r2 = 0.0
        for i in range(self.dim):
            r2 += (x[i] - 0.5)**2
        # Sharp Gaussian peak with width sigma = 0.1
        sigma = 0.1
        return 1000.0 * exp(-r2 / (2 * sigma**2))


def run(order=1, meshfile='', max_dofs=100000, max_iterations=10,
        visualization=False, error_threshold=0.7, plot_convergence=True,
        max_elem_refinement=None):
    '''
    Run ex0p with adaptive mesh refinement
    '''
    
    if myid == 0:
        print('='*70)
        print('Adaptive Mesh Refinement for Laplace Problem')
        print('Problem: -Delta u = f, where f = sharp Gaussian peak')
        print('='*70)
        print(f'Polynomial order: {order}')
        print(f'Mesh file: {os.path.basename(meshfile)}')
        print(f'Error threshold: {error_threshold}')
        print(f'Max DOFs: {max_dofs}')
        print('='*70)
    
    # 1. Read the mesh from the given mesh file
    serial_mesh = mfem.Mesh(meshfile)
    
    # 2. Ensure the mesh is in non-conforming mode for local refinement
    serial_mesh.EnsureNCMesh(True)
    
    # 3. Create parallel mesh
    mesh = mfem.ParMesh(MPI.COMM_WORLD, serial_mesh)
    dim = mesh.Dimension()
    
    if myid == 0:
        print(f'Mesh dimension: {dim}')
        print(f'Initial number of elements: {mesh.GetNE()}')
    
    # 4. Define a finite element space
    fec = mfem.H1_FECollection(order, dim)
    fespace = mfem.ParFiniteElementSpace(mesh, fec)
    
    # 5. Extract the list of boundary DOFs
    boundary_dofs = mfem.intArray()
    if mesh.bdr_attributes.Size() > 0:
        ess_bdr = mfem.intArray(mesh.bdr_attributes.Max())
        ess_bdr.Assign(1)
        fespace.GetEssentialTrueDofs(ess_bdr, boundary_dofs)
    
    # 6. Set up the bilinear and linear forms (not assembled yet)
    one = mfem.ConstantCoefficient(1.0)
    
    a = mfem.ParBilinearForm(fespace)
    integ = mfem.DiffusionIntegrator(one)
    a.AddDomainIntegrator(integ)
    
    # Use sharp peak RHS to trigger adaptive refinement
    peak_rhs = PeakRHS(dim)
    b = mfem.ParLinearForm(fespace)
    b.AddDomainIntegrator(mfem.DomainLFIntegrator(peak_rhs))
    
    # 7. Initialize solution
    x = mfem.ParGridFunction(fespace)
    x.Assign(0.0)
    
    # 8. Set up the Zienkiewicz-Zhu error estimator
    flux_fec = mfem.L2_FECollection(order, dim)
    flux_fes = mfem.ParFiniteElementSpace(mesh, flux_fec, dim)
    
    smooth_flux_fec = mfem.H1_FECollection(order, dim)
    smooth_flux_fes = mfem.ParFiniteElementSpace(mesh, smooth_flux_fec, dim)
    
    estimator = mfem.L2ZienkiewiczZhuEstimator(integ, x, flux_fes,
                                               smooth_flux_fes)
    
    # 9. Set up the mesh refiner with threshold strategy
    refiner = mfem.ThresholdRefiner(estimator)
    refiner.SetTotalErrorFraction(error_threshold)
    
    # Set maximum refinement level (controls refinement depth per element)
    if max_elem_refinement is not None:
        # This limits how many times a single element can be subdivided
        # For example, nc_limit=3 means an element can be refined max 3 times
        # (becoming 8 elements in 3D or 4 elements in 2D per refinement)
        refiner.SetNCLimit(max_elem_refinement)
    
    # Set more aggressive refinement to make error decrease faster
    # Refine elements with error > threshold * max_error
    # Lower values = more elements refined per iteration = faster convergence
    if myid == 0:
        print(f'\nRefinement strategy:')
        print(f'  Threshold fraction = {error_threshold}')
        print('  (Lower value = more aggressive refinement)')
        print('  Recommended: 0.3-0.5 for faster convergence')
        if max_elem_refinement is not None:
            print(f'  Max refinement depth = {max_elem_refinement} levels')
            print(f'  (Each element can be subdivided max {max_elem_refinement} times)')
        else:
            print('  Max refinement depth = unlimited')
    
    # 10. Initialize visualization
    if visualization:
        sol_sock = mfem.socketstream("localhost", 19916)
        if not sol_sock.good():
            if myid == 0:
                print('Unable to connect to GLVis server. Disabling visualization.')
            visualization = False
    
    # 10b. Set up ParaView data collection for export
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    if myid == 0:
        os.makedirs(results_dir, exist_ok=True)
    MPI.COMM_WORLD.Barrier()  # Wait for directory creation
    
    paraview_dc = mfem.ParaViewDataCollection('amr_solution', mesh)
    paraview_dc.SetPrefixPath(results_dir)
    paraview_dc.SetLevelsOfDetail(order)
    paraview_dc.SetDataFormat(mfem.VTKFormat_BINARY)
    paraview_dc.SetHighOrderOutput(True)
    paraview_dc.SetCycle(0)  # Iteration counter
    paraview_dc.SetTime(0.0)  # Time value
    paraview_dc.RegisterField("solution", x)
    
    if myid == 0:
        print(f'\nParaView output directory: {results_dir}')
        print('ParaView files will be saved after each AMR iteration')
    
    # Storage for convergence history
    dofs_history = []
    error_history = []
    elements_history = []
    time_history = []
    
    # 11. Main AMR loop
    total_start_time = time.time()
    for it in range(max_iterations):
        iter_start_time = time.time()
        
        global_dofs = fespace.GlobalTrueVSize()
        global_elements = mesh.GetGlobalNE()
        
        if myid == 0:
            print(f'\n--- AMR Iteration {it} ---')
            print(f'Number of DOFs: {global_dofs}')
            print(f'Number of elements: {global_elements}')
        
        # Update boundary DOFs
        boundary_dofs = mfem.intArray()
        if mesh.bdr_attributes.Size() > 0:
            ess_bdr = mfem.intArray(mesh.bdr_attributes.Max())
            ess_bdr.Assign(1)
            fespace.GetEssentialTrueDofs(ess_bdr, boundary_dofs)
        
        # Assemble the system
        b.Assemble()
        a.Assemble()
        
        # Form linear system
        A = mfem.HypreParMatrix()
        B = mfem.Vector()
        X = mfem.Vector()
        a.FormLinearSystem(boundary_dofs, x, b, A, X, B)
        
        # Solve using PCG with AMG preconditioner
        M = mfem.HypreBoomerAMG(A)
        M.SetPrintLevel(0)
        
        cg = mfem.CGSolver(MPI.COMM_WORLD)
        cg.SetRelTol(1e-12)
        cg.SetMaxIter(2000)
        cg.SetPrintLevel(0)
        cg.SetPreconditioner(M)
        cg.SetOperator(A)
        cg.Mult(B, X)
        
        # Recover solution
        a.RecoverFEMSolution(X, b, x)
        
        # Visualize solution
      #   if visualization and sol_sock.good():
      #       sol_sock.send_text("parallel " + str(num_procs) + " " + str(myid))
      #       sol_sock.precision(8)
      #       sol_sock.send_solution(mesh, x)
      #       sol_sock.send_text("window_title 'AMR Iteration " + str(it) + "'")
      #       sol_sock.flush()
        
        # Check stopping criteria
        if global_dofs > max_dofs:
            if myid == 0:
                print(f'\nReached maximum DOFs ({max_dofs}). Stopping.')
            break
        
        # Refine the mesh based on error estimates (estimator is called internally)
        refiner.Apply(mesh)
        
        # Get error estimate after refinement
        error_estimate = estimator.GetTotalError()
        
        # Record elapsed time for this iteration
        iter_elapsed_time = time.time() - iter_start_time
        
        if myid == 0:
            print(f'Estimated L2 error: {error_estimate:.6e}')
            print(f'Iteration time: {iter_elapsed_time:.3f} seconds')
        
        # Store history
        dofs_history.append(global_dofs)
        error_history.append(error_estimate)
        elements_history.append(global_elements)
        time_history.append(iter_elapsed_time)
        
        if refiner.Stop():
            if myid == 0:
                print('\nRefinement stopping criterion met.')
            break
        
        # Update finite element space and solution
        fespace.Update()
        x.Update()
        
        # Rebalance parallel mesh (only for non-conforming meshes)
        if mesh.Nonconforming():
            mesh.Rebalance()
            # Update again after rebalancing (redistributes GridFunctions)
            fespace.Update()
            x.Update()
        
        # Update the bilinear and linear forms
        a.Update()
        b.Update()
        
        # Save ParaView output for this iteration
        paraview_dc.SetCycle(it)
        paraview_dc.SetTime(float(it))
        paraview_dc.Save()
    
    # 12. Send final solution to GLVis
    if visualization and sol_sock.good():
        sol_sock.send_text("parallel " + str(num_procs) + " " + str(myid))
        sol_sock.precision(8)
        sol_sock.send_solution(mesh, x)
        sol_sock.send_text("window_title 'Final AMR Solution'")
        sol_sock.flush()
        if myid == 0:
            print('\nFinal solution sent to GLVis')
    
    # 13. Print final statistics
    total_elapsed_time = time.time() - total_start_time
    
    if myid == 0:
        print('\n' + '='*70)
        print('AMR Summary')
        print('='*70)
        print(f'{"Iteration":<10} {"DOFs":<10} {"Elements":<10} {"Error Est.":<13} {"Time(s)":<10} {"Cum.Time":<10}')
        print('-'*70)
        cumulative_time = 0.0
        for i, (dofs, elems, err, t) in enumerate(zip(dofs_history, elements_history, error_history, time_history)):
            cumulative_time += t
            print(f'{i:<10} {dofs:<10} {elems:<10} {err:<13.6e} {t:<10.3f} {cumulative_time:<10.3f}')
        print('-'*70)
        print(f'Total time: {total_elapsed_time:.3f} seconds')
        print('='*70)
        
        # Compute error reduction
        if len(error_history) > 1:
            error_reduction = error_history[0] / error_history[-1]
            dof_increase = dofs_history[-1] / dofs_history[0]
            print(f'Error reduction: {error_reduction:.2f}x')
            print(f'DOF increase: {dof_increase:.2f}x')
            print(f'Efficiency: {error_reduction/dof_increase:.3f} (error reduction per DOF increase)')
            print('-'*70)
            print('Tips for faster convergence:')
            print('  1. Use higher polynomial order: -o 2 or -o 3')
            print('  2. Use lower error threshold: -et 0.3 or -et 0.5')
            print('  3. Set max refinement depth: -ml 3 or -ml 4')
            print('     (prevents unlimited refinement in small regions)')
            print('  4. Combine these options for best results!')
        print('='*70)
    
    # 14. Save final mesh and solution
    smyid = '{:0>6d}'.format(myid)
    
    # Save traditional MFEM format
    sol_path = os.path.join(results_dir, 'sol_amr.' + smyid)
    mesh_path = os.path.join(results_dir, 'mesh_amr.' + smyid)
    x.Save(sol_path)
    mesh.Print(mesh_path)
    
    # Save final ParaView output
    paraview_dc.SetCycle(len(dofs_history) - 1)
    paraview_dc.SetTime(float(len(dofs_history) - 1))
    paraview_dc.Save()
    
    if myid == 0:
        print(f'\nSolution saved to: {results_dir}/')
        print(f'  - MFEM format: sol_amr.* and mesh_amr.*')
        print(f'    View with: glvis -np {num_procs} -m {mesh_path} -g {sol_path}')
        print(f'  - ParaView format: amr_solution/amr_solution.pvd')
        print(f'    Open {results_dir}/amr_solution/amr_solution.pvd in ParaView')
    
    # 15. Plot convergence history
    if plot_convergence and myid == 0:
        plot_amr_convergence(dofs_history, error_history, elements_history, time_history, results_dir)


def plot_amr_convergence(dofs_history, error_history, elements_history, time_history, results_dir='.'):
    """Plot AMR convergence history"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    iterations = range(len(dofs_history))
    cumulative_time = np.cumsum(time_history)
    
    # Plot 1: Error vs DOFs
    ax1.loglog(dofs_history, error_history, 'o-', linewidth=2, 
               markersize=8, label='AMR')
    ax1.set_xlabel('Number of DOFs', fontsize=12)
    ax1.set_ylabel('Estimated L2 Error', fontsize=12)
    ax1.set_title('Error Convergence', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Add convergence rate if we have enough points
    if len(dofs_history) > 2:
        # Compute rate between first and last iteration
        rate = np.log(error_history[-1]/error_history[0]) / \
               np.log(dofs_history[-1]/dofs_history[0])
        ax1.text(0.05, 0.05, f'Convergence Rate: {rate:.3f}',
                transform=ax1.transAxes, fontsize=11,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Elements and DOFs vs Iteration
    ax2.plot(iterations, elements_history, 's-', linewidth=2, 
             markersize=6, label='Elements')
    ax2.plot(iterations, dofs_history, 'o-', linewidth=2, 
             markersize=6, label='DOFs')
    ax2.set_xlabel('AMR Iteration', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Mesh Growth', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_yscale('log')
    
    # Plot 3: Time per iteration
    ax3.bar(iterations, time_history, alpha=0.7, color='steelblue')
    ax3.set_xlabel('AMR Iteration', fontsize=12)
    ax3.set_ylabel('Time (seconds)', fontsize=12)
    ax3.set_title('Time per Iteration', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Error vs Time
    ax4.semilogy(cumulative_time, error_history, 'o-', linewidth=2, 
                 markersize=8, color='crimson')
    ax4.set_xlabel('Cumulative Time (seconds)', fontsize=12)
    ax4.set_ylabel('Estimated L2 Error', fontsize=12)
    ax4.set_title('Error vs Time', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = os.path.join(results_dir, 'amr_convergence.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f'\nConvergence plot saved to: {output_file}')
    plt.close()


if __name__ == "__main__":
    from mfem.common.arg_parser import ArgParser

    parser = ArgParser(description='Ex0p - Adaptive Mesh Refinement')
    parser.add_argument('-m', '--mesh',
                        default='star.mesh',
                        action='store', type=str,
                        help='Mesh file to use.')
    parser.add_argument('-o', '--order',
                        action='store', default=1, type=int,
                        help="Finite element order (polynomial degree).")
    parser.add_argument('-md', '--max-dofs',
                        action='store', default=1000000, type=int,
                        help="Maximum number of degrees of freedom.")
    parser.add_argument('-mi', '--max-iterations',
                        action='store', default=10, type=int,
                        help="Maximum number of AMR iterations.")
    parser.add_argument('-et', '--error-threshold',
                        action='store', default=0.7, type=float,
                        help="Error threshold for refinement (0.0-1.0). Lower = more aggressive. "
                             "Typical values: 0.7 (default, conservative), 0.5 (moderate), "
                             "0.3 (aggressive, faster convergence).")
    parser.add_argument('-ml', '--max-level',
                        action='store', default=None, type=int,
                        help="Maximum refinement level/depth per element. "
                             "Limits how many times a single element can be subdivided. "
                             "For example, -ml 3 means max 3 levels of refinement per element. "
                             "Default: unlimited.")
    parser.add_argument('-vis', '--visualization',
                        action='store_true',
                        help='Enable GLVis visualization')
    parser.add_argument('-no-plot', '--no-plot',
                        action='store_true',
                        help='Disable convergence plotting')

    args = parser.parse_args()
    
    if myid == 0:
        parser.print_options(args)

    meshfile = expanduser(join(os.path.dirname(__file__), '..', 'data', args.mesh))

    run(order=args.order,
        meshfile=meshfile,
        max_dofs=args.max_dofs,
        max_iterations=args.max_iterations,
        visualization=args.visualization,
        error_threshold=args.error_threshold,
        plot_convergence=not args.no_plot,
        max_elem_refinement=args.max_level)
