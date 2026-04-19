'''
   PyMFEM Example: Indefinite Maxwell Equation with Dipole Source

   Solves the indefinite Maxwell equation:
      curl curl E - k^2 E = -iω μ J
   
   where:
      E = electric field (complex-valued)
      k = wavenumber (ω/c)
      J = dipole current source
      PEC boundary conditions: E × n = 0

   This example demonstrates:
      - H(curl) finite element spaces (Nedelec elements)
      - Complex-valued system for time-harmonic Maxwell
      - Dipole source implementation
      - PEC (Perfect Electric Conductor) boundary conditions
      - AMS preconditioner for curl-curl systems
      - Optional MUMPS direct solver for better accuracy
      - ParaView/VTK export for visualization

   How to run:
      mpirun -n <np> python ex_maxwell_dipole.py [options]

   Example of arguments:
      mpirun --allow-run-as-root -n 4 python ex_maxwell_dipole.py -m inline-hex.mesh
      mpirun --allow-run-as-root -n 4 python ex_maxwell_dipole.py -m fichera.mesh -o 2
      mpirun --allow-run-as-root -n 4 python ex_maxwell_dipole.py -vis -k 2.0
      mpirun --allow-run-as-root -n 4 python ex_maxwell_dipole.py -mumps -pv
      mpirun --allow-run-as-root -n 4 python ex_maxwell_dipole.py -o 2 -k 1.5 -pv -vis
'''
import os
from os.path import expanduser, join
import numpy as np
from numpy import pi, sqrt, exp, sin, cos

import mfem.par as mfem
from mpi4py import MPI

num_procs = MPI.COMM_WORLD.size
myid = MPI.COMM_WORLD.rank


class DipoleSource(mfem.VectorPyCoefficient):
    """
    Point dipole source at the center of the domain.
    J = p * delta(x - x0)
    where p is the dipole moment vector.
    
    We approximate the delta function with a smooth Gaussian:
    delta(x - x0) ≈ (1/(2π σ²))^(3/2) * exp(-|x-x0|²/(2σ²))
    """
    def __init__(self, dim, dipole_pos, dipole_moment, width=0.1):
        super(DipoleSource, self).__init__(dim)
        self.dim = dim
        self.dipole_pos = np.array(dipole_pos)
        self.dipole_moment = np.array(dipole_moment)
        self.width = width  # Width of Gaussian approximation
        self.amplitude = 1.0 / (2 * pi * width**2)**(dim/2.0)
        
    def Eval(self, v, T, ip):
        x = T.Transform(ip)
        
        # Distance from dipole position
        r_squared = 0.0
        for i in range(self.dim):
            r_squared += (x[i] - self.dipole_pos[i])**2
        
        # Gaussian envelope
        envelope = self.amplitude * exp(-r_squared / (2 * self.width**2))
        
        # Apply dipole moment
        for i in range(self.dim):
            v[i] = envelope * self.dipole_moment[i]


def run(order=1, 
        meshfile='',
        rs=2,
        rp=0,
        wavenumber=1.0,
        visualization=False,
        freq=1.0,
        use_mumps=False,
        paraview_output=False):
    '''
    Run the indefinite Maxwell solver with dipole source
    '''
    
    if myid == 0:
        print('='*70)
        print('Indefinite Maxwell Equation - Dipole Source')
        print('='*70)
        print(f'Polynomial order: {order}')
        print(f'Mesh file: {os.path.basename(meshfile)}')
        print(f'Wavenumber k: {wavenumber}')
        print(f'Frequency: {freq} (for complex formulation)')
        print('='*70)
    
    # 1. Read the mesh
    mesh = mfem.Mesh(meshfile, 1, 1)
    dim = mesh.Dimension()
    
    # 2. Refine the mesh
    for lev in range(rs):
        mesh.UniformRefinement()
    
    # 3. Define parallel mesh
    pmesh = mfem.ParMesh(MPI.COMM_WORLD, mesh)
    del mesh
    for lev in range(rp):
        pmesh.UniformRefinement()
    
    if myid == 0:
        print(f'Mesh dimension: {dim}')
        print(f'Number of elements: {pmesh.GetNE()}')
    
    # 4. Define H(curl) finite element space (Nedelec elements)
    if dim == 2:
        fec = mfem.ND_FECollection(order, dim)
    else:
        fec = mfem.ND_FECollection(order, dim)
    
    fespace = mfem.ParFiniteElementSpace(pmesh, fec)
    size = fespace.GlobalTrueVSize()
    
    if myid == 0:
        print(f'Number of H(curl) unknowns: {size}')
    
    # 5. Define PEC boundary conditions (homogeneous Dirichlet on all boundaries)
    #    This enforces E × n = 0 on the boundary
    ess_tdof_list = mfem.intArray()
    ess_bdr = mfem.intArray([1] * pmesh.bdr_attributes.Max())
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list)
    
    if myid == 0:
        print(f'Number of boundary DOFs: {ess_tdof_list.Size()}')
    
    # 6. Set up dipole source at the center of the domain
    # Get bounding box of the mesh
    bbox_min, bbox_max = pmesh.GetBoundingBox(dim)
    bbox_min = np.array(bbox_min)
    bbox_max = np.array(bbox_max)
    
    # Place dipole at center
    dipole_pos = (bbox_min + bbox_max) / 2.0
    
    # Dipole moment (z-directed for 3D, y-directed for 2D)
    dipole_moment = np.zeros(dim)
    if dim == 3:
        dipole_moment[2] = 1.0  # z-directed dipole
    else:
        dipole_moment[1] = 1.0  # y-directed dipole
    
    if myid == 0:
        print(f'Dipole position: {dipole_pos}')
        print(f'Dipole moment: {dipole_moment}')
    
    # Create dipole source coefficient
    dipole_width = 0.05 * np.linalg.norm(bbox_max - bbox_min)
    J = DipoleSource(dim, dipole_pos, dipole_moment, dipole_width)
    
    # 7. Set up the anti-linear form for the RHS: (J, E)
    #    For time-harmonic: -iωμ J
    omega_mu = 2.0 * pi * freq * 1.0  # ω * μ (μ=1 here)
    
    b_real = mfem.ParLinearForm(fespace)
    b_real.AddDomainIntegrator(mfem.VectorFEDomainLFIntegrator(J))
    b_real.Assemble()
    
    # 8. Define solution vector
    E_real = mfem.ParGridFunction(fespace)
    E_real.Assign(0.0)
    
    E_imag = mfem.ParGridFunction(fespace)
    E_imag.Assign(0.0)
    
    # 9. Set up the bilinear form: curl curl E - k^2 E
    #    This creates an indefinite system
    
    # Curl-curl term
    muinv = mfem.ConstantCoefficient(1.0)
    a = mfem.ParBilinearForm(fespace)
    a.AddDomainIntegrator(mfem.CurlCurlIntegrator(muinv))
    
    # Mass term: -k^2
    k_squared = mfem.ConstantCoefficient(-wavenumber**2)
    a.AddDomainIntegrator(mfem.VectorFEMassIntegrator(k_squared))
    
    # Add small imaginary part for stability (lossy medium)
    # This makes the system better conditioned
    loss = mfem.ConstantCoefficient(0.01 * wavenumber**2)
    a.AddDomainIntegrator(mfem.VectorFEMassIntegrator(loss))
    
    if myid == 0:
        print('\nAssembling system...')
    
    a.Assemble()
    
    # 10. Form the linear system
    A = mfem.OperatorPtr()
    B = mfem.Vector()
    X = mfem.Vector()
    
    a.FormLinearSystem(ess_tdof_list, E_real, b_real, A, X, B)
    
    if myid == 0:
        print(f'System size: {A.Height()}')
        print('Solving linear system...')
        if use_mumps:
            print('Using MUMPS direct solver')
        else:
            print('Using GMRES with AMS preconditioner')
    
    # 11. Solve the linear system
    AA = A.AsHypreParMatrix()
    
    if use_mumps:
        # Use MUMPS direct solver for indefinite systems
        try:
            from mfem._par.mumps import MUMPSSolver
            mumps = MUMPSSolver(MPI.COMM_WORLD)
            mumps.SetMatrixSymType(MUMPSSolver.UNSYMMETRIC)
            mumps.SetPrintLevel(1 if myid == 0 else 0)
            mumps.SetOperator(AA)
            mumps.Mult(B, X)
            
            if myid == 0:
                print('MUMPS solver completed')
        except:
            if myid == 0:
                print('MUMPS not available, falling back to GMRES')
            use_mumps = False
    
    if not use_mumps:
        # Use GMRES with AMS preconditioner
        # AMS (Auxiliary-space Maxwell Solver) is designed for curl-curl systems
        ams = mfem.HypreAMS(AA, fespace)
        ams.SetPrintLevel(0)
        
        # Use GMRES for the indefinite system
        gmres = mfem.GMRESSolver(MPI.COMM_WORLD)
        gmres.SetPrintLevel(1)
        gmres.SetKDim(100)
        gmres.SetMaxIter(500)
        gmres.SetRelTol(1e-8)
        gmres.SetAbsTol(1e-10)
        gmres.SetOperator(AA)
        gmres.SetPreconditioner(ams)
        
        gmres.Mult(B, X)
        
        if gmres.GetConverged():
            if myid == 0:
                print(f'GMRES converged in {gmres.GetNumIterations()} iterations')
        else:
            if myid == 0:
                print('GMRES did not converge')
    
    # 12. Recover the solution
    a.RecoverFEMSolution(X, b_real, E_real)
    
    # 13. Compute field magnitude for visualization
    E_mag = mfem.ParGridFunction(fespace)
    for i in range(E_real.Size()):
        E_mag[i] = abs(E_real[i])
    
    # 14. Save the solution
    smyid = '{:0>6d}'.format(myid)
    pmesh.Print('mesh.' + smyid, 8)
    E_real.Save('sol.' + smyid, 8)
    
    if myid == 0:
        print(f'\nSolution saved to: mesh.* and sol.*')
        print(f'View with: glvis -np {num_procs} -m mesh -g sol')
    
    # 14b. Export to ParaView/VTK format
    if paraview_output:
        paraview_dc = mfem.ParaViewDataCollection("maxwell_dipole", pmesh)
        paraview_dc.SetPrefixPath("ParaView")
        paraview_dc.SetLevelsOfDetail(order)
        paraview_dc.SetDataFormat(mfem.VTKFormat_BINARY)
        paraview_dc.SetHighOrderOutput(True)
        paraview_dc.SetCycle(0)
        paraview_dc.SetTime(0.0)
        paraview_dc.RegisterField("E_real", E_real)
        paraview_dc.Save()
        
        if myid == 0:
            print(f'\nParaView output saved to: ParaView/maxwell_dipole/')
            print('Open ParaView and load: ParaView/maxwell_dipole/maxwell_dipole.pvd')
    
    # 15. Send to GLVis
    if visualization:
        sol_sock = mfem.socketstream("localhost", 19916)
        if sol_sock.good():
            sol_sock.send_text("parallel " + str(num_procs) + " " + str(myid))
            sol_sock.precision(8)
            sol_sock.send_solution(pmesh, E_real)
            sol_sock.send_text("window_title 'Electric Field (Real Part)'")
            sol_sock.flush()
            
            if myid == 0:
                print('\nSolution sent to GLVis')
        else:
            if myid == 0:
                print('Unable to connect to GLVis')
    
    # 16. Compute some field statistics
    E_norm = sqrt(mfem.InnerProduct(MPI.COMM_WORLD, E_real, E_real))
    
    if myid == 0:
        print('\n' + '='*70)
        print('Field Statistics:')
        print(f'||E|| (L2 norm): {E_norm:.6e}')
        print('='*70)


if __name__ == "__main__":
    from mfem.common.arg_parser import ArgParser

    parser = ArgParser(description='Indefinite Maxwell - Dipole Source')
    parser.add_argument('-m', '--mesh',
                        default='inline-hex.mesh',
                        action='store', type=str,
                        help='Mesh file to use.')
    parser.add_argument('-o', '--order',
                        action='store', default=1, type=int,
                        help="Finite element order (polynomial degree)")
    parser.add_argument('-rs', '--refine-serial',
                        action='store', default=2, type=int,
                        help="Number of times to refine the mesh in serial")
    parser.add_argument('-rp', '--refine-parallel',
                        action='store', default=0, type=int,
                        help="Number of times to refine the mesh in parallel")
    parser.add_argument('-k', '--wavenumber',
                        action='store', default=1.0, type=float,
                        help="Wavenumber k (ω/c)")
    parser.add_argument('-f', '--frequency',
                        action='store', default=1.0, type=float,
                        help="Frequency for time-harmonic formulation")
    parser.add_argument('-vis', '--visualization',
                        action='store_true',
                        help='Enable GLVis visualization')
    parser.add_argument('-mumps', '--use-mumps',
                        action='store_true',
                        help='Use MUMPS direct solver instead of GMRES')
    parser.add_argument('-pv', '--paraview',
                        action='store_true',
                        help='Export solution to ParaView/VTK format')

    args = parser.parse_args()
    
    if myid == 0:
        parser.print_options(args)

    meshfile = expanduser(join(os.path.dirname(__file__), '..', 'data', args.mesh))

    run(order=args.order,
        meshfile=meshfile,
        rs=args.refine_serial,
        rp=args.refine_parallel,
        wavenumber=args.wavenumber,
        visualization=args.visualization,
        freq=args.frequency,
        use_mumps=args.use_mumps,
        paraview_output=args.paraview)
