'''
   PyMFEM Example: AMR Maxwell Equation with Dipole Source and Conductivity

    Solves the time-harmonic H(curl) Maxwell problem:
        curl (mu^{-1} curl E) - omega^2 epsilon E + i omega sigma E = -i omega J_s

   where:
        E = complex electric field
        omega = angular frequency
        epsilon = permittivity
        k^2 = omega^2 epsilon (with mu = 1 in this example)
      sigma = conductivity coefficient
        J_s = impressed dipole current source
      PEC boundary conditions: E x n = 0

   This example demonstrates:
      - H(curl) finite element spaces (Nedelec elements)
        - Complex-valued time-harmonic Maxwell assembly
      - Parallel nonconforming adaptive mesh refinement (AMR)
      - Dipole source implementation
        - Conductivity term i omega sigma in the Maxwell operator
      - Element-wise conductivity assignment through element attributes
        - Block preconditioning for the complex curl-curl system
        - Optional MUMPS direct solver for the assembled block system

   The helper function `set_element_conductivities` assigns different
   conductivity values to selected original serial mesh elements. Internally,
   those elements are re-labeled with unique attributes so the material map is
   preserved after uniform refinement and AMR.

   How to run:
      mpirun -n <np> python ex_maxwell_dipole_amr.py [options]

    Example of arguments:
        mpirun --allow-run-as-root -n 4 python ex_maxwell_dipole_amr.py -m inline-hex.mesh
        mpirun --allow-run-as-root -n 4 python ex_maxwell_dipole_amr.py -f 0.5 -amr 4 -rf 0.20
        mpirun --allow-run-as-root -n 4 python ex_maxwell_dipole_amr.py -sigma 0.05
        mpirun --allow-run-as-root -n 4 python ex_maxwell_dipole_amr.py -es 0:1.0,5:4.0,8:8.0
        mpirun --allow-run-as-root -n 4 python ex_maxwell_dipole_amr.py -mumps -pv -vis
        mpirun --allow-run-as-root -n 4 python ex_maxwell_dipole_amr.py -cmumps
        mpirun --allow-run-as-root -n 4 python ex_maxwell_dipole_amr.py -pardiso
'''
import os
import sys
from os.path import expanduser, join

import numpy as np
from numpy import pi, sqrt

import mfem.par as mfem
from mpi4py import MPI

num_procs = MPI.COMM_WORLD.size
myid = MPI.COMM_WORLD.rank


class DipoleSource(mfem.VectorPyCoefficient):
    """
    Point dipole source descriptor for explicit H(curl) RHS assembly.

    For a source inside an element, the load is assembled from the local
    vector basis values at the source point. If the source lies on a mesh
    vertex, the contribution is split across the incident cells using a local
    corner-angle weight.
    """

    def __init__(self, dim, dipole_pos, dipole_moment, vertex_tol=1e-10):
        super(DipoleSource, self).__init__(dim)
        self.dim = dim
        self.dipole_pos = np.array(dipole_pos, dtype=float)
        self.dipole_moment = np.array(dipole_moment, dtype=float)
        self.vertex_tol = float(vertex_tol)

    def distance_squared(self, x):
        delta = np.array(x, dtype=float)[:self.dim] - self.dipole_pos
        return float(np.dot(delta, delta))

    def Eval(self, v, T, ip):
        for i in range(self.dim):
            v[i] = self.dipole_moment[i]


def _to_mfem_vector(values):
    vec = mfem.Vector(len(values))
    for i, value in enumerate(values):
        vec[i] = float(value)
    return vec


def _get_element_vertices(fespace, element_id):
    vertices = mfem.intArray()
    fespace.GetElementVertices(element_id, vertices)
    return vertices.ToList()


def _find_source_element(mesh, point):
    point_vec = _to_mfem_vector(point)

    for element_id in range(mesh.GetNE()):
        tr = mesh.GetElementTransformation(element_id)
        inverse = mfem.InverseElementTransformation(tr)
        ip = mfem.IntegrationPoint()
        status = inverse.Transform(point_vec, ip)
        if status != mfem.InverseElementTransformation.Outside:
            local_ip = mfem.IntegrationPoint()
            local_ip.Set(ip.x, ip.y, ip.z, ip.weight)
            return element_id, local_ip

    return None, None


def _find_matching_vertex(mesh, fespace, element_id, point, tol):
    for vertex_id in _get_element_vertices(fespace, element_id):
        vertex = np.array(mesh.GetVertexArray(vertex_id), dtype=float)[:len(point)]
        if np.linalg.norm(vertex - point) <= tol:
            return vertex_id
    return None


def _get_incident_elements(mesh, vertex_id):
    table = mesh.GetVertexToElementTable()
    connected = mfem.intArray()
    table.GetRow(vertex_id, connected)
    return connected.ToList()


def _vertex_corner_measure(mesh, fespace, element_id, vertex_id, point, tol):
    vertex = np.array(mesh.GetVertexArray(vertex_id), dtype=float)[:len(point)]
    neighbors = []
    for other_vertex_id in _get_element_vertices(fespace, element_id):
        if other_vertex_id == vertex_id:
            continue
        other = np.array(mesh.GetVertexArray(other_vertex_id), dtype=float)[:len(point)]
        edge = other - vertex
        norm = np.linalg.norm(edge)
        if norm > tol:
            neighbors.append((norm, edge))

    if not neighbors:
        return 0.0

    neighbors.sort(key=lambda item: item[0])
    dim = len(point)
    edge_vectors = [edge for _, edge in neighbors[:dim]]

    if dim == 1 or len(edge_vectors) == 1:
        return 1.0

    if dim == 2 and len(edge_vectors) >= 2:
        first = edge_vectors[0] / np.linalg.norm(edge_vectors[0])
        second = edge_vectors[1] / np.linalg.norm(edge_vectors[1])
        cosine = np.clip(np.dot(first, second), -1.0, 1.0)
        return float(np.arccos(cosine))

    if dim == 3 and len(edge_vectors) >= 3:
        a, b, c = edge_vectors[:3]
        la = np.linalg.norm(a)
        lb = np.linalg.norm(b)
        lc = np.linalg.norm(c)
        denom = (la * lb * lc + np.dot(a, b) * lc +
                 np.dot(b, c) * la + np.dot(c, a) * lb)
        numer = abs(np.dot(a, np.cross(b, c)))
        return float(2.0 * np.arctan2(numer, denom))

    return 1.0


def _add_dipole_to_element(rhs, fespace, element_id, point, moment, scale):
    tr = fespace.GetElementTransformation(element_id)
    inverse = mfem.InverseElementTransformation(tr)
    ip = mfem.IntegrationPoint()
    status = inverse.Transform(_to_mfem_vector(point), ip)
    if status == mfem.InverseElementTransformation.Outside:
        raise ValueError('Dipole point could not be mapped to the selected element.')

    tr.SetIntPoint(ip)
    fe = fespace.GetFE(element_id)
    vshape = mfem.DenseMatrix(fe.GetDof(), len(point))
    fe.CalcPhysVShape(tr, vshape)

    element_rhs = mfem.Vector(fe.GetDof())
    row = mfem.Vector(len(point))
    for i in range(fe.GetDof()):
        vshape.GetRow(i, row)
        element_rhs[i] = scale * float(np.dot(row.GetDataArray()[:len(point)], moment))

    vdofs = mfem.intArray()
    fespace.GetElementVDofs(element_id, vdofs)
    rhs.AddElementVector(vdofs, element_rhs)


def assemble_dipole_rhs(rhs, fespace, source):
    mesh = fespace.GetParMesh()
    element_id, source_ip = _find_source_element(mesh, source.dipole_pos)
    local_found = 1 if element_id is not None else 0
    global_found = MPI.COMM_WORLD.allreduce(local_found, op=MPI.SUM)
    if global_found == 0:
        raise ValueError('Dipole position lies outside the mesh.')

    # Determine vertex tolerance (needed before the mode branch so all ranks
    # can participate in the vertex-path collective calls below).
    bbox_min, bbox_max = mesh.GetBoundingBox()
    mesh_scale = max(np.linalg.norm(np.array(bbox_max) - np.array(bbox_min)), 1.0)
    vertex_tol = max(source.vertex_tol, 1e-12 * mesh_scale)

    # Determine local mode (off-rank ranks contribute nothing).
    vertex_id = None
    if element_id is not None:
        vertex_id = _find_matching_vertex(mesh, fespace, element_id,
                                          source.dipole_pos, vertex_tol)

    # Communicate whether any rank found a vertex match so all ranks agree
    # on which collective path to take.
    local_is_vertex = 1 if vertex_id is not None else 0
    global_is_vertex = MPI.COMM_WORLD.allreduce(local_is_vertex, op=MPI.SUM)

    if global_is_vertex == 0:
        # Interior-element path: no further collectives needed.
        if element_id is not None:
            _add_dipole_to_element(rhs, fespace, element_id,
                                   source.dipole_pos, source.dipole_moment, 1.0)
            return {'mode': 'element', 'local_elements': [element_id]}
        return {'mode': 'off-rank', 'local_elements': []}

    # Vertex path: ALL ranks must participate in the two allreduce calls below,
    # even those that don't own any element incident to the vertex (they
    # contribute zeros).
    local_elements = []
    local_raw_weights = []
    if vertex_id is not None:
        for connected_element in _get_incident_elements(mesh, vertex_id):
            if vertex_id in _get_element_vertices(fespace, connected_element):
                local_elements.append(int(connected_element))
                local_raw_weights.append(
                    _vertex_corner_measure(mesh, fespace, connected_element,
                                           vertex_id, source.dipole_pos, vertex_tol))

    local_weight_sum = float(sum(local_raw_weights))
    global_weight_sum = MPI.COMM_WORLD.allreduce(local_weight_sum, op=MPI.SUM)
    global_element_count = MPI.COMM_WORLD.allreduce(len(local_elements), op=MPI.SUM)

    if global_element_count == 0:
        raise ValueError('Dipole node was found, but no incident elements were detected.')

    if global_weight_sum > 0.0:
        scales = [weight / global_weight_sum for weight in local_raw_weights]
    else:
        scales = [1.0 / float(global_element_count) for _ in local_elements]

    for connected_element, scale in zip(local_elements, scales):
        _add_dipole_to_element(rhs, fespace, connected_element,
                               source.dipole_pos, source.dipole_moment, scale)

    return {
        'mode': 'vertex',
        'vertex_id': vertex_id,
        'local_elements': local_elements,
        'local_weights': scales,
    }


def parse_element_sigma_map(spec):
    """
    Parse strings like "0:1.0,5:4.0,8:8.0" into {element_id: sigma}.
    Element ids refer to the original serial mesh before refinement.
    """
    mapping = {}
    if not spec:
        return mapping

    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise ValueError(
                "Invalid --element-sigma entry '{0}'. Use elem_id:sigma".format(
                    item))
        elem_text, sigma_text = item.split(':', 1)
        element_id = int(elem_text.strip())
        sigma_value = validate_sigma_value(float(sigma_text.strip()),
                                           "element id {0}".format(element_id))
        mapping[element_id] = sigma_value
    return mapping


def summarize_element_sigma_map(element_sigma_map):
    if not element_sigma_map:
        return None

    element_ids = sorted(element_sigma_map)
    sigma_values = [element_sigma_map[element_id] for element_id in element_ids]
    return {
        'count': len(element_ids),
        'min_element_id': element_ids[0],
        'max_element_id': element_ids[-1],
        'min_sigma': min(sigma_values),
        'max_sigma': max(sigma_values),
    }


def validate_sigma_value(value, context='sigma'):
    sigma_value = float(value)
    if not np.isfinite(sigma_value):
        raise ValueError("Conductivity for {0} must be finite.".format(context))
    if sigma_value < 0.0:
        raise ValueError("Conductivity for {0} must be nonnegative.".format(context))
    return sigma_value


def set_element_conductivities(mesh, element_sigma_map, default_sigma=0.0):
    """
    Assign sigma values to selected original serial elements.

    All elements start with attribute 1 and conductivity default_sigma. Each
    distinct non-default conductivity gets a dedicated attribute, which is then
    inherited by refined children.
    """
    if mesh.GetNE() == 0:
        raise ValueError("Mesh has no elements.")

    default_sigma = validate_sigma_value(default_sigma, 'default sigma')

    sigma_by_attr = {1: default_sigma}
    sigma_to_attr = {default_sigma: 1}

    for element_id in range(mesh.GetNE()):
        mesh.SetAttribute(element_id, 1)

    for element_id, sigma_value in sorted(element_sigma_map.items()):
        if element_id < 0 or element_id >= mesh.GetNE():
            raise ValueError(
                "Element id {0} is outside the valid range [0, {1}]".format(
                    element_id, mesh.GetNE() - 1))

        sigma_value = validate_sigma_value(sigma_value,
                                           'element id {0}'.format(element_id))
        attr = sigma_to_attr.get(sigma_value)
        if attr is None:
            attr = len(sigma_to_attr) + 1
            sigma_to_attr[sigma_value] = attr
            sigma_by_attr[attr] = sigma_value

        mesh.SetAttribute(element_id, attr)

    mesh.SetAttributes()
    return sigma_by_attr


def build_sigma_coefficient(mesh, sigma_by_attr):
    num_attr = max(1, mesh.attributes.Max())
    sigma_values = mfem.Vector(num_attr)
    sigma_values.Assign(0.0)

    for attr, sigma_value in sigma_by_attr.items():
        if 1 <= attr <= num_attr:
            sigma_values[attr - 1] = sigma_value

    return mfem.PWConstCoefficient(sigma_values), sigma_values


def scale_linear_form(linear_form, factor):
    data = linear_form.GetDataArray()
    for index in range(len(data)):
        data[index] *= factor


def get_element_center(mesh, element_id):
    tr = mesh.GetElementTransformation(element_id)
    geom = mesh.GetElementBaseGeometry(element_id)
    center_ip = mfem.Geometries.GetCenter(geom)
    return np.array(tr.Transform(center_ip), dtype=float)


def get_element_size(mesh, fespace, element_id):
    vertices = _get_element_vertices(fespace, element_id)
    points = [np.array(mesh.GetVertexArray(vertex_id), dtype=float)[:mesh.SpaceDimension()]
              for vertex_id in vertices]

    max_distance = 0.0
    for i, first in enumerate(points):
        for second in points[i + 1:]:
            max_distance = max(max_distance, np.linalg.norm(first - second))

    return float(max_distance)


def get_source_region_size(mesh, fespace, dipole_info):
    local_sizes = [get_element_size(mesh, fespace, element_id)
                   for element_id in dipole_info.get('local_elements', [])]
    local_max_size = max(local_sizes) if local_sizes else 0.0
    global_max_size = MPI.COMM_WORLD.allreduce(local_max_size, op=MPI.MAX)
    return global_max_size, local_sizes


def mark_amr_elements(mesh, fespace, source, sigma_by_attr, default_sigma,
                      refine_fraction, forced_elements=None):
    """
    Mark elements with the strongest source contribution and conductivity
    contrast for local refinement.
    """
    refine_fraction = min(max(refine_fraction, 0.0), 1.0)
    num_elements = mesh.GetNE()
    if num_elements == 0 or refine_fraction <= 0.0:
        forced_elements = forced_elements or []
        marked = mfem.intArray(len(forced_elements))
        for i, element_id in enumerate(forced_elements):
            marked[i] = element_id
        return marked, []

    target_count = max(1, int(np.ceil(refine_fraction * num_elements)))
    attrs = mesh.GetAttributeArray()

    scored_elements = []
    for element_id in range(num_elements):
        center = get_element_center(mesh, element_id)
        source_score = 1.0 / (source.distance_squared(center) + 1e-12)

        sigma_value = float(default_sigma)
        if len(attrs) > element_id:
            sigma_value = float(sigma_by_attr.get(attrs[element_id], default_sigma))

        conductivity_jump = abs(sigma_value - default_sigma)
        score = source_score * (1.0 + conductivity_jump) + conductivity_jump
        scored_elements.append((score, element_id))

    scored_elements.sort(reverse=True)
    selected = [element_id for score, element_id in scored_elements[:target_count]
                if score > 0.0]

    if not selected and scored_elements:
        selected = [scored_elements[0][1]]

    if forced_elements:
        selected_set = set(selected)
        for element_id in forced_elements:
            selected_set.add(int(element_id))
        selected = sorted(selected_set)

    marked = mfem.intArray(len(selected))
    for i, element_id in enumerate(selected):
        marked[i] = element_id

    return marked, scored_elements[:min(5, len(scored_elements))]


def save_parallel_solution(pmesh, solution, results_dir, basename):
    smyid = '{:0>6d}'.format(myid)
    mesh_path = os.path.join(results_dir, basename + '_mesh.' + smyid)
    sol_real_path = os.path.join(results_dir, basename + '_sol_r.' + smyid)
    sol_imag_path = os.path.join(results_dir, basename + '_sol_i.' + smyid)
    pmesh.Print(mesh_path, 8)
    solution.real().Save(sol_real_path, 8)
    solution.imag().Save(sol_imag_path, 8)
    return mesh_path, sol_real_path, sol_imag_path


def run(order=1,
        meshfile='',
        rs=1,
        rp=0,
        sigma=0.0,
        element_sigma_map=None,
        amr_iterations=3,
        refine_fraction=0.15,
        source_max_size=0.01,
        visualization=False,
        freq=1.0,
        use_mumps=False,
        use_complex_mumps=False,
        use_pardiso=False,
        paraview_output=False):
    """
    Run the AMR Maxwell solver with dipole source and conductivity.
    """
    if element_sigma_map is None:
        element_sigma_map = {}
    element_sigma_summary = summarize_element_sigma_map(element_sigma_map)

    if myid == 0:
        print('=' * 70)
        print('Adaptive Maxwell Equation - Dipole Source with Conductivity')
        print('=' * 70)
        print('Equation: curl(mu^-1 curl E) - omega^2 epsilon E + i omega sigma E = -i omega J_s')
        print(f'Polynomial order: {order}')
        print(f'Mesh file: {os.path.basename(meshfile)}')
        print(f'Default conductivity sigma: {sigma}')
        print(f'Frequency: {freq}')
        print(f'AMR iterations: {amr_iterations}')
        print(f'Refine fraction: {refine_fraction}')
        print(f'Source-region max element size: {source_max_size}')
        if element_sigma_summary is not None:
            print('Element conductivity overrides:')
            print(f"  count: {element_sigma_summary['count']}")
            print(f"  element id range: [{element_sigma_summary['min_element_id']}, {element_sigma_summary['max_element_id']}]")
            print(f"  sigma range: [{element_sigma_summary['min_sigma']}, {element_sigma_summary['max_sigma']}]")
            print(f'  map: {element_sigma_map}')
        print('=' * 70)

    serial_mesh = mfem.Mesh(meshfile, 1, 1)
    serial_mesh.EnsureNCMesh(True)
    dim = serial_mesh.Dimension()

    sigma_by_attr = set_element_conductivities(serial_mesh, element_sigma_map, sigma)

    for _ in range(rs):
        serial_mesh.UniformRefinement()

    pmesh = mfem.ParMesh(MPI.COMM_WORLD, serial_mesh)
    del serial_mesh

    for _ in range(rp):
        pmesh.UniformRefinement()

    sigma_coef, sigma_values = build_sigma_coefficient(pmesh, sigma_by_attr)
    freq = float(freq)
    if not np.isfinite(freq) or freq <= 0.0:
        raise ValueError('Frequency must be positive and finite for the complex Maxwell formulation.')
    omega = 2.0 * pi * freq
    epsilon = 1.0
    wavenumber = omega * sqrt(epsilon)

    fec = mfem.ND_FECollection(order, dim)
    fespace = mfem.ParFiniteElementSpace(pmesh, fec)

    bbox_min, bbox_max = pmesh.GetBoundingBox()
    bbox_min = np.array(bbox_min)
    bbox_max = np.array(bbox_max)
    dipole_pos = 0.5 * (bbox_min + bbox_max)

    dipole_moment = np.zeros(dim)
    dipole_moment[dim - 1] = 1.0
    J = DipoleSource(dim, dipole_pos, dipole_moment)

    conv = mfem.ComplexOperator.HERMITIAN
    muinv = mfem.ConstantCoefficient(1.0)
    mass_coef = mfem.ConstantCoefficient(-omega**2 * epsilon)
    loss_coef = mfem.ConstantCoefficient(omega)

    a = mfem.ParSesquilinearForm(fespace, conv)
    a.AddDomainIntegrator(mfem.CurlCurlIntegrator(muinv), None)
    a.AddDomainIntegrator(mfem.VectorFEMassIntegrator(mass_coef),
                          mfem.VectorFEMassIntegrator(mfem.ProductCoefficient(loss_coef,
                                                                               sigma_coef)))

    pc_op = mfem.ParBilinearForm(fespace)
    pc_op.AddDomainIntegrator(mfem.CurlCurlIntegrator(muinv))
    pc_op.AddDomainIntegrator(mfem.VectorFEMassIntegrator(
        mfem.ConstantCoefficient(omega**2 * epsilon)))
    pc_op.AddDomainIntegrator(mfem.VectorFEMassIntegrator(
        mfem.ProductCoefficient(loss_coef, sigma_coef)))

    b = mfem.ParComplexLinearForm(fespace, conv)

    E = mfem.ParComplexGridFunction(fespace)
    E.Assign(0.0)

    results_dir = os.path.join(os.path.dirname(__file__), 'results', 'maxwell_dipole_amr')
    if myid == 0:
        os.makedirs(results_dir, exist_ok=True)
    MPI.COMM_WORLD.Barrier()

    paraview_dc = None
    if paraview_output:
        paraview_dc = mfem.ParaViewDataCollection('maxwell_dipole_amr', pmesh)
        paraview_dc.SetPrefixPath(results_dir)
        paraview_dc.SetLevelsOfDetail(order)
        paraview_dc.SetDataFormat(mfem.VTKFormat_BINARY)
        paraview_dc.SetHighOrderOutput(True)
        paraview_dc.RegisterField('E_real', E.real())
        paraview_dc.RegisterField('E_imag', E.imag())

    if myid == 0:
        print(f'Mesh dimension: {dim}')
        print(f'Initial local elements: {pmesh.GetNE()}')
        print(f'Initial H(curl) unknowns: {fespace.GlobalTrueVSize()}')
        print(f'Dipole position: {dipole_pos}')
        print(f'Dipole moment: {dipole_moment}')
        print(f'Derived wavenumber k: {wavenumber}')
        print(f'Angular frequency omega: {omega}')
        print(f'Permittivity epsilon: {epsilon}')
        print('Sigma by attribute:')
        for attr in sorted(sigma_by_attr):
            print(f'  attr {attr}: sigma = {sigma_by_attr[attr]}')
        if sigma_values.Size() > 0:
            print(f'Max conductivity: {max(sigma_values[i] for i in range(sigma_values.Size())):.6e}')
    # sys.exit()
    if visualization:
        sol_sock_real = mfem.socketstream('localhost', 19916)
        sol_sock_imag = mfem.socketstream('localhost', 19916)
        if (not sol_sock_real.good() or not sol_sock_imag.good()) and myid == 0:
            print('Unable to connect to GLVis. Disabling visualization.')
            visualization = False
    else:
        sol_sock_real = None
        sol_sock_imag = None

    amr_step = 0
    while True:
        ess_tdof_list = mfem.intArray()
        if pmesh.bdr_attributes.Size() > 0:
            ess_bdr = mfem.intArray([1] * pmesh.bdr_attributes.Max())
            fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list)

        if myid == 0:
            print('\n' + '-' * 70)
            print(f'AMR step {amr_step}')
            print(f'Local elements: {pmesh.GetNE()}')
            print(f'Global H(curl) unknowns: {fespace.GlobalTrueVSize()}')
            print('Assembling system...')

        b.Assign(0.0)
        b.real().Assign(0.0)
        b.imag().Assign(0.0)
        dipole_info = assemble_dipole_rhs(b.imag(), fespace, J)
        scale_linear_form(b.imag(), -omega)
        a.Assemble()
        pc_op.Assemble()

        A = mfem.OperatorHandle()
        B = mfem.Vector()
        X = mfem.Vector()
        a.FormLinearSystem(ess_tdof_list, E, b, A, X, B)
        pc_handle = mfem.OperatorHandle()
        pc_op.FormSystemMatrix(ess_tdof_list, pc_handle)
        AA = pc_handle.AsHypreParMatrix()
        global_dofs = fespace.GlobalTrueVSize()
        if myid == 0:
            print(f'Local system size on rank 0: {A.Height()} (= 2 x {A.Height()//2} local DOFs)')
            print(f'Global system size: {fespace.GlobalTrueVSize()}')

        active_solver = 'fgmres'

        # --- ComplexMUMPS: native complex ZMUMPS solver ---
        if use_complex_mumps and active_solver == 'fgmres':
            try:
                from mfem._par.mumps import ComplexMUMPSSolver
                if not A.IsComplexHypreParMatrix():
                    if myid == 0:
                        print('ComplexMUMPS: operator is not a ComplexHypreParMatrix; '
                              'falling back.')
                else:
                    t_ah = MPI.Wtime()
                    Ah = A.AsComplexHypreParMatrix()
                    t0 = MPI.Wtime()
                    csolver = ComplexMUMPSSolver(MPI.COMM_WORLD)
                    csolver.SetMatrixSymType(ComplexMUMPSSolver.UNSYMMETRIC)
                    csolver.SetMemRelaxation(100)
                    csolver.SetPrintLevel(1 if myid == 0 else 0)
                    if global_dofs> 1000000:
                        # ICNTL(14): large value avoids MUMPS -8/-9 retry-realloc loop
                        # (incremental memory growth). 200 = allocate 2x the estimate upfront.
                        csolver.SetMemRelaxation(200)
                        # ParMETIS gives 3-5x better fill reduction than AMD for 3D H(curl)
                        csolver.SetReorderingStrategy(ComplexMUMPSSolver.PARMETIS)
                        # BLR: reduces memory and factor time for large problems.
                        # Mode 2 = BLR in both factorization and solution phases.
                        # Tol 1e-4 balances accuracy vs. compression; tighten if solution
                        # accuracy degrades.
                        csolver.SetBLRMode(2)
                        csolver.SetBLRTol(1e-4)
                        csolver.SetBLRCompressionType(1)   # ICNTL(36)=1: UCFS (lower memory)
                        csolver.SetBLRCBCompression(0)     # ICNTL(37)=1: compress contribution blocks
                        # csolver.SetNumThreads(4)
                        # csolver.SetPivotThreshold(0.01)
                        # csolver.SetOutOfCore(0)
                        # csolver.SetReorderingReuse(True)
                    csolver.SetOperator(Ah)
                    if myid == 0:
                        print('Solving with ComplexMUMPS...')
                    csolver.Mult(B, X)
                    if myid == 0:
                        print(f'ComplexMUMPS solve completed in {MPI.Wtime() - t0:.4f} s')
                    active_solver = 'complex_mumps'
            except Exception as exc:
                if myid == 0:
                    print(f'ComplexMUMPS not available, falling back: {exc}')

        # --- Real MUMPS: 2x2 block-expanded real system ---
        if use_mumps and active_solver == 'fgmres':
            try:
                from mfem._par.mumps import MUMPSSolver
                t0 = MPI.Wtime()
                mumps = MUMPSSolver(MPI.COMM_WORLD)
                mumps.SetMatrixSymType(MUMPSSolver.UNSYMMETRIC)
                mumps.SetPrintLevel(1 if myid == 0 else 0)
                mumps.SetOperator(A.Ptr())
                if myid == 0:
                    print('Solving with MUMPS (real 2x2 block)...')
                mumps.Mult(B, X)
                if myid == 0:
                    print(f'MUMPS solve completed in {MPI.Wtime() - t0:.4f} s')
                active_solver = 'mumps'
            except Exception as exc:
                if myid == 0:
                    print(f'MUMPS not available, falling back: {exc}')

        # --- Real Pardiso: Intel MKL Pardiso on 2x2 block-expanded real system ---
        if use_pardiso and active_solver == 'fgmres':
            try:
                from mfem._par.pardiso import PardisoSolver
                t0 = MPI.Wtime()
                pardiso = PardisoSolver()
                pardiso.SetMatrixType(PardisoSolver.REAL_NONSYMMETRIC)
                pardiso.SetPrintLevel(0)
                pardiso.SetOperator(A.Ptr())
                if myid == 0:
                    print('Solving with Pardiso (real 2x2 block)...')
                pardiso.Mult(B, X)
                if myid == 0:
                    print(f'Pardiso solve completed in {MPI.Wtime() - t0:.4f} s')
                active_solver = 'pardiso'
            except Exception as exc:
                if myid == 0:
                    print(f'Pardiso not available, falling back: {exc}')

        # --- FGMRES + AMS block diagonal preconditioner (default) ---
        if active_solver == 'fgmres':
            block_true_offsets = mfem.intArray()
            block_true_offsets.SetSize(3)
            block_true_offsets[0] = 0
            block_true_offsets[1] = A.Height() // 2
            block_true_offsets[2] = A.Height() // 2
            block_true_offsets.PartialSum()

            bdp = mfem.BlockDiagonalPreconditioner(block_true_offsets)
            ams = mfem.HypreAMS(AA, fespace)
            ams.SetPrintLevel(0)
            ams_imag = mfem.ScaledOperator(ams,
                                           -1 if conv == mfem.ComplexOperator.HERMITIAN else 1)
            bdp.SetDiagonalBlock(0, ams)
            bdp.SetDiagonalBlock(1, ams_imag)

            gmres = mfem.FGMRESSolver(MPI.COMM_WORLD)
            gmres.SetPrintLevel(1)
            gmres.SetKDim(100)
            gmres.SetMaxIter(500)
            gmres.SetRelTol(1e-12)
            gmres.SetAbsTol(1e-10)
            gmres.SetOperator(A.Ptr())
            gmres.SetPreconditioner(bdp)
            gmres.Mult(B, X)

            if myid == 0:
                if gmres.GetConverged():
                    print(f'FGMRES converged in {gmres.GetNumIterations()} iterations')
                else:
                    print('FGMRES did not converge')

        a.RecoverFEMSolution(X, b, E)

        e_real_norm = sqrt(mfem.InnerProduct(MPI.COMM_WORLD, E.real(), E.real()))
        e_imag_norm = sqrt(mfem.InnerProduct(MPI.COMM_WORLD, E.imag(), E.imag()))
        E_norm = sqrt(e_real_norm**2 + e_imag_norm**2)
        source_region_size, local_source_sizes = get_source_region_size(
            pmesh, fespace, dipole_info)
        if myid == 0:
            print(f'||E|| (L2 norm): {E_norm:.6e}')
            print(f'  real part norm: {e_real_norm:.6e}')
            print(f'  imag part norm: {e_imag_norm:.6e}')
            if dipole_info['mode'] == 'element':
                print(f"Dipole assembled in one element: {dipole_info['local_elements'][0]}")
            elif dipole_info['mode'] == 'vertex':
                print(f"Dipole assembled from node-sharing cells across vertex {dipole_info['vertex_id']}")
            print(f'Source-region max element size: {source_region_size:.6e}')

        if paraview_dc is not None:
            paraview_dc.SetCycle(amr_step)
            paraview_dc.SetTime(float(amr_step))
            paraview_dc.Save()

        if visualization and sol_sock_real is not None and sol_sock_imag is not None:
            if sol_sock_real.good():
                sol_sock_real.send_text('parallel ' + str(num_procs) + ' ' + str(myid))
                sol_sock_real.precision(8)
                sol_sock_real.send_solution(pmesh, E.real())
                sol_sock_real.send_text("window_title 'AMR Maxwell Dipole (real)'")
                sol_sock_real.flush()
            if sol_sock_imag.good():
                sol_sock_imag.send_text('parallel ' + str(num_procs) + ' ' + str(myid))
                sol_sock_imag.precision(8)
                sol_sock_imag.send_solution(pmesh, E.imag())
                sol_sock_imag.send_text("window_title 'AMR Maxwell Dipole (imag)'")
                sol_sock_imag.flush()

        source_needs_refinement = source_region_size > source_max_size
        if amr_step >= amr_iterations:
            break

        forced_elements = []
        if source_needs_refinement:
            forced_elements = [element_id for element_id, element_size in zip(
                dipole_info.get('local_elements', []), local_source_sizes)
                if element_size > source_max_size]

        marked, top_markers = mark_amr_elements(
            pmesh, fespace, J, sigma_by_attr, sigma, refine_fraction,
            forced_elements=forced_elements)

        global_marked_count = MPI.COMM_WORLD.allreduce(marked.Size(), op=MPI.SUM)

        if global_marked_count == 0:
            if myid == 0:
                print('No elements were marked for refinement. Stopping AMR.')
            break

        if myid == 0:
            preview = ', '.join(
                f'(score={score:.3e}, elem={element_id})'
                for score, element_id in top_markers)
            print(f'Refining {marked.Size()} local elements')
            if source_needs_refinement:
                print('Forcing refinement near the dipole source until the local size target is met.')
            print(f'Top markers: {preview}')

        pmesh.GeneralRefinement(marked)

        fespace.Update()
        E.Update()

        if pmesh.Nonconforming():
            pmesh.Rebalance()
            fespace.Update()
            E.Update()

        a.Update()
        pc_op.Update()
        b.Update()
        amr_step += 1

    mesh_path, sol_real_path, sol_imag_path = save_parallel_solution(
        pmesh, E, results_dir, 'maxwell_dipole_amr_final')

    if myid == 0:
        print('\n' + '=' * 70)
        print('Final output')
        print(f'MFEM mesh: {mesh_path}')
        print(f'MFEM solution (real): {sol_real_path}')
        print(f'MFEM solution (imag): {sol_imag_path}')
        if paraview_output:
            print('ParaView output: maxwell_dipole_amr/maxwell_dipole_amr.pvd')
        print('=' * 70)


if __name__ == '__main__':
    from mfem.common.arg_parser import ArgParser

    parser = ArgParser(description='AMR Maxwell Dipole with Conductivity')
    parser.add_argument('-m', '--mesh',
                        default='inline-hex.mesh',
                        action='store', type=str,
                        help='Mesh file to use.')
    parser.add_argument('-o', '--order',
                        action='store', default=1, type=int,
                        help='Finite element order (polynomial degree).')
    parser.add_argument('-rs', '--refine-serial',
                        action='store', default=1, type=int,
                        help='Number of serial uniform refinement steps.')
    parser.add_argument('-rp', '--refine-parallel',
                        action='store', default=0, type=int,
                        help='Number of parallel uniform refinement steps.')
    parser.add_argument('-sigma', '--sigma',
                        action='store', default=0.1, type=float,
                        help='Default conductivity value. Must be finite and nonnegative.')
    parser.add_argument('-es', '--element-sigma',
                        action='store', default='', type=str,
                        help='Comma-separated element_id:sigma list on the original serial mesh. Each sigma must be finite and nonnegative.')
    parser.add_argument('-amr', '--max-amr-iterations',
                        action='store', default=1, type=int,
                        help='Number of AMR refinement steps after the initial solve.')
    parser.add_argument('-rf', '--refine-fraction',
                        action='store', default=0.15, type=float,
                        help='Fraction of local elements marked in each AMR step.')
    parser.add_argument('-sms', '--source-max-size',
                        action='store', default=0.01, type=float,
                        help='Refine near the dipole until the local element size is below this threshold.')
    parser.add_argument('-f', '--frequency',
                        action='store', default=1.0, type=float,
                        help='Frequency used to define omega = 2*pi*f and i*omega*sigma.')
    parser.add_argument('-vis', '--visualization',
                        action='store_true',
                        help='Enable GLVis visualization.')
    parser.add_argument('-mumps', '--use-mumps',
                        action='store_true',
                        help='Use the real MUMPS direct solver (2x2 block-expanded system).')
    parser.add_argument('-cmumps', '--use-complex-mumps',
                        action='store_true',
                        help='Use ComplexMUMPS (ZMUMPS) direct solver on the native complex system.')
    parser.add_argument('-pardiso', '--use-pardiso',
                        action='store_true',
                        help='Use Intel MKL Pardiso direct solver (2x2 block-expanded real system, requires MFEM_USE_MKL_PARDISO).')
    parser.add_argument('-pv', '--paraview',
                        action='store_true',
                        help='Export the AMR states to ParaView/VTK.')

    args = parser.parse_args()

    if myid == 0:
        parser.print_options(args)
    
    meshfile = expanduser(join(os.path.dirname(__file__), '..', 'data', args.mesh))
    element_sigma_map = parse_element_sigma_map(args.element_sigma)

    if myid == 0 and element_sigma_map:
        summary = summarize_element_sigma_map(element_sigma_map)
        print('Parsed --element-sigma overrides:')
        print(f"  count: {summary['count']}")
        print(f"  element id range: [{summary['min_element_id']}, {summary['max_element_id']}]")
        print(f"  sigma range: [{summary['min_sigma']}, {summary['max_sigma']}]")

    run(order=args.order,
        meshfile=meshfile,
        rs=args.refine_serial,
        rp=args.refine_parallel,
        sigma=args.sigma,
        element_sigma_map=element_sigma_map,
        amr_iterations=args.max_amr_iterations,
        refine_fraction=args.refine_fraction,
        source_max_size=args.source_max_size,
        visualization=args.visualization,
        freq=args.frequency,
        use_mumps=args.use_mumps,
        use_complex_mumps=args.use_complex_mumps,
        use_pardiso=args.use_pardiso,
        paraview_output=args.paraview)