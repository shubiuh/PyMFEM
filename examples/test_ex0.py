from __future__ import print_function
import os
import sys
import time
import pytest
from os.path import expanduser, join
from ex0 import run

MESH_DIR = expanduser(join(os.path.dirname(__file__), '..', 'data'))


def get_meshfile(name='star.mesh'):
    return join(MESH_DIR, name)


def test_run_default():
    """Test run with default order=1 and star.mesh."""
    meshfile = get_meshfile('star.mesh')
    run(order=1, meshfile=meshfile, use_pardiso=False, use_full_assembly=False)
    assert os.path.isfile('sol.gf'), "sol.gf was not created"
    assert os.path.isfile('mesh.mesh'), "mesh.mesh was not created"


def test_run_order2():
    """Test run with order=2."""
    meshfile = get_meshfile('star.mesh')
    run(order=2, meshfile=meshfile, use_pardiso=False, use_full_assembly=False)
    assert os.path.isfile('sol.gf'), "sol.gf was not created"


def test_run_fichera():
    """Test run with fichera.mesh and order=2."""
    meshfile = get_meshfile('fichera.mesh')
    run(order=2, meshfile=meshfile, use_pardiso=False, use_full_assembly=False)
    assert os.path.isfile('sol.gf'), "sol.gf was not created"


def test_run_full_assembly():
    """Test run with full assembly enabled."""
    meshfile = get_meshfile('star.mesh')
    run(order=1, meshfile=meshfile, use_pardiso=False, use_full_assembly=True)
    assert os.path.isfile('sol.gf'), "sol.gf was not created"


def test_run_elapsed_time():
    """Test that the solver completes in a reasonable time."""
    meshfile = get_meshfile('star.mesh')
    start = time.time()
    run(order=1, meshfile=meshfile, use_pardiso=False, use_full_assembly=False)
    elapsed = time.time() - start
    print(f"Elapsed time for solver: {elapsed:.4f}s")
    assert elapsed < 60, f"Solver took too long: {elapsed:.2f}s"


def test_run_pardiso():
    """Test run with Pardiso solver if available."""
    pytest.importorskip('mfem._ser.pardiso', reason='Pardiso not available')
    meshfile = get_meshfile('star.mesh')
    start = time.time()
    run(order=1, meshfile=meshfile, use_pardiso=True, use_full_assembly=False)
    elapsed = time.time() - start
    print(f"Elapsed time for Pardiso solver: {elapsed:.4f}s")
    assert os.path.isfile('sol.gf'), "sol.gf was not created"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])