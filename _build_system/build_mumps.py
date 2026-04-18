# ----------------------------------------------------------------------------------------
# Routines for MUMPS
# ----------------------------------------------------------------------------------------

import sys
import os
import re
import subprocess
import tempfile

__all__ = ["cmake_make_mumps"]

from build_utils import *
from build_consts import *

import build_globals as bglb


def _find_lib(name, search_dirs):
    '''Search for a shared or static library in common system locations.'''
    for d in search_dirs:
        for ext in ('.so', '.a'):
            p = os.path.join(d, 'lib' + name + ext)
            if os.path.exists(p):
                return p
    return None


def _get_runpath(path):
    result = subprocess.run(
        ['readelf', '-d', path],
        check=True,
        capture_output=True,
        text=True)
    match = re.search(r'\(RUNPATH\)\s+Library runpath: \[(.*)\]', result.stdout)
    if match:
        return match.group(1)
    match = re.search(r'\(RPATH\)\s+Library rpath: \[(.*)\]', result.stdout)
    return match.group(1) if match else ''


def _set_relative_rpath(path, new_rpath):
    old_rpath = _get_runpath(path)
    if not old_rpath or old_rpath == new_rpath:
        return

    script = (
        'file(RPATH_CHANGE\n'
        f'  FILE "{path}"\n'
        f'  OLD_RPATH "{old_rpath}"\n'
        f'  NEW_RPATH "{new_rpath}")\n'
    )
    with tempfile.NamedTemporaryFile('w', suffix='.cmake', delete=False,
                                     encoding='utf-8') as stream:
        stream.write(script)
        script_path = stream.name
    try:
        subprocess.run(['cmake', '-P', script_path], check=True)
    finally:
        os.unlink(script_path)


def _normalize_mumps_rpaths(prefix):
    libdir = os.path.join(prefix, 'lib')
    if not os.path.isdir(libdir):
        return

    for name in os.listdir(libdir):
        if not (name.startswith('lib') and '.so' in name):
            continue
        if 'mumps' not in name and name != 'libpord.so':
            continue
        _set_relative_rpath(os.path.join(libdir, name), '$ORIGIN')


def cmake_make_mumps():
    '''
    build MUMPS using scivision/mumps-superbuild
    
    This uses CMake-enabled MUMPS from:
    https://github.com/scivision/mumps-superbuild
    
    MUMPS requires:
    - BLAS/LAPACK (required)
    - ScaLAPACK (auto-built if not found)
    - MPI (C and Fortran)
    - METIS (optional, for matrix ordering)
    - Fortran compiler (required)
    
    The superbuild will automatically download and build ScaLAPACK if needed.
    '''
    if bglb.verbose:
        print("Building MUMPS using scivision/mumps-superbuild")

    cmbuild = 'cmbuild'
    path = os.path.join(extdir, 'mumps', cmbuild)
    if os.path.exists(path):
        print("working directory already exists!")
    else:
        os.makedirs(path)

    pwd = chdir(path)

    # MUMPS superbuild CMake options
    cmake_opts = {
        'DBUILD_SHARED_LIBS': '1',
        'DCMAKE_INSTALL_PREFIX': bglb.mumps_prefix,
        'DCMAKE_C_COMPILER': bglb.mpicc_command,
        'DCMAKE_Fortran_COMPILER': bglb.mpifort_command,
        'DCMAKE_BUILD_TYPE': 'Release',
    }
    
    # Enable RPATH for shared libraries
    if sys.platform == "darwin":
        cmake_opts['DCMAKE_MACOSX_RPATH'] = 'YES'
        cmake_opts['DCMAKE_INSTALL_NAME_DIR'] = '@rpath'
        cmake_opts['DMUMPS_ENABLE_RPATH'] = 'ON'
    elif sys.platform in ("linux", "linux2"):
        cmake_opts['DCMAKE_INSTALL_RPATH'] = "$ORIGIN"
        cmake_opts['DCMAKE_BUILD_WITH_INSTALL_RPATH'] = '1'
        cmake_opts['DMUMPS_ENABLE_RPATH'] = 'ON'
    
    # Enable METIS if available - must pass library and include paths explicitly
    # so that find_package(METIS) succeeds. If MUMPS_find_metis is OFF (default),
    # the superbuild falls back to a git submodule which fails for tarball downloads.
    if bglb.metis_prefix != '':
        metis_lib = _find_lib("metis", [
            os.path.join(bglb.metis_prefix, "lib"),
            os.path.join(bglb.metis_prefix, "lib64"),
        ])
        metis_inc = os.path.join(bglb.metis_prefix, "include")
        if metis_lib and os.path.exists(metis_inc):
            cmake_opts['DMUMPS_metis'] = 'ON'
            cmake_opts['DMUMPS_find_metis'] = 'ON'
            cmake_opts['DMETIS_LIBRARY'] = metis_lib
            cmake_opts['DMETIS_INCLUDE_DIR'] = metis_inc

            # Enable ParMETIS if prefix is set and library is present
            parmetis_lib = _find_lib("parmetis", [
                os.path.join(bglb.parmetis_prefix, "lib"),
                os.path.join(bglb.parmetis_prefix, "lib64"),
            ]) if bglb.parmetis_prefix != '' else None
            if parmetis_lib:
                cmake_opts['DMUMPS_parmetis'] = 'ON'
                cmake_opts['DPARMETIS_LIBRARY'] = parmetis_lib
            else:
                cmake_opts['DMUMPS_parmetis'] = 'OFF'
        else:
            cmake_opts['DMUMPS_metis'] = 'OFF'
            cmake_opts['DMUMPS_parmetis'] = 'OFF'
    else:
        cmake_opts['DMUMPS_metis'] = 'OFF'
        cmake_opts['DMUMPS_parmetis'] = 'OFF'

    # Precision options - match spack: +double +float +complex ~int64
    cmake_opts['DBUILD_SINGLE'] = 'ON'    # +float
    cmake_opts['DBUILD_DOUBLE'] = 'ON'    # +double
    cmake_opts['DBUILD_COMPLEX'] = 'ON'   # +complex (complex32)
    cmake_opts['DBUILD_COMPLEX16'] = 'ON' # +complex (complex64)

    # Disable optional features - match spack: ~openmp ~scotch ~ptscotch ~blr_mt
    cmake_opts['DMUMPS_openmp'] = 'OFF'
    cmake_opts['DMUMPS_scotch'] = 'OFF'
    cmake_opts['DMUMPS_ptscotch'] = 'OFF'
    cmake_opts['DMUMPS_matlab'] = 'OFF'
    cmake_opts['DMUMPS_intsize64'] = 'OFF'  # ~int64
    
    # Auto-build ScaLAPACK if not found (superbuild feature).
    # If we already built ScaLAPACK ourselves, point the superbuild to it
    # so it skips its own git-submodule-based build.
    cmake_opts['DMUMPS_find_SCALAPACK'] = 'ON'
    if bglb.scalapack_prefix != '':
        scalapack_lib = _find_lib("scalapack", [
            os.path.join(bglb.scalapack_prefix, "lib"),
            os.path.join(bglb.scalapack_prefix, "lib64"),
        ])
        if scalapack_lib:
            cmake_opts['DSCALAPACK_LIBRARY'] = scalapack_lib
    
    # Add BLAS/LAPACK - explicitly search common locations if not specified
    if bglb.blas_libraries != "":
        cmake_opts['DBLAS_LIBRARIES'] = bglb.blas_libraries
    else:
        blas_lib = _find_lib("blas", [
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib/aarch64-linux-gnu",
            "/usr/lib",
            "/usr/lib64",
        ])
        if blas_lib:
            cmake_opts['DBLAS_LIBRARIES'] = blas_lib

    if bglb.lapack_libraries != "":
        cmake_opts['DLAPACK_LIBRARIES'] = bglb.lapack_libraries
    else:
        lapack_lib = _find_lib("lapack", [
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib/aarch64-linux-gnu",
            "/usr/lib",
            "/usr/lib64",
        ])
        if lapack_lib:
            cmake_opts['DLAPACK_LIBRARIES'] = lapack_lib
    
    if bglb.verbose:
        cmake_opts['DCMAKE_VERBOSE_MAKEFILE'] = '1'

    try:
        cmake('..', **cmake_opts)
        make('mumps')
        make_install('mumps')
        if sys.platform in ("linux", "linux2"):
            _normalize_mumps_rpaths(bglb.mumps_prefix)
    except Exception as e:
        print("="*70)
        print("WARNING: MUMPS build failed.")
        print("="*70)
        print(f"Error: {e}")
        print("")
        print("MUMPS requires:")
        print("  - BLAS/LAPACK libraries")
        print("  - MPI Fortran compiler (mpif90/mpifort)")
        print("  - Optional: METIS for matrix ordering")
        print("")
        print("ScaLAPACK will be auto-built if not found.")
        print("")
        print("Alternatively, use a pre-installed MUMPS:")
        print("  spack install mumps+mpi+double")
        print("  python -m pip install . -C\"mumps-prefix=$(spack location -i mumps)\"")
        print("="*70)
        raise

    os.chdir(pwd)
