# ----------------------------------------------------------------------------------------
# Routines for SuiteSparse
# ----------------------------------------------------------------------------------------
# SuiteSparse is a suite of sparse matrix algorithms including UMFPACK, KLU, CHOLMOD, etc.
# Source: https://github.com/DrTimothyAldenDavis/SuiteSparse
# MFEM uses it via MFEM_USE_SUITESPARSE=YES which requires UMFPACK, KLU, AMD, BTF,
# CHOLMOD, COLAMD, CAMD, CCOLAMD, and the SuiteSparse_config component.
# ----------------------------------------------------------------------------------------

import sys
import os

__all__ = ["cmake_make_suitesparse"]

from build_utils import *
from build_consts import *

import build_globals as bglb


def cmake_make_suitesparse():
    '''
    Build SuiteSparse using CMake.

    SuiteSparse is required by MFEM when MFEM_USE_SUITESPARSE=YES.
    It provides UMFPackSolver, KLUSolver, CHOLMOD etc.
    Source: https://github.com/DrTimothyAldenDavis/SuiteSparse

    Requires:
    - BLAS and LAPACK
    - C and C++ compilers
    - METIS (optional, used by CHOLMOD for ordering)
    '''
    if bglb.verbose:
        print("Building SuiteSparse")

    path = os.path.join(extdir, 'suitesparse')
    if not bglb.dry_run and not os.path.exists(path):
        assert False, "suitesparse source not found – run download('suitesparse') first"

    cmbuild = os.path.join(path, 'cmbuild')
    if os.path.exists(cmbuild):
        print("working directory already exists!")
    else:
        os.makedirs(cmbuild)

    pwd = chdir(cmbuild)

    cmake_opts = {
        'DBUILD_SHARED_LIBS': '1',
        'DCMAKE_INSTALL_PREFIX': bglb.suitesparse_prefix,
        'DCMAKE_C_COMPILER': bglb.cc_command,
        'DCMAKE_CXX_COMPILER': bglb.cxx_command,
        'DCMAKE_BUILD_TYPE': 'Release',
        # Build only what MFEM needs; skip MATLAB interface, CUDA, GraphBLAS etc.
        'DSUITESPARSE_ENABLE_PROJECTS': 'suitesparse_config;amd;camd;colamd;ccolamd;cholmod;umfpack;klu;btf',
        'DSUITESPARSE_USE_CUDA': 'OFF',
        'DSUITESPARSE_USE_OPENMP': 'OFF',
        'DSUITESPARSE_USE_FORTRAN': 'OFF',
        'DCHOLMOD_USE_CUDA': 'OFF',
    }

    # METIS support in CHOLMOD
    if bglb.metis_prefix != '':
        metis_lib_dir = os.path.join(bglb.metis_prefix, 'lib')
        metis_inc_dir = os.path.join(bglb.metis_prefix, 'include')
        if os.path.isdir(metis_inc_dir):
            cmake_opts['DSUITESPARSE_USE_SYSTEM_METIS'] = '1'
            cmake_opts['DMETIS_INCLUDE_DIRS'] = metis_inc_dir
            cmake_opts['DMETIS_LIBRARIES'] = os.path.join(metis_lib_dir, 'libmetis.so')

    # Explicit BLAS/LAPACK if specified
    if bglb.blas_libraries != '':
        cmake_opts['DBLAS_LIBRARIES'] = bglb.blas_libraries
    if bglb.lapack_libraries != '':
        cmake_opts['DLAPACK_LIBRARIES'] = bglb.lapack_libraries

    if sys.platform in ('linux', 'linux2'):
        cmake_opts['DCMAKE_INSTALL_RPATH'] = '$ORIGIN'
        cmake_opts['DCMAKE_BUILD_WITH_INSTALL_RPATH'] = '1'
    elif sys.platform == 'darwin':
        cmake_opts['DCMAKE_MACOSX_RPATH'] = 'YES'
        cmake_opts['DCMAKE_INSTALL_NAME_DIR'] = '@rpath'

    if bglb.verbose:
        cmake_opts['DCMAKE_VERBOSE_MAKEFILE'] = '1'

    cmake('..', **cmake_opts)
    make('suitesparse')
    make_install('suitesparse')

    os.chdir(pwd)
