# ----------------------------------------------------------------------------------------
# Routines for ScaLAPACK
# ----------------------------------------------------------------------------------------
import sys
import os

__all__ = ["cmake_make_scalapack"]

from build_utils import *
from build_consts import *

import build_globals as bglb


def _patch_scalapack_blacs_cmake(path):
    cmake_file = os.path.join(path, 'BLACS', 'INSTALL', 'CMakeLists.txt')
    if not os.path.exists(cmake_file):
        return

    with open(cmake_file, 'r', encoding='utf-8') as stream:
        content = stream.read()

    legacy = 'cmake_minimum_required(VERSION 2.8)'
    updated = 'cmake_minimum_required(VERSION 3.5)'
    if legacy not in content:
        return

    with open(cmake_file, 'w', encoding='utf-8') as stream:
        stream.write(content.replace(legacy, updated, 1))


def _default_mkl_blas_lapack_libraries():
    if bglb.mkl_library_dir == '':
        return ''

    candidates = [
        os.path.join(bglb.mkl_library_dir, 'libmkl_gf_lp64.so'),
        os.path.join(bglb.mkl_library_dir, 'libmkl_intel_lp64.so'),
    ]
    interface_lib = next((path for path in candidates if os.path.exists(path)), '')
    if interface_lib == '':
        return ''

    thread_candidates = [
        os.path.join(bglb.mkl_library_dir, 'libmkl_gnu_thread.so'),
        os.path.join(bglb.mkl_library_dir, 'libmkl_intel_thread.so'),
        os.path.join(bglb.mkl_library_dir, 'libmkl_sequential.so'),
    ]
    thread_lib = next((path for path in thread_candidates if os.path.exists(path)), '')
    if thread_lib == '':
        return ''

    core_lib = os.path.join(bglb.mkl_library_dir, 'libmkl_core.so')
    if not os.path.exists(core_lib):
        return ''

    libraries = [interface_lib, thread_lib, core_lib]
    if os.path.basename(thread_lib) == 'libmkl_gnu_thread.so':
        libraries.extend(['gomp', 'pthread', 'm', 'dl'])
    elif os.path.basename(thread_lib) == 'libmkl_intel_thread.so':
        libraries.extend(['iomp5', 'pthread', 'm', 'dl'])
    else:
        libraries.extend(['pthread', 'm', 'dl'])

    return ';'.join(libraries)


def cmake_make_scalapack():
    '''
    Build ScaLAPACK (Scalable Linear Algebra PACKage) using CMake.

    ScaLAPACK is required by MUMPS for parallel distributed linear algebra.
    Source: https://github.com/Reference-ScaLAPACK/scalapack

    Requires:
    - MPI (C and Fortran)
    - BLAS and LAPACK
    - Fortran compiler (mpif90)
    '''
    if bglb.verbose:
        print("Building ScaLAPACK")

    path = os.path.join(extdir, 'scalapack')
    if not bglb.dry_run and not os.path.exists(path):
        assert False, "scalapack source not found – run download('scalapack') first"

    # ScaLAPACK's BLACS install probe configures a nested legacy CMake project
    # that CMake 4.x rejects unless its minimum version is raised.
    _patch_scalapack_blacs_cmake(path)

    cmbuild = os.path.join(path, 'cmbuild')
    if os.path.exists(cmbuild):
        print("working directory already exists!")
    else:
        os.makedirs(cmbuild)

    pwd = chdir(cmbuild)

    cmake_opts = {
        'DBUILD_SHARED_LIBS': '1',
        'DCMAKE_INSTALL_PREFIX': bglb.scalapack_prefix,
        'DCMAKE_C_COMPILER': bglb.mpicc_command,
        'DCMAKE_Fortran_COMPILER': bglb.mpifort_command,
        'DCMAKE_BUILD_TYPE': 'Release',
        # PyPI's CMake 4.x rejects ScaLAPACK's legacy cmake_minimum_required(3.2)
        # unless a policy floor is provided explicitly.
        'DCMAKE_POLICY_VERSION_MINIMUM': '3.5',
        'DSCALAPACK_BUILD_TESTS': 'OFF',
        # ScaLAPACK legacy Fortran code has rank mismatches that GCC >= 10
        # treats as errors by default. This flag downgrades them to warnings.
        'DCMAKE_Fortran_FLAGS': '-fallow-argument-mismatch',
    }

    auto_mkl_libs = ''
    if ((bglb.enable_mkl_pardiso or bglb.enable_mkl_cpardiso) and
            bglb.blas_libraries == '' and bglb.lapack_libraries == ''):
        auto_mkl_libs = _default_mkl_blas_lapack_libraries()

    # Pass explicit BLAS/LAPACK paths if known
    if bglb.blas_libraries != "":
        cmake_opts['DBLAS_LIBRARIES'] = bglb.blas_libraries
    elif auto_mkl_libs != '':
        cmake_opts['DBLAS_LIBRARIES'] = auto_mkl_libs
    if bglb.lapack_libraries != "":
        cmake_opts['DLAPACK_LIBRARIES'] = bglb.lapack_libraries
    elif auto_mkl_libs != '':
        cmake_opts['DLAPACK_LIBRARIES'] = auto_mkl_libs

    if sys.platform in ("linux", "linux2"):
        cmake_opts['DCMAKE_INSTALL_RPATH'] = '$ORIGIN'
        cmake_opts['DCMAKE_BUILD_WITH_INSTALL_RPATH'] = '1'
    elif sys.platform == "darwin":
        cmake_opts['DCMAKE_MACOSX_RPATH'] = 'YES'
        cmake_opts['DCMAKE_INSTALL_NAME_DIR'] = '@rpath'

    if bglb.verbose:
        cmake_opts['DCMAKE_VERBOSE_MAKEFILE'] = '1'

    cmake('..', **cmake_opts)
    make('scalapack')
    make_install('scalapack')

    os.chdir(pwd)
