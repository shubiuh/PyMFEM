# ----------------------------------------------------------------------------------------
# Routines for ParMETIS
# ----------------------------------------------------------------------------------------
import sys
import os
import shutil
from sys import platform

__all__ = ["cmake_make_parmetis"]

from build_utils import *
from build_consts import *

import build_globals as bglb


def cmake_make_parmetis():
    '''
    Build ParMETIS using CMake.

    ParMETIS is the parallel version of METIS and is required by MUMPS
    for parallel matrix ordering (+parmetis variant in spack).

    IMPORTANT: ParMETIS 4.0.3's CMakeLists.txt requires the full METIS
    *source* directory (not the install prefix) at METIS_PATH, because it
    directly includes GKlib/GKlibSystem.cmake and compiles libmetis from
    source alongside libparmetis.

    The already-built libmetis.so from bglb.metis_prefix is still used at
    runtime; CMake just needs the source tree to build libparmetis.so.
    '''
    if bglb.verbose:
        print("Building ParMETIS")

    path = os.path.join(extdir, 'parmetis')
    if not bglb.dry_run and not os.path.exists(path):
        assert False, "parmetis source not found – run download('parmetis') first"

    # ParMETIS needs the METIS *source* tree (not the install prefix).
    # Use realpath to avoid `..` components that confuse CMake's symlink creation.
    metis_src = os.path.realpath(os.path.join(extdir, 'metis'))
    if not bglb.dry_run and not os.path.exists(
            os.path.join(metis_src, 'GKlib', 'GKlibSystem.cmake')):
        assert False, (
            "ParMETIS requires METIS source at: " + metis_src +
            "\nMake sure METIS was downloaded (not just installed).")

    cmbuild = os.path.join(path, 'cmbuild')
    if os.path.exists(cmbuild):
        print("cleaning existing ParMETIS build directory")
        shutil.rmtree(cmbuild)
    os.makedirs(cmbuild)

    pwd = chdir(cmbuild)

    # ParMETIS's CMakeLists.txt runs:
    #   execute_process(COMMAND cmake -E create_symlink ${METIS_PATH} metis)
    # in the SOURCE directory to create a `metis` symlink. This fails when
    # a `metis` directory already exists in the source tree (which is the case
    # in the 4.0.3 tarball) because cmake -E create_symlink cannot replace a
    # real directory. Pre-remove and pre-create the symlink ourselves so cmake's
    # execute_process becomes a no-op (it will overwrite an existing symlink).
    metis_link = os.path.join(path, 'metis')
    if os.path.islink(metis_link):
        os.remove(metis_link)
    elif os.path.isdir(metis_link):
        shutil.rmtree(metis_link)
    os.symlink(metis_src, metis_link)

    cmake_opts = {
        'DBUILD_SHARED_LIBS': '1',
        'DCMAKE_INSTALL_PREFIX': bglb.parmetis_prefix,
        'DCMAKE_C_COMPILER': bglb.mpicc_command,
        'DCMAKE_BUILD_TYPE': 'Release',
        # PyPI's CMake 4.x rejects ParMETIS's legacy cmake_minimum_required(2.8)
        # unless a policy floor is provided explicitly.
        'DCMAKE_POLICY_VERSION_MINIMUM': '3.5',
        # Must point to METIS source tree, not install prefix
        'DMETIS_PATH': metis_src,
        'DGKLIB_PATH': os.path.join(metis_src, 'GKlib'),
        'DSHARED': '1',
    }

    if sys.platform == "darwin":
        cmake_opts['DCMAKE_MACOSX_RPATH'] = 'YES'
        cmake_opts['DCMAKE_INSTALL_NAME_DIR'] = '@rpath'
    elif sys.platform in ("linux", "linux2"):
        cmake_opts['DCMAKE_INSTALL_RPATH'] = '$ORIGIN'
        cmake_opts['DCMAKE_BUILD_WITH_INSTALL_RPATH'] = '1'

    if bglb.verbose:
        cmake_opts['DCMAKE_VERBOSE_MAKEFILE'] = '1'

    cmake('..', **cmake_opts)
    make('parmetis')
    make_install('parmetis')

    if platform == "darwin":
        lib = os.path.join(bglb.parmetis_prefix, 'lib', 'libparmetis.dylib')
        make_call(['install_name_tool', '-id',
                   os.path.join('@rpath', 'libparmetis.dylib'), lib])

    os.chdir(pwd)
