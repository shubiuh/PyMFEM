# ----------------------------------------------------------------------------------------
# Routines for HDF5
# ----------------------------------------------------------------------------------------

import os

__all__ = ["cmake_make_hdf5"]

from build_utils import *
from build_consts import *

import build_globals as bglb


def cmake_make_hdf5(serial=True):
    """
    Build and install HDF5 with CMake.

    Installs into:
      - <hdf5_prefix>/serial   (non-MPI build)
      - <hdf5_prefix>/openmpi  (MPI build)
    """
    if bglb.verbose:
        print("Building hdf5", "serial" if serial else "parallel")

    cmbuild = "cmbuild_ser" if serial else "cmbuild_par"
    build_dir = os.path.join(extdir, "hdf5", cmbuild)
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)

    install_prefix = os.path.join(
        bglb.hdf5_prefix,
        "serial" if serial else "openmpi",
    )

    cmake_opts = {
        "DCMAKE_INSTALL_PREFIX": install_prefix,
        "DBUILD_SHARED_LIBS": "1",
        "DBUILD_TESTING": "0",
        "DHDF5_BUILD_EXAMPLES": "0",
        "DHDF5_BUILD_HL_LIB": "1",
        "DHDF5_BUILD_CPP_LIB": "1" if serial else "0",
        "DHDF5_BUILD_TOOLS": "0",
        "DHDF5_ENABLE_PARALLEL": "0" if serial else "1",
        "DCMAKE_POSITION_INDEPENDENT_CODE": "1",
    }

    if bglb.verbose:
        cmake_opts["DCMAKE_VERBOSE_MAKEFILE"] = "1"

    if serial:
        cmake_opts["DCMAKE_C_COMPILER"] = bglb.cc_command
        cmake_opts["DCMAKE_CXX_COMPILER"] = bglb.cxx_command
    else:
        cmake_opts["DCMAKE_C_COMPILER"] = bglb.mpicc_command
        cmake_opts["DCMAKE_CXX_COMPILER"] = bglb.mpicxx_command

    pwd = chdir(build_dir)

    cmake("..", **cmake_opts)
    make("hdf5_" + ("serial" if serial else "parallel"))
    make_install("hdf5_" + ("serial" if serial else "parallel"))

    os.chdir(pwd)
