"""
Helper functions for setup.py
"""

import os
import sys
import configparser
from urllib import request
import itertools
import site
import re
import subprocess
import multiprocessing
import ssl
import tarfile
import shutil
from collections import namedtuple
from shutil import which as find_command

__all__ = ["print_config",
           "initialize_cmd_options",
           "cmd_options",
           "process_cmd_options",
           "configure_build",
           "clean_dist_info",
           ]

from build_utils import *
from build_consts import *
import build_globals as bglb


def _default_mkl_prefix():
    mklroot = os.getenv('MKLROOT', '').strip()
    if mklroot != '':
        return os.path.abspath(mklroot)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, 'external', 'intel', 'oneapi', 'mkl', 'latest')


def _default_mkl_library_dir(prefix):
    candidates = [
        os.path.join(prefix, 'lib', 'x86_64-linux-gnu'),
        os.path.join(prefix, 'lib', 'intel64'),
        os.path.join(prefix, 'lib'),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


def _default_mkl_compiler_dir(prefix):
    prefix_parent = os.path.dirname(os.path.dirname(prefix))
    candidates = [
        os.path.join(prefix_parent, 'compiler', 'latest', 'lib', 'intel64_lin'),
        os.path.join(prefix_parent, 'compiler', 'latest', 'lib'),
        '/usr/lib/x86_64-linux-gnu',
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


def _default_mkl_mpi_wrapper_lib():
    return os.getenv('MKL_MPI_WRAPPER_LIB', 'mkl_blacs_openmpi_lp64').strip()


def _default_mkl_include_dir(prefix):
    candidates = [
        os.path.join(prefix, 'include', 'mkl'),
        os.path.join(prefix, 'include'),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


def print_config():
    print("----configuration----")
    print(" prefix", bglb.prefix)
    print(" when needed, the dependency (mfem/hypre/metis) will be installed under " +
          bglb.ext_prefix)
    print(" build mfem : " + ("Yes" if bglb.build_mfem else "No"))
    print(" build miniapps: " + ("Yes" if bglb.mfem_miniapps else "No"))
    print(" build metis : " + ("Yes" if bglb.build_metis else "No"))
    print(" build hypre : " + ("Yes" if bglb.build_hypre else "No"))
    print(" build mumps : " + ("Yes" if bglb.build_mumps else "No"))
    print(" build suitesparse : " + ("Yes" if bglb.build_suitesparse else "No"))
    print(" build libceed : " + ("Yes" if bglb.build_libceed else "No"))
    print(" build gslib : " + ("Yes" if bglb.build_gslib else "No"))
    print(" call SWIG wrapper generator: " +
          ("Yes" if bglb.run_swig else "No"))
    print(" build serial wrapper: " + ("Yes" if bglb.build_serial else "No"))
    print(" build parallel wrapper : " +
          ("Yes" if bglb.build_parallel else "No"))

    print(" hypre prefix", bglb.hypre_prefix)
    print(" metis prefix", bglb.metis_prefix)
    if bglb.enable_parmetis:
        print(" parmetis prefix", bglb.parmetis_prefix)
    if bglb.enable_scalapack:
        print(" scalapack prefix", bglb.scalapack_prefix)
    if bglb.enable_mumps:
        print(" mumps prefix", bglb.mumps_prefix)
    if bglb.enable_suitesparse:
        print(" suitesparse prefix", bglb.suitesparse_prefix)
    if bglb.enable_mkl_pardiso:
        print(" mkl pardiso prefix", bglb.mkl_pardiso_prefix)
        print(" mkl library dir", bglb.mkl_library_dir)
        print(" mkl compiler dir", bglb.mkl_compiler_dir)
    if bglb.enable_mkl_cpardiso:
        print(" mkl cpardiso prefix", bglb.mkl_cpardiso_prefix)
        print(" mkl library dir", bglb.mkl_library_dir)
        print(" mkl mpi wrapper lib", bglb.mkl_mpi_wrapper_lib)
    print(f" c compiler : {bglb.cc_command}")
    print(f" c++ compiler : {bglb.cxx_command}")
    print(f" mpi-c compiler : {bglb.mpicc_command}")
    print(f" mpi-c++ compiler : {bglb.mpicxx_command}")

    print(" verbose : " + ("Yes" if bglb.verbose else "No"))
    print(f" SWIG : {swig_command}")

    if bglb.blas_libraries != "":
        print(" BLAS libraries : " + bglb.blas_libraries)
    if bglb.lapack_libraries != "":
        print(" Lapack libraries : " + bglb.lapack_libraries)

    print("")


def clean_dist_info(wheeldir):
    if not os.path.isdir(wheeldir):
        return
    for x in os.listdir(wheeldir):
        if x.endswith(".dist-info"):
            fname = os.path.join(wheeldir, x)
            print("!!! removing existing ", fname)
            shutil.rmtree(fname)


def initialize_cmd_options(command_obj):
    command_obj.prefix = ''

    command_obj.swig = False
    command_obj.skip_swig = False
    command_obj.ext_only = False

    command_obj.git_sshclone = False
    command_obj.skip_ext = False
    command_obj.with_parallel = False
    command_obj.no_serial = False
    command_obj.mfem_prefix = ''
    command_obj.mfems_prefix = ''
    command_obj.mfemp_prefix = ''
    command_obj.mfem_source = bglb.mfem_source
    command_obj.mfem_branch = ''
    command_obj.mfem_debug = False
    command_obj.mfem_miniapps = True
    command_obj.metis_prefix = ''
    command_obj.hypre_prefix = ''

    command_obj.with_cuda = False
    command_obj.with_cuda_hypre = False
    command_obj.cuda_arch = None
    command_obj.with_metis64 = False

    command_obj.with_pumi = False
    command_obj.pumi_prefix = ''

    command_obj.with_strumpack = False
    command_obj.strumpack_prefix = ''

    command_obj.with_mumps = True
    command_obj.mumps_prefix = ''
    command_obj.with_mkl_pardiso = False
    command_obj.mkl_pardiso_prefix = ''
    command_obj.with_mkl_cpardiso = False
    command_obj.mkl_cpardiso_prefix = ''
    command_obj.mkl_library_dir = ''
    command_obj.mkl_compiler_dir = ''
    command_obj.mkl_mpi_wrapper_lib = ''
    command_obj.mkl_include_dir = ''

    command_obj.with_parmetis = False
    command_obj.parmetis_prefix = ''

    command_obj.with_scalapack = False
    command_obj.scalapack_prefix = ''

    command_obj.with_suitesparse = False
    command_obj.suitesparse_prefix = ''

    command_obj.with_lapack = False
    command_obj.blas_libraries = ""
    command_obj.lapack_libraries = ""

    command_obj.with_libceed = False
    command_obj.libceed_prefix = ''
    command_obj.libceed_only = False

    command_obj.with_gslib = False
    command_obj.gslib_prefix = ''
    command_obj.gslib_only = False

    command_obj.CC = ''
    command_obj.CXX = ''
    command_obj.MPICC = ''
    command_obj.MPICXX = ''
    command_obj.vv = False

    command_obj.unverifiedSSL = False


cmd_options = [
    ('vv', None, 'More verbose output (CMAKE_VERBOSE_MAKEFILE etc)'),
    ('prefix=', None, 'Install prefix'),
    ('with-parallel', None, 'Installed both serial and parallel version'),
    ('no-serial', None, 'Skip building the serial wrapper'),
    ('mfem-prefix=', None, 'Specify locaiton of mfem' +
     'libmfem.so must exits under <mfem-prefix>/lib. ' +
     'This mode uses clean-swig + run-swig, unless mfem-prefix-no-swig is on'),
    ('mfemp-prefix=', None, 'Specify locaiton of parallel mfem ' +
     'libmfem.so must exits under <mfemp-prefix>/lib. ' +
     'Need to use it with mfem-prefix'),
    ('mfems-prefix=', None, 'Specify locaiton of serial mfem ' +
     'libmfem.so must exits under <mfems-prefix>/lib. ' +
     'Need to use it with mfem-prefix'),
    ('mfem-branch=', None, 'Specify branch of mfem' +
     'MFEM is cloned and built using the specfied branch '),
    ('mfem-source=', None, 'Specify mfem source location' +
     'MFEM source directory. Required to run-swig '),
    ('mfem-debug', None, 'Build MFME with MFEM_DEBUG enabled'),
    ('mfem-miniapps', None, 'build MFME Miniapps'),
    ('hypre-prefix=', None, 'Specify locaiton of hypre' +
     'libHYPRE.so must exits under <hypre-prefix>/lib'),
    ('metis-prefix=', None, 'Specify locaiton of metis' +
     'libmetis.so must exits under <metis-prefix>/lib'),
    ('git-sshclone', None, 'Use SSH for git clone' +
     'try if default git clone using https fails (need Github account and setting for SSH)'),
    ('swig', None, 'Run Swig and exit'),
    ('skip-swig', None,
     'Skip running swig (used when wrapper is generated for the MFEM C++ library to be used'),
    ('ext-only', None, 'Build metis, hypre, mfem(C++) only'),
    ('skip-ext', None, 'Skip building metis, hypre, mfem(C++) only'),
    ('CC=', None, 'c compiler'),
    ('CXX=', None, 'c++ compiler'),
    ('MPICC=', None, 'mpic compiler'),
    ('MPICXX=', None, 'mpic++ compiler'),
    ('unverifiedSSL', None, 'use unverified SSL context for downloading'),
    ('with-cuda', None, 'enable cuda'),
    ('with-cuda-hypre', None, 'enable cuda in hypre'),
    ('cuda-arch=', None, 'set cuda compute capability. Ex if A100, set to 80'),
    ('with-metis64', None, 'use 64bit int in metis'),
    ('with-pumi', None, 'enable pumi (parallel only)'),
    ('pumi-prefix=', None, 'Specify locaiton of pumi'),
    ('with-suitesparse', None,
     'build MFEM with SuiteSparse (MFEM_USE_SUITESPARSE=YES; enables UMFPackSolver, KLU, CHOLMOD etc.)'),

    ('suitesparse-prefix=', None,
     'Specify locaiton of suitesparse (=SuiteSparse_DIR)'),
    ('with-libceed', None, 'enable libceed'),
    ('libceed-prefix=', None, 'Specify locaiton of libceed'),
    ('libceed-only', None, 'Build libceed only'),
    ('gslib-prefix=', None, 'Specify locaiton of gslib'),
    ('with-gslib', None, 'enable gslib'),
    ('gslib-only', None, 'Build gslib only'),
    ('with-strumpack', None, 'enable strumpack (parallel only)'),
    ('strumpack-prefix=', None, 'Specify locaiton of strumpack'),
    ('with-mumps', None, 'enable mumps (parallel only)'),
    ('mumps-prefix=', None, 'Specify locaiton of mumps'),
    ('with-mkl-pardiso', None, 'enable Intel MKL Pardiso solver support'),
    ('mkl-pardiso-prefix=', None, 'Specify location of Intel oneMKL for Pardiso'),
    ('with-mkl-cpardiso', None, 'enable Intel MKL Cluster Pardiso solver support (parallel only)'),
    ('mkl-cpardiso-prefix=', None, 'Specify location of Intel oneMKL for Cluster Pardiso'),
    ('mkl-library-dir=', None, 'Specify full path to the Intel oneMKL library directory'),
    ('mkl-compiler-dir=', None, 'Specify full path to the Intel compiler runtime library directory'),
    ('mkl-mpi-wrapper-lib=', None, 'Specify the MKL BLACS MPI wrapper library name, e.g. mkl_blacs_openmpi_lp64'),
    ('mkl-include-dir=', None, 'Specify full path to the Intel oneMKL include directory'),
    ('with-parmetis', None, 'enable parmetis (used by mumps for parallel ordering)'),
    ('parmetis-prefix=', None, 'Specify location of parmetis'),
    ('with-scalapack', None, 'enable scalapack (required by mumps)'),
    ('scalapack-prefix=', None, 'Specify location of scalapack'),
    ('with-lapack', None, 'build MFEM with lapack'),
    ('blas-libraries=', None, 'Specify locaiton of Blas library (used to build MFEM)'),
    ('lapack-libraries=', None,
     'Specify locaiton of Lapack library (used to build MFEM)'),
]


def process_cmd_options(command_obj, cfs):
    '''
    called when install workflow is used
    '''
    cc = cfs.pop("CC", "")
    if cc != "":
        command_obj.cc_command = cc

    cc = cfs.pop("CXX", "")
    if cc != "":
        command_obj.cxx_command = cc

    cc = cfs.pop("MPICC", "")
    if cc != "":
        command_obj.mpicc_command = cc

    cc = cfs.pop("MPICXX", "")
    if cc != "":
        command_obj.mpicxx_command = cc

    for item in cmd_options:
        param, _none, hit = item
        attr = "_".join(param.split("-"))

        if param.endswith("="):
            param = param[:-1]
            attr = attr[:-1]
            value = cfs.pop(param, "")
            if value != "":
                if not hasattr(command_obj, attr):
                    assert False, str(command_obj) + " does not have " + attr
                setattr(command_obj, attr, value)
        else:
            if not hasattr(command_obj, attr):
                assert False, str(command_obj) + " does not have " + attr

            if getattr(command_obj, attr):
                value = cfs.pop(param, "Yes")
            else:
                value = cfs.pop(param, "No")

            if value.upper() in ("YES", "TRUE", "1"):
                setattr(command_obj, attr, True)
            else:
                setattr(command_obj, attr, False)

    if len(cfs) != 0:
        assert False, "unknonw input is given " + str(cfs)


def process_setup_options(command_obj, args):
    for item in args:
        if item.startswith('--'):
            item = item[2:]
        if item.startswith('-'):
            item = item[1:]

        if len(item.split('=')) == 2:
            param = item.split('=')[0]
            value = item.split('=')[1]
        else:
            param = item.strip()
            value = True
        attr = "_".join(param.split("-"))

        setattr(command_obj, attr, value)


def configure_install(self):
    '''
    called when install workflow is used

    '''
    if sys.argv[0] == 'setup.py' and sys.argv[1] == 'install':
        process_setup_options(self, sys.argv[2:])
    else:
        if bglb.verbose:
            print("!!!!!!!!  command-line input (pip): ", bglb.cfs)
        process_cmd_options(self, bglb.cfs)

    bglb.verbose = bool(self.vv) if not bglb.verbose else bglb.verbose
    if bglb.dry_run:
        bglb.verbose = True

    bglb.git_sshclone = bool(self.git_sshclone)

    bglb.mfem_source = abspath(self.mfem_source)

    bglb.skip_ext = bool(self.skip_ext)
    bglb.skip_swig = bool(self.skip_swig)

    bglb.swig_only = bool(self.swig)
    bglb.ext_only = bool(self.ext_only)

    bglb.metis_64 = bool(self.with_metis64)
    bglb.enable_pumi = bool(self.with_pumi)
    bglb.enable_strumpack = bool(self.with_strumpack)
    bglb.enable_mumps = bool(self.with_mumps)
    bglb.enable_mkl_pardiso = bool(self.with_mkl_pardiso)
    bglb.enable_mkl_cpardiso = bool(self.with_mkl_cpardiso)
    bglb.enable_parmetis = bool(self.with_parmetis)
    bglb.enable_scalapack = bool(self.with_scalapack)
    bglb.enable_cuda = bool(self.with_cuda)
    bglb.enable_cuda_hypre = bool(self.with_cuda_hypre)
    if self.cuda_arch is not None:
        bglb.cuda_arch = self.cuda_arch
    bglb.enable_libceed = bool(self.with_libceed)
    bglb.libceed_only = bool(self.libceed_only)
    bglb.enable_gslib = bool(self.with_gslib)
    bglb.gslib_only = bool(self.gslib_only)
    bglb.enable_suitesparse = bool(self.with_suitesparse)
    bglb.enable_lapack = bool(self.with_lapack)

    # controlls PyMFEM parallel
    bglb.build_parallel = bool(self.with_parallel)
    bglb.build_serial = not bool(self.no_serial)

    bglb.clean_swig = True
    bglb.run_swig = True
    bglb.run_swig_parallel = bool(self.with_parallel)

    bglb.mfem_debug = bool(self.mfem_debug)
    bglb.mfem_miniapps = bool(self.mfem_miniapps)

    if bglb.build_serial:
        bglb.build_serial = (not bglb.swig_only and not bglb.ext_only)

    if bglb.build_parallel:
        try:
            import mpi4py
        except ImportError:
            assert False, "Can not import mpi4py"

    if bglb.enable_mkl_cpardiso and not bglb.build_parallel:
        assert False, "with-mkl-cpardiso requires with-parallel"

    if self.mfem_prefix != '':
        bglb.mfem_prefix = abspath(self.mfem_prefix)
        bglb.mfems_prefix = abspath(self.mfem_prefix)
        bglb.mfemp_prefix = abspath(self.mfem_prefix)
        if self.mfems_prefix != '':
            bglb.mfems_prefix = abspath(self.mfems_prefix)
        if self.mfemp_prefix != '':
            bglb.mfemp_prefix = abspath(self.mfemp_prefix)

        check = find_libpath_from_prefix('mfem', bglb.mfems_prefix)
        assert check != '', "libmfem.so is not found in the specified <path>/lib"
        check = find_libpath_from_prefix('mfem', bglb.mfemp_prefix)
        assert check != '', "libmfem.so is not found in the specified <path>/lib"

        bglb.mfem_outside = True
        bglb.build_mfem = False
        hypre_prefix = bglb.mfem_prefix
        metis_prefix = bglb.mfem_prefix

        if bglb.swig_only:
            bglb.clean_swig = False

    else:
        bglb.mfem_outside = False
        bglb.build_mfem = True
        bglb.build_mfemp = bglb.build_parallel
        bglb.build_hypre = bglb.build_parallel
        bglb.build_metis = bglb.build_parallel or bglb.enable_suitesparse

        if bglb.ext_prefix == '':
            bglb.ext_prefix = external_install_prefix(bglb.prefix)
        bglb.hypre_prefix = os.path.join(bglb.ext_prefix)
        bglb.metis_prefix = os.path.join(bglb.ext_prefix)

        bglb.mfem_prefix = bglb.ext_prefix
        bglb.mfems_prefix = os.path.join(bglb.ext_prefix, 'ser')
        bglb.mfemp_prefix = os.path.join(bglb.ext_prefix, 'par')
        # enable_gslib = True

    if self.mfem_branch != '':
        bglb.mfem_branch = self.mfem_branch

    if self.hypre_prefix != '':
        check = find_libpath_from_prefix('HYPRE', self.hypre_prefix)
        assert check != '', "libHYPRE.so is not found in the specified <path>/lib or lib64"
        hypre_prefix = os.path.expanduser(self.hypre_prefix)
        build_hypre = False

    if self.metis_prefix != '':
        check = find_libpath_from_prefix('metis', self.metis_prefix)
        assert check != '', "libmetis.so is not found in the specified <path>/lib or lib64"
        bglb.metis_prefix = os.path.expanduser(self.metis_prefix)
        bglb.build_metis = False

    if bglb.enable_libceed or bglb.libceed_only:
        if self.libceed_prefix != '':
            bglb.libceed_prefix = os.path.expanduser(self.libceed_prefix)
            bglb.build_libceed = False
        else:
            bglb.libceed_prefix = bglb.mfem_prefix
            bglb.build_libceed = True

    if bglb.enable_gslib or bglb.gslib_only:
        if self.gslib_prefix != '':
            bglb.build_gslib = False
            bglb.gslibs_prefix = os.path.expanduser(self.gslib_prefix)
            bglb.gslibp_prefix = os.path.expanduser(self.gslib_prefix)
        else:
            bglb.gslibs_prefix = bglb.mfems_prefix
            bglb.gslibp_prefix = bglb.mfemp_prefix
            bglb.build_gslib = True

    if bglb.enable_suitesparse:
        if self.suitesparse_prefix != '':
            bglb.suitesparse_prefix = abspath(self.suitesparse_prefix)
            bglb.build_suitesparse = False
        else:
            bglb.suitesparse_prefix = bglb.mfem_prefix
            bglb.build_suitesparse = True

    if self.pumi_prefix != '':
        bglb.pumi_prefix = abspath(self.pumi_prefix)
    else:
        bglb.pumi_prefix = bglb.mfem_prefix

    if self.strumpack_prefix != '':
        bglb.strumpack_prefix = abspath(self.strumpack_prefix)
    else:
        bglb.strumpack_prefix = bglb.mfem_prefix

    if self.mumps_prefix != '':
        bglb.mumps_prefix = abspath(self.mumps_prefix)
    else:
        bglb.mumps_prefix = bglb.mfem_prefix

    if self.mkl_pardiso_prefix != '':
        bglb.mkl_pardiso_prefix = abspath(self.mkl_pardiso_prefix)
    else:
        bglb.mkl_pardiso_prefix = _default_mkl_prefix()

    if self.mkl_cpardiso_prefix != '':
        bglb.mkl_cpardiso_prefix = abspath(self.mkl_cpardiso_prefix)
    elif bglb.mkl_pardiso_prefix != '':
        bglb.mkl_cpardiso_prefix = bglb.mkl_pardiso_prefix
    else:
        bglb.mkl_cpardiso_prefix = _default_mkl_prefix()

    if self.mkl_library_dir != '':
        bglb.mkl_library_dir = abspath(self.mkl_library_dir)
    else:
        base_prefix = bglb.mkl_cpardiso_prefix if bglb.enable_mkl_cpardiso else bglb.mkl_pardiso_prefix
        bglb.mkl_library_dir = _default_mkl_library_dir(base_prefix)

    if self.mkl_compiler_dir != '':
        bglb.mkl_compiler_dir = abspath(self.mkl_compiler_dir)
    else:
        bglb.mkl_compiler_dir = _default_mkl_compiler_dir(bglb.mkl_pardiso_prefix)

    if self.mkl_mpi_wrapper_lib != '':
        bglb.mkl_mpi_wrapper_lib = self.mkl_mpi_wrapper_lib.strip()
    else:
        bglb.mkl_mpi_wrapper_lib = _default_mkl_mpi_wrapper_lib()

    if self.mkl_include_dir != '':
        bglb.mkl_include_dir = abspath(self.mkl_include_dir)
    else:
        base_prefix = bglb.mkl_cpardiso_prefix if bglb.enable_mkl_cpardiso else bglb.mkl_pardiso_prefix
        bglb.mkl_include_dir = _default_mkl_include_dir(base_prefix)

    if self.parmetis_prefix != '':
        bglb.parmetis_prefix = abspath(self.parmetis_prefix)
    else:
        bglb.parmetis_prefix = bglb.metis_prefix  # install alongside metis

    if bglb.enable_parmetis:
        if self.parmetis_prefix != '':
            bglb.build_parmetis = False
        else:
            bglb.build_parmetis = bglb.build_parallel

    if self.scalapack_prefix != '':
        bglb.scalapack_prefix = abspath(self.scalapack_prefix)
    else:
        bglb.scalapack_prefix = bglb.metis_prefix  # install alongside other math libs

    if bglb.enable_scalapack:
        if self.scalapack_prefix != '':
            bglb.build_scalapack = False
        else:
            bglb.build_scalapack = bglb.build_parallel

    if bglb.enable_mumps:
        if self.mumps_prefix != '':
            bglb.build_mumps = False
        else:
            bglb.build_mumps = bglb.build_parallel
            # Auto-enable parmetis and scalapack when building mumps from source
            # (matches spack spec: +parmetis)
            if not bglb.enable_parmetis:
                bglb.enable_parmetis = True
                bglb.build_parmetis = bglb.build_parallel
                bglb.parmetis_prefix = bglb.metis_prefix
            if not bglb.enable_scalapack:
                bglb.enable_scalapack = True
                bglb.build_scalapack = bglb.build_parallel
                bglb.scalapack_prefix = bglb.metis_prefix

    if bglb.enable_cuda:
        nvcc = find_command('nvcc')
        assert nvcc is not None, "nvcc not found but with-cuda was requested"
        bglb.cuda_prefix = os.path.dirname(os.path.dirname(nvcc))

    if self.CC != '':
        bglb.cc_command = self.CC
    if self.CXX != '':
        bglb.cxx_command = self.CXX
    if self.MPICC != '':
        bglb.mpicc_command = self.MPICC
    if self.MPICXX != '':
        bglb.mpicxx_command = self.MPICXX

    if self.blas_libraries != "":
        bglb.blas_libraries = self.blas_libraries
    if self.lapack_libraries != "":
        bglb.lapack_libraries = self.lapack_libraries

    if bglb.swig_only:
        bglb.build_serial = False
        bglb.build_parallel = False
        bglb.clean_swig = False
        bglb.keep_temp = True
        bglb.skip_ext = True

    if bglb.skip_ext:
        bglb.build_metis = False
        bglb.build_hypre = False
        bglb.build_parmetis = False
        bglb.build_scalapack = False
        bglb.build_mumps = False
        bglb.build_mfem = False
        bglb.build_mfemp = False
        bglb.build_libceed = False
        bglb.build_gslib = False
        bglb.keep_temp = True

    if bglb.skip_swig:
        bglb.clean_swig = False
        bglb.run_swig = False
        bglb.keep_temp = True

    if bglb.ext_only:
        bglb.clean_swig = False
        bglb.run_swig = False
        bglb.build_serial = False
        bglb.build_parallel = False
        bglb.keep_temp = True

    if bglb.libceed_only:
        bglb.clean_swig = False
        bglb.run_swig = False
        bglb.build_mfem = False
        bglb.build_mfemp = False
        bglb.build_metis = False
        bglb.build_hypre = False
        bglb.build_mumps = False
        bglb.build_gslib = False
        bglb.build_serial = False
        bglb.build_parallel = False
        bglb.build_libceed = True
        bglb.keep_temp = True

    if bglb.gslib_only:
        bglb.clean_swig = False
        bglb.run_swig = False
        bglb.build_mfem = False
        bglb.build_mfemp = False
        bglb.build_metis = False
        bglb.build_hypre = False
        bglb.build_mumps = False
        bglb.build_serial = False
        bglb.build_libceed = False
        bglb.build_gslib = True
        bglb.keep_temp = True

    bglb.is_configured = True


configure_build = configure_install
