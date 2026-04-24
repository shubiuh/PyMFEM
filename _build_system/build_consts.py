# ----------------------------------------------------------------------------------------
# Global build constant parameters
# ----------------------------------------------------------------------------------------
from sys import platform
import os
from shutil import which as find_command
from collections import namedtuple

__all__ = ["swig_command", "rootdir", "extdir",
           "REPOS", "dylibext", "osx_sysroot"]

# ----------------------------------------------------------------------------------------
#  package directory
# ----------------------------------------------------------------------------------------
rootdir = os.path.realpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))
extdir = os.path.join(rootdir, 'external')
if not os.path.exists(extdir):
    os.mkdir(os.path.join(rootdir, 'external'))

# ----------------------------------------------------------------------------------------
# Platform dependency
# ----------------------------------------------------------------------------------------

osx_sysroot = ''
dylibext = '.so'

if platform == "linux" or platform == "linux2":
    dylibext = '.so'

elif platform == "darwin":
    # OS X
    dylibext = '.dylib'
    import sysconfig
    for i, x in enumerate(sysconfig.get_config_vars()['CFLAGS'].split()):
        if x == '-isysroot':
            osx_sysroot = sysconfig.get_config_vars()['CFLAGS'].split()[i+1]
            break

elif platform == "win32":
    # Windows...
    assert False, "Windows is not supported yet. Contribution is welcome"

# ----------------------------------------------------------------------------------------
# SWIG
# ----------------------------------------------------------------------------------------

swig_command = (find_command('swig') if os.getenv("SWIG") is None
                else os.getenv("SWIG"))
if swig_command is None:
    assert False, "SWIG is not installed (hint: pip install swig)"

# ----------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------

release = namedtuple('Release', ['version', 'hash', 'tarball'])
REPOS = dict(
    # mfem=dict(
    #     url="https://github.com/mfem/mfem.git",
    #     # version, hash, tarball
    #     releases=[
    #         release("4.7", "dc9128ef596e84daf1138aa3046b826bba9d259f", None),
    #         release("4.8", "a01719101027383954b69af1777dc828bf795d62", None),
    #         release("4.9", "d9d6526cc1749980a2ba1da16e2c1ca1e07d82ec", None),
    #     ]
    # ),
    mfem=dict(
        url="https://github.com/shubiuh/mfem.git",
        # version, hash, tarball
        releases=[
            release("4.8.2", "da4e1a215b9f9473b3b534c96f6010b077b0b36c", None),
            release("4.8.3", "5f1afe5a6513429758f2cd3f6bf37b5e978fda6c", None),
            release("4.9", "8faf89ed6fd5cc63b8b66f54a4c98710972f6add", None),
        ]
    ),
    metis=dict(
        url="https://github.com/KarypisLab/METIS",
        releases=[
            release("5.1.0", "94c03a6e2d1860128c2d0675cbbb86ad4f261256",
                    "https://github.com/mfem/tpls/raw/gh-pages/metis-5.1.0.tar.gz"),
        ]
    ),
    gklib=dict(
        url="https://github.com/KarypisLab/GKlib",
        releases=[
            release("5.1.1", "a7f8172703cf6e999dd0710eb279bba513da4fec",
                    "https://github.com/KarypisLab/GKlib/archive/refs/tags/METIS-v5.1.1-DistDGL-0.5.tar.gz"),
        ]
    ),
    libceed=dict(
        url="https://github.com/CEED/libCEED.git",
        releases=[
            release(
                "0.12.0", None, "https://github.com/CEED/libCEED/archive/refs/tags/v0.12.0.tar.gz"),
        ]
    ),
    hypre=dict(
        url=None,
        releases=[
            release(
                "2.28.0", None, "https://github.com/hypre-space/hypre/archive/v2.28.0.tar.gz"),
            release(
                "2.32.0", None, "https://github.com/hypre-space/hypre/archive/v2.32.0.tar.gz"),
        ]
    ),
    gslib=dict(
        url=None,
        releases=[
            release(
                "1.0.8", None, "https://github.com/Nek5000/gslib/archive/refs/tags/v1.0.8.tar.gz"),
            release(
                "1.0.9", None, "https://github.com/Nek5000/gslib/archive/refs/tags/v1.0.9.tar.gz"),
        ]
    ),
    parmetis=dict(
        url="https://github.com/KarypisLab/ParMETIS",
        releases=[
            release(
                "4.0.3", None, "https://github.com/mfem/tpls/raw/gh-pages/parmetis-4.0.3.tar.gz"),
        ]
    ),
    scalapack=dict(
        url="https://github.com/Reference-ScaLAPACK/scalapack",
        releases=[
            release(
                "2.2.0", None, "https://github.com/Reference-ScaLAPACK/scalapack/archive/refs/tags/v2.2.0.tar.gz"),
            release(
                "2.2.1", None, "https://github.com/Reference-ScaLAPACK/scalapack/archive/refs/tags/v2.2.1.tar.gz"),
        ]
    ),
    mumps=dict(
        url="https://github.com/scivision/mumps-superbuild",
        releases=[
            release(
                "5.7.3.1", None, "https://github.com/scivision/mumps-superbuild/archive/refs/tags/v5.7.3.1.tar.gz"),
            release(
                "5.8.0.0", None, "https://github.com/scivision/mumps-superbuild/archive/refs/tags/v5.8.0.0.tar.gz"),
            release(
                "5.8.1.0", None, "https://github.com/scivision/mumps-superbuild/archive/refs/tags/v5.8.1.0.tar.gz"),
            release(
                "5.8.2.2", None, "https://github.com/scivision/mumps-superbuild/archive/refs/tags/v5.8.2.2.tar.gz"),
        ]
    ),
    suitesparse=dict(
        url="https://github.com/DrTimothyAldenDavis/SuiteSparse",
        releases=[
            release(
                "7.8.3", None, "https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/refs/tags/v7.8.3.tar.gz"),
        ]
    ),
    hdf5=dict(
        url="https://github.com/HDFGroup/hdf5",
        releases=[
            release(
                "1.14.6", None, "https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5_1.14.6.tar.gz"),
        ]
    ),
)
