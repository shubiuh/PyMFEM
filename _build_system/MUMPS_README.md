# MUMPS Support in PyMFEM

## Overview
MUMPS (MUltifrontal Massively Parallel sparse direct Solver) is a parallel sparse direct solver that can be used with MFEM for solving linear systems.

PyMFEM now uses the **scivision/mumps-superbuild** project which provides CMake support and can automatically build dependencies like ScaLAPACK.

## Quick Start

### Automatic Build (Recommended) ✅

```bash
# Clean any previous build
rm -rf build/ *.egg-info external/mumps external/MUMPS

# Build with MUMPS (will auto-download and build)
python -m pip install . -C"with-parallel=Yes" \
                         -C"with-gslib=Yes" \
                         -C"with-mumps=Yes" \
                         -C"mfem-branch=v48" --verbose
```

**What gets built automatically:**
- ✅ MUMPS 5.8.2.2 (from scivision/mumps-superbuild)
- ✅ ScaLAPACK (auto-built if not found on system)  
- ✅ PORD (included with MUMPS)

**What you need to provide:**
- ✅ C/C++ toolchain (`gcc`, `g++`, `make`)
- ✅ Python build environment (`python3`, `python3-venv`, `python3-dev`)
- ✅ BLAS/LAPACK (system packages usually sufficient)
- ✅ MPI with Fortran support (`mpif90` or `mpifort`)
- ✅ Fortran compiler (`gfortran`)

## Requirements

### Install System Dependencies

These are the system packages needed for the full PyMFEM + MFEM + MUMPS build,
not just the MUMPS solver itself.

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y build-essential git \
                        python3 python3-dev python3-venv \
                        gfortran \
                        libopenmpi-dev openmpi-bin \
                        libblas-dev liblapack-dev
```

**RHEL/CentOS:**
```bash
sudo yum install -y gcc gcc-c++ gcc-gfortran git \
                    python3 python3-devel \
                    openmpi openmpi-devel \
                    blas-devel lapack-devel
```

### Python Build Packages

The isolated `pip install .` build will automatically install these from
[pyproject.toml](/volume/volume1/pymfem_cpu_dev/PyMFEM/pyproject.toml):

- `setuptools>=80.0.1`
- `pip>=25.1.0`
- `numpy>=2.0.0`
- `cmake>=4.0.0`
- `swig>=4.3`

For the parallel Python wrappers, install `mpi4py` in your active virtualenv
before building:

```bash
python -m pip install --upgrade pip setuptools wheel mpi4py
```

If your environment cannot install binary wheels for CMake or SWIG from PyPI,
install `cmake` and `swig` from your system package manager instead.

### Environment Variables

```bash
export MPICC=mpicc
export MPICXX=mpic++
export MPIFORT=mpif90  # or mpifort
```

## Building Options

### Option 1: Automatic Build (NEW) ✨

Uses scivision/mumps-superbuild with CMake - builds everything automatically:

```bash
python -m pip install . -C"with-parallel=Yes" -C"with-mumps=Yes"
```

### Option 2: Use Spack

```bash
spack install mumps+mpi+double
export MUMPS_DIR=$(spack location -i mumps)
python -m pip install . -C"with-parallel=Yes" \
                         -C"with-mumps=Yes" \
                         -C"mumps-prefix=$MUMPS_DIR"
```

### Option 3: Use Pre-installed MUMPS

```bash
python -m pip install . -C"with-parallel=Yes" \
                         -C"with-mumps=Yes" \
                         -C"mumps-prefix=/path/to/mumps"
```

## Version Control

Available MUMPS versions (in `_build_system/build_consts.py`):
- **v5.8.2.2** (latest, default) ← Recommended
- v5.8.1.0
- v5.8.0.0
- v5.7.3.1

The build system uses the latest version by default.

## Testing

After installation, test MUMPS functionality:

```bash
cd examples
mpirun -n 4 python ex_maxwell_dipole.py --use-mumps
```

## Troubleshooting

### "mpifort not found"
```bash
# Ubuntu/Debian
sudo apt install libopenmpi-dev gfortran

# RHEL/CentOS
sudo yum install openmpi-devel gcc-gfortran
```

### "Cannot find BLAS/LAPACK"
```bash
# Ubuntu/Debian
sudo apt install libblas-dev liblapack-dev

# RHEL/CentOS
sudo yum install blas-devel lapack-devel
```

### CMake version too old
MUMPS superbuild requires CMake >= 3.20:
```bash
cmake --version
# If too old: pip install cmake
```

### Build fails
Check you have all required dependencies:
```bash
which gcc g++ make python3 mpicc mpic++ mpif90 gfortran
```

## What's New

The new scivision/mumps-superbuild provides:
- ✅ **CMake support** - No more manual Makefile configuration!
- ✅ **Auto-build ScaLAPACK** - Builds automatically if not found
- ✅ **Better portability** - Works on Linux and macOS
- ✅ **RPATH support** - Proper runtime library paths
- ✅ **Modern build** - Supports MUMPS 5.2.x through 5.8.x

## Resources

- **MUMPS superbuild**: https://github.com/scivision/mumps-superbuild
- **MUMPS official**: http://mumps.enseeiht.fr/
- **MFEM documentation**: https://mfem.org
