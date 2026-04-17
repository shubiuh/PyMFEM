apt-get update
apt-get install -y python3-venv
# apt-get install -y openmpi-bin libopenmpi-dev # if mpi is mising use openmpi

/volume/volume1/pymfem_cpu_dev/PyMFEM
python3 -m venv .venv
. .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel mpi4py matplotlib
# parallel version using dev branch (not work yet as mfem v4.9 has updated the cpp source code)
# python -m pip install . -C"no-serial=Yes" -C"with-parallel=Yes" -C"with-gslib=Yes" -C"mfem-branch=dev" --verbose

# parallel version using release v4.8.3
python -m pip install . -C"no-serial=Yes" -C"with-parallel=Yes" -C"with-gslib=Yes" -C"mfem-branch=5f1afe5a6513429758f2cd3f6bf37b5e978fda6c" --verbose

# parallel version using branch v48
python -m pip install . -C"no-serial=Yes" -C"with-parallel=Yes" -C"with-gslib=Yes" -C"mfem-branch=v48" --verbose

# Then, generate swig wrappers, using the swig option, together with skip-ext, so that external libraies are not rebuild.
# python -m pip install . -C"with-parallel=Yes" -C"skip-ext=Yes"  -C"swig=Yes" --verbose

# If you are not happy with the wrapper (*.cxx and *.py), you edit *.i and redo the same. 
# When you are happy, build the wrapper with skip-swig and skip-ext.
# python -m pip install . -C"with-parallel=Yes" -C"skip-ext=Yes"  -C"skip-swig=Yes" --verbose