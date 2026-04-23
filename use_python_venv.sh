apt-get update
apt-get install -y python3-venv
# apt-get install -y openmpi-bin libopenmpi-dev # if mpi is mising use openmpi
# required for MUMPS build: BLAS, LAPACK, and Fortran compiler
apt-get install -y libblas-dev liblapack-dev gfortran

cd /volume/volume1/pymfem_cpu_dev/PyMFEM_dev
python3 -m venv .venv
. .venv/bin/activate

# Make Intel MKL shared libraries visible to the dynamic linker.
# The 'latest' symlink resolves to the installed version (e.g. 2025.3).
MKL_LIB=/volume/volume1/pymfem_cpu_dev/PyMFEM_dev/external/intel/oneapi/mkl/latest/lib
export LD_LIBRARY_PATH="${MKL_LIB}:${LD_LIBRARY_PATH:-}"

# intel OneMKL silently
# wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/6a17080f-f0de-41b9-b587-52f92512c59a/intel-onemkl-2025.3.1.11_offline.sh
# sudo sh ./intel-onemkl-2025.3.1.11_offline.sh -a -s --install-dir /volume/volume1/pymfem_cpu_dev/PyMFEM_dev/external/intel/oneapi --eula accept

ln -sfn /workspace/results /volume/volume1/pymfem_cpu_dev/PyMFEM_dev/examples/results 

python -m pip install --upgrade pip setuptools wheel mpi4py matplotlib glvis requests pytest
# parallel version using dev branch (not work yet as mfem v4.9 has updated the cpp source code)
# python -m pip install . -C"no-serial=Yes" -C"with-parallel=Yes" -C"with-gslib=Yes" -C"mfem-branch=dev" --verbose

# parallel version using release v4.8.3
# python -m pip install . -C"no-serial=Yes" -C"with-parallel=Yes" -C"with-gslib=Yes" -C"mfem-branch=5f1afe5a6513429758f2cd3f6bf37b5e978fda6c" --verbose

# parallel version using branch v48
# python -m pip install . -C"with-parallel=Yes" -C"with-gslib=Yes" -C"mfem-branch=v48" --verbose

# parallel version using branch v48, enable MUMPS
python -m pip install . -C"with-parallel=Yes" -C"with-gslib=Yes" -C"with-mumps=Yes" -C"mfem-branch=v48"

# Then, generate swig wrappers, using the swig option, together with skip-ext, so that external libraies are not rebuild.
python -m pip install . -C"with-parallel=Yes" -C"with-gslib=Yes" -C"with-mumps=Yes" -C"skip-ext=Yes"  -C"swig=Yes" -C"mfem-branch=v48" --verbose

# If you are not happy with the wrapper (*.cxx and *.py), you edit *.i and redo the same. 
# When you are happy, build the wrapper with skip-swig and skip-ext.
python -m pip install . -C"with-parallel=Yes" -C"with-gslib=Yes" -C"with-mumps=Yes" -C"skip-ext=Yes"  -C"skip-swig=Yes" -C"mfem-branch=v48" --verbose
# python -m pip install . -C"with-parallel=Yes" -C"skip-ext=Yes"  -C"skip-swig=Yes" -C"mfem-branch=v48" --verbose

# to run the examples use
# python3 ex0.py
# mpirun --allow-run-as-root -n 4 python3 ex0p.py -mumps

# build with pardiso and suitesparse
python -m pip install . \
  -C"with-parallel=Yes" \
  -C"with-gslib=Yes" \
  -C"with-mumps=Yes" \
  -C"with-mkl-pardiso=Yes" \
  -C"with-mkl-cpardiso=Yes" \
  -C"with-suitesparse=Yes" \
  -C"mfem-branch=v48" \
  -C"skip-ext=No" \
  -C"skip-swig=No" \
  --verbose

# PC with GPU, 
#The forwarding of ports 3000, 8000 and 8080 is needed for VS Code, GLVis and the websocket connection between them. 
# The --cap-add=SYS_PTRACE option is added to resolve MPI warnings.
docker run --gpus all --cap-add=SYS_PTRACE \
  -p 3000:3000 -p 8000:8000 -p 8080:8080 \
  --name ws_dev \
  -v D:/docker_ws:/workspace \
  -v docker_ws_volume:/volume \
  shubinuh/spack-hpc:v2.3.7

# cot-42
docker run --rm -it \
  -p 3000:3000 -p 8000:8000 -p 8080:8080 \
  -v docker_ws_volume:/volume \
  --name ws_dev \
  -v /export/home/mnle8/01_szeng/docker_ws:/docker_ws \
  shubinuh/spack-hpc:v2.3.7 bash && cd /docker_ws/PyMFEM_dev

export LD_LIBRARY_PATH=/docker_ws/PyMFEM_dev/external/intel/oneapi/mkl/2025.3/lib:$LD_LIBRARY_PATH

# use cluster pardiso (real-valued)
mpirun --allow-run-as-root -np 16 python3 ex0p.py --use-cpardiso --full-assembly -r 2 -o 3 -m inline-hex.mesh

# us real mumps
mpirun --allow-run-as-root -n 4 python3 ex0p.py -mumps -o 2 -r 3  -m inline-hex.mesh

# use complex mumps
mpirun --allow-run-as-root -n 4 python3 ex0p.py -cmumps -o 2 -r 3  -m inline-hex.mesh

# use complex mumps with paraview visualization and glvis visualization, and amr
mpirun --allow-run-as-root -n 4 python ex_maxwell_dipole_amr.py -cmumps -vis -amr 4 -o 2 -pv