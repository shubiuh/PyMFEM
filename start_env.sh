. .venv/bin/activate
echo "Python virtual environment activated for PyMFEM. Please start your work..."

# run the mfem examples
cd external/mfem/cmbuild_par/examples

export LD_LIBRARY_PATH=/volume/volume1/pymfem_cpu_dev/PyMFEM_dev/.venv/lib/python3.12/site-packages/mfem/external/par/lib:/volume/volume1/pymfem_cpu_dev/PyMFEM_dev/.venv/lib/python3.12/site-packages/mfem/external/lib:/volume/volume1/pymfem_cpu_dev/PyMFEM_dev/external/intel/oneapi/mkl/2025.3/lib:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=/volume//PyMFEM_dev/.venv/lib/python3.12/site-packages/mfem/external/par/lib:/volume/PyMFEM_dev/.venv/lib/python3.12/site-packages/mfem/external/lib:/volume//PyMFEM_dev/external/intel/oneapi/mkl/2025.3/lib:$LD_LIBRARY_PATH

mpirun --allow-run-as-root -np 4 ex6p -m ../data/star-hilbert.mesh -o 2

export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH