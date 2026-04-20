//
// Copyright (c) 2020-2025, Princeton Plasma Physics Laboratory, All rights reserved.
//
%module(package="mfem._par", directors="0") pardiso
%feature("autodoc", "1");

%{
#include <mpi.h>
#include "config/config.hpp"
#include "linalg/pardiso.hpp"
#include "numpy/arrayobject.h"
#include "../common/pyoperator.hpp"
%}

%init %{
import_array1(-1);
%}

%include "../common/mfem_config.i"

#ifdef MFEM_USE_MPI
%include mpi4py/mpi4py.i
%mpi4py_typemap(Comm, MPI_Comm);
#endif

%include "exception.i"
%import "array.i"
%import "vector.i"
%import "operators.i"
%import "densemat.i"
%import "sparsemat.i"
%import "../common/exception.i"

%include "linalg/pardiso.hpp"