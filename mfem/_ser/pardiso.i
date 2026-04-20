//
// Copyright (c) 2020-2025, Princeton Plasma Physics Laboratory, All rights reserved.
//
%module(package="mfem._ser", directors="0") pardiso
%feature("autodoc", "1");

%{
#include "config/config.hpp"
#include "linalg/pardiso.hpp"
#include "numpy/arrayobject.h"
#include "../common/pyoperator.hpp"
%}

%init %{
import_array1(-1);
%}

%include "../common/mfem_config.i"

%include "exception.i"
%import "array.i"
%import "vector.i"
%import "operators.i"
%import "densemat.i"
%import "sparsemat.i"
%import "../common/exception.i"

%include "linalg/pardiso.hpp"