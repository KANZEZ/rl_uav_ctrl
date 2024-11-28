import numpy
import ctypes

name = "FORCES_NLP_solver"
requires_callback = True
lib = "lib/libFORCES_NLP_solver.so"
lib_static = "lib/libFORCES_NLP_solver.a"
c_header = "include/FORCES_NLP_solver.h"
nstages = 21

# Parameter             | Type    | Scalar type      | Ctypes type    | Numpy type   | Shape     | Len
params = \
[("xinit"               , "dense" , ""               , ctypes.c_double, numpy.float64, (  6,   1),    6),
 ("x0"                  , "dense" , ""               , ctypes.c_double, numpy.float64, (168,   1),  168),
 ("all_parameters"      , "dense" , ""               , ctypes.c_double, numpy.float64, ( 21,   1),   21),
 ("reinitialize"        , "dense" , "FORCES_NLP_solver_int", ctypes.c_int   , numpy.int32  , (  1,   1),    1)]

# Output                | Type    | Ctypes type    | Numpy type   | Shape     | Len
outputs = \
[("x01"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x02"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x03"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x04"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x05"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x06"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x07"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x08"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x09"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x10"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x11"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x12"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x13"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x14"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x15"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x16"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x17"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x18"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x19"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x20"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x21"                 , ""               , ctypes.c_double, numpy.float64,     (  8,),    8)]

# Info Struct Fields
info = \
[('it', ctypes.c_int),
 ('res_eq', ctypes.c_double),
 ('rsnorm', ctypes.c_double),
 ('pobj', ctypes.c_double),
 ('solvetime', ctypes.c_double),
 ('fevalstime', ctypes.c_double),
 ('QPtime', ctypes.c_double),
 ('QPit', ctypes.c_int),
 ('QPexitflag', ctypes.c_int),
 ('solver_id', ctypes.c_int * 8)
]

# Dynamics dimensions
#   nvar    |   neq   |   dimh    |   dimp    |   diml    |   dimu    |   dimhl   |   dimhu    
dynamics_dims = [
	(8, 6, 0, 1, 8, 8, 0, 0), 
	(8, 6, 0, 1, 8, 8, 0, 0), 
	(8, 6, 0, 1, 8, 8, 0, 0), 
	(8, 6, 0, 1, 8, 8, 0, 0), 
	(8, 6, 0, 1, 8, 8, 0, 0), 
	(8, 6, 0, 1, 8, 8, 0, 0), 
	(8, 6, 0, 1, 8, 8, 0, 0), 
	(8, 6, 0, 1, 8, 8, 0, 0), 
	(8, 6, 0, 1, 8, 8, 0, 0), 
	(8, 6, 0, 1, 8, 8, 0, 0), 
	(8, 6, 0, 1, 8, 8, 0, 0), 
	(8, 6, 0, 1, 8, 8, 0, 0), 
	(8, 6, 0, 1, 8, 8, 0, 0), 
	(8, 6, 0, 1, 8, 8, 0, 0), 
	(8, 6, 0, 1, 8, 8, 0, 0), 
	(8, 6, 0, 1, 8, 8, 0, 0), 
	(8, 6, 0, 1, 8, 8, 0, 0), 
	(8, 6, 0, 1, 8, 8, 0, 0), 
	(8, 6, 0, 1, 8, 8, 0, 0), 
	(8, 6, 0, 1, 8, 8, 0, 0), 
	(8, 6, 0, 1, 8, 8, 0, 0)
]