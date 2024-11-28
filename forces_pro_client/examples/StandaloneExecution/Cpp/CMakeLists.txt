# Sample CMAKE configuration for compiling a standalone FORCESPRO solver for 
# controlling an overhead crane model.
# 
# (C) embotech AG, Zurich, Switzerland, 2013-2023. All rights reserved.
#
# This file is part of the FORCESPRO client, and carries the same license.

cmake_minimum_required(VERSION 3.1)

if(NOT DEFINED SOLVERNAME)
    set(SOLVERNAME "CraneSolver")
endif ()
if(NOT DEFINED DEBUG)
    set(DEBUG 0)
endif ()
if(NOT DEFINED LIB_DIR)
    if (UNIX)
        set(LIB_DIR "lib_target")
    else ()
        set(LIB_DIR "lib")
    endif (UNIX)
endif ()
if(NOT DEFINED EXT_FUNC_SOURCE)
    set(EXT_FUNC_SOURCE CraneSolver_adtool2forces.c CraneSolver_casadi.c)
endif ()
if(NOT DEFINED MAIN_SOURCE)
    set(MAIN_SOURCE crane.cpp)
endif ()


project(${SOLVERNAME}_standalone)

set(CMAKE_C_STANDARDS 11)
set(CMAKE_CXX_STANDARDS C++11)

# debug flags
if (UNIX)
    if (DEBUG)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0 -mavx")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -mavx")
    else ()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -mavx")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx")
    endif (DEBUG)
endif ()

if (NOT DEFINED MAIN_SOURCE)
    file(GLOB MAIN_SOURCE ${CMAKE_SOURCE_DIR}/*.cpp)
endif()

set(ADDLIBDIR)
set(ADDLIB)

if (UNIX)
    set(ADDLIB ${ADDLIB} m)
endif()

# add solver either as a library
set(ADDLIBDIR ${ADDLIBDIR} ${CMAKE_SOURCE_DIR}/${SOLVERNAME}/${LIB_DIR})
if (WIN32 AND FORCES_CLIENT)
    set(ADDLIBDIR ${ADDLIBDIR} ${FORCES_CLIENT}/libs_Intel/win64) 
endif (WIN32 AND FORCES_CLIENT)
find_library (SOLVER_LIB ${SOLVERNAME} HINTS ${CMAKE_SOURCE_DIR}/${SOLVERNAME}/${LIB_DIR})
set(ADDLIB ${SOLVER_LIB} ${ADDLIB})

if (DEFINED EXT_FUNC_SOURCE)
    set(ADDSRC ${ADDSRC} ${EXT_FUNC_SOURCE})
endif()

set(ADDSRC ${ADDSRC} ${MAIN_SOURCE})

if (PARALLEL)
    find_package(OpenMP)
    if (OpenMP_C_FOUND)
        set(ADDLIB OpenMP::OpenMP_C ${ADDLIB})
    endif()
endif()

# include headers into project
include_directories(${CMAKE_SOURCE_DIR}/${SOLVERNAME}/include
                    ${CMAKE_SOURCE_DIR}/)

link_directories(${ADDLIBDIR})

add_executable(${SOLVERNAME}_standalone ${ADDSRC})

target_link_libraries(${SOLVERNAME}_standalone ${ADDLIB})

install(TARGETS ${SOLVERNAME}_standalone RUNTIME DESTINATION ${CMAKE_SOURCE_DIR})
