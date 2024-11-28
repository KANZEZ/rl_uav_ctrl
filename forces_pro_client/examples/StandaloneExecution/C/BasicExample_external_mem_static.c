/*
 * (C) embotech AG, Zurich, Switzerland, 2013-2023. All rights reserved.
 *
 * This file is part of the FORCESPRO client, and carries the same license.
 */

#include "FORCESNLPsolver/include/FORCESNLPsolver.h"
#include <stdio.h>

/* memory size in bytes s.t. MEM_SIZE >= FORCESNLPsolver_get_mem_size(): */
#ifndef MEM_SIZE
#define MEM_SIZE 14200
#endif

/*
 * External memory interface
 * Preconditions:
 * - codeoptions.max_num_mem = 0
 *   (recommended, disables internal memory)
 * - memory size MEM_SIZE for static allocation must be an integer constant
 *   expression so that MEM_SIZE >= FORCESNLPsolver_get_mem_size().
 *
 *   The minimum required MEM_SIZE is system and compiler dependent.
 *   Instructions to get required value of MEM_SIZE on any device: compile and
 *   run FORCESNLPsolver/interface/FORCESNLPsolver_get_mem_size.c (requires
 *   linking against solver library) and set MEM_SIZE to the output value.
 *   This is equivalent to the return value of the library function
 *   FORCESNLPsolver_get_mem_size().
 */

/*
 * Instructions:
 * 1) run example examples/Python/HighLevelInterface/BasicExample with required codeoptions (see above)
 * 2) copy generated FORCESNLPsolver to this directory
 * 3) compile this file and link against solver library in FORCESNLPsolver/lib
*/

int main()
{
    FORCESNLPsolver_params params;

    FORCESNLPsolver_info info;
    FORCESNLPsolver_output output;
    FORCESNLPsolver_mem * mem_handle;

    int exit_code;
    int return_val = 0;
    int i;

    /* Statically allocated memory buffer of size MEM_SIZE bytes */
    static char mem[MEM_SIZE];

    /* i can be set to 0 if no thread safety required */
    i = 0;
    mem_handle = FORCESNLPsolver_external_mem(mem, i, MEM_SIZE);

    params.xinit[0] = -4.;
    params.xinit[1] = 2.;

    /* Check that memory is in valid state */
    if (mem_handle == NULL)
    {
        printf("Provided memory was of insufficient size. Set MEM_SIZE to %u. Exiting.\n", FORCESNLPsolver_get_mem_size());
        return_val = 1;
        return return_val;
    }

    exit_code = FORCESNLPsolver_solve(&params, &output, &info, mem_handle, NULL);

    if (exit_code != 1)
    {
        printf("FORCESNLPsolver did not return optimal solution. Exiting.\n");
        return_val = 1;
    }

    return return_val;
}
