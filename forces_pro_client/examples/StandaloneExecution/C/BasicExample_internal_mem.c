/*
 * (C) embotech AG, Zurich, Switzerland, 2013-2023. All rights reserved.
 *
 * This file is part of the FORCESPRO client, and carries the same license.
 */
#include "FORCESNLPsolver/include/FORCESNLPsolver.h"

/* additional header for internal memory functionality */
#include "FORCESNLPsolver/include/FORCESNLPsolver_memory.h"

#include <stdio.h>

/*
 * Internal memory interface
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

    /* Handle to the solver memory */
    FORCESNLPsolver_mem * mem_handle;

    int exit_code;
    int return_val = 0;
    int i;

    params.xinit[0] = -4.;
    params.xinit[1] = 2.;

    /* Get i-th memory buffer */
    i = 0;
    mem_handle = FORCESNLPsolver_internal_mem(i);
    /* Note: number of available memory buffers is controlled by code option max_num_mem */

    /* Check that memory is in valid state */
    if (mem_handle == NULL)
    {
        printf("Solver has no internal memory. Regenerate solver with codeoptions.max_num_mem > 0. Exiting.\n");
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
