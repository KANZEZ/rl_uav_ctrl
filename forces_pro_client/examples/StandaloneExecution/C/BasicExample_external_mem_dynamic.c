/*
 * (C) embotech AG, Zurich, Switzerland, 2013-2023. All rights reserved.
 *
 * This file is part of the FORCESPRO client, and carries the same license.
 */

#include "FORCESNLPsolver/include/FORCESNLPsolver.h"
#include <stdio.h>
#include <stdlib.h>

/*
 * External memory interface, dynamically allocated memory
 * Preconditions:
 * - codeoptions.max_num_mem = 0
 *   (recommended, disables internal memory)
 * - MEM_SIZE >= FORCESNLPsolver_get_mem_size()
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

    /* Memory buffer allocated by the user of type char (representing bytes) */
    char * mem;

    /* Handle to the solver memory */
    FORCESNLPsolver_mem * mem_handle;

    int exit_code;
    int return_val = 0;

    size_t mem_size;
    int i;

    params.xinit[0] = -4.;
    params.xinit[1] = 2.;

    /* Required memory size in bytes */
    mem_size = FORCESNLPsolver_get_mem_size();

    /* Dynamically allocate memory buffer */
    mem = malloc(mem_size);

    /* Cast memory buffer to solver memory
     * note: i can be set to 0 if no thread safety required */
    i = 0;
    mem_handle = FORCESNLPsolver_external_mem(mem, i, mem_size);

    /* Check that memory is in valid state */
    if (mem_handle == NULL)
    {
        printf("Received invalid solver memory. Exiting.\n");
        return_val = 1;
        return return_val;
    }

    exit_code = FORCESNLPsolver_solve(&params, &output, &info, mem_handle, NULL);

    /* Free user-allocated memory */
    free(mem);

    if (exit_code != 1)
    {
        printf("FORCESNLPsolver did not return optimal solution. Exiting.\n");
        return_val = 1;
    }

    return return_val;
}
