/*
 * (C) embotech AG, Zurich, Switzerland, 2013-2023. All rights reserved.
 *
 * This file is part of the FORCESPRO client, and carries the same license.
 */
#include "FORCESNLPsolver/include/FORCESNLPsolver.h"

/* additional header for internal memory functionality */
#include "FORCESNLPsolver/include/FORCESNLPsolver_memory.h"

#include <stdio.h>
#include <omp.h>

/* number of solvers to be run */
#define NUM_SOLVERS 4
/* maximum no of threads to run solvers concurrently */
#define NUM_THREADS 2

/*
 * Thread safe solver using internal memory within a multithreaded environment
 * Preconditions:
 * - codeoptions.threadSafeStorage = 1
 * - codeoptions.max_num_mem >= NUM_THREADS
 */

/*
 * Instructions:
 * 1) run example examples/Python/HighLevelInterface/BasicExample with required codeoptions (see above)
 * 2) copy generated FORCESNLPsolver to this directory
 * 3) compile this file with OpenMP enabled and link against solver library in FORCESNLPsolver/lib
*/

int main()
{
    FORCESNLPsolver_params params[NUM_SOLVERS];
    FORCESNLPsolver_info info[NUM_SOLVERS];
    FORCESNLPsolver_output output[NUM_SOLVERS];

    /* Each thread must be assigned its own memory space */
    FORCESNLPsolver_mem * mem_handle[NUM_THREADS];

    int exit_code[NUM_SOLVERS];
    int return_val = 0;

    int i, i_thread;
    int i_solver;

    omp_set_num_threads(NUM_THREADS);

    params[0].xinit[0] = -4.;
    params[0].xinit[1] = 2.;
    params[1].xinit[0] = -3.;
    params[1].xinit[1] = 2.;
    params[2].xinit[0] = -1.;
    params[2].xinit[1] = 1.;
    params[3].xinit[0] = -3.;
    params[3].xinit[1] = 1.;

    /* Get memory buffer for each thread */
    for (i=0; i<NUM_THREADS; i++)
    {
        /* Get i-th memory */
        mem_handle[i] = FORCESNLPsolver_internal_mem(i);

        /* Check that memory is in valid state */
        if (mem_handle[i] == NULL)
        {
            printf("Solver has less internal memory instances than NUM_THREADS. Regenerate solver with codeoptions.max_num_mem >= %i. Exiting.\n", NUM_THREADS);
            return_val = 1;
            return return_val;
        }

    }

    /* Parallel call to the solver using OpenMP */
#pragma omp parallel for
    for (i_solver=0; i_solver<NUM_SOLVERS; i_solver++)
    {
        int i_thread = omp_get_thread_num();
        printf("solver %i assigned to thread %i\n", i_solver, i_thread);
        exit_code[i_solver] = FORCESNLPsolver_solve(&params[i_solver], &output[i_solver], &info[i_solver], mem_handle[i_thread], NULL);
    }

    for (i_solver=0; i_solver<NUM_SOLVERS; i_solver++)
    {
        if (exit_code[i_solver] != 1)
        {
            printf("FORCESNLPsolver did not return optimal solution for solver %i. Exiting.\n", i_solver);
            return_val = 1;
        }
    }

    return return_val;
}
