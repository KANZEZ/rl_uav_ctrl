/*
 * Sample main function for compiling a standalone FORCESPRO solver for 
 * controlling an overhead crane model.
 * 
 * (C) embotech AG, Zurich, Switzerland, 2013-2023. All rights reserved.
 *
 * This file is part of the FORCESPRO client, and carries the same license.
 */

#include <stdio.h>
#include "crane.h"
#include "CraneSolver/include/CraneSolver.h"
#include "CraneSolver/include/CraneSolver_memory.h"

#ifdef __cplusplus
extern "C" {
#endif

extern solver_int32_default CraneSolver_adtool2forces(double *x, double *y, double *l, double *p, double *f, double *nabla_f, double *c, double *nabla_c, double *h, double *nabla_h, double *H, int stage, int iterations, int threadID);
CraneSolver_extfunc callback = &CraneSolver_adtool2forces;

#ifdef __cplusplus
} /* extern "C" */
#endif


int main(int argc, char *argv[])
{
    int i, exitflag;
    static CraneSolver_params myparams;
    static CraneSolver_output myoutput;
    static CraneSolver_info myinfo;

    /* COPY PARAMETER DATA */
    exitflag = 0;
    for( i=0; i<200; i++ )
    {
        myparams.x0[i] = x0[i];
    }
    for( i=0; i<8; i++ )
    {
        myparams.xinit[i] = xinit[i];
    }
    for( i=0; i<40; i++ )
    {
        myparams.all_parameters[i] = all_parameters[i];
    }
    
    /* CALL SOLVER */
    exitflag = CraneSolver_solve(&myparams, &myoutput, &myinfo, CraneSolver_internal_mem(0), stdout, callback);

    printf("FORCESPRO exited with exitflag %d, in %d iterations.\n",  exitflag, myinfo.it);
    
    if (exitflag == 1)
    {
        printf("Solution found.\n");
        return 0;
    }
    else if (exitflag == 0)
    {
        printf("Solver reached maximum number of iterations.\n");
        return 1;
    }
    else
    {
        printf("Solver failed to solve problem.\n");
        return -1;
    }
}
