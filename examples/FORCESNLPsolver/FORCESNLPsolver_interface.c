/*
 * AD tool to FORCESPRO Template - missing information to be filled in by createADTool.m 
 * (C) embotech AG, Zurich, Switzerland, 2013-2023. All rights reserved.
 *
 * This file is part of the FORCESPRO client, and carries the same license.
 */ 

#ifdef __cplusplus
extern "C" {
#endif

#include "include/FORCESNLPsolver.h"

#ifndef NULL
#define NULL ((void *) 0)
#endif

#include "FORCESNLPsolver_model.h"



/* copies data from sparse matrix into a dense one */
static void FORCESNLPsolver_sparse2fullcopy(solver_int32_default nrow, solver_int32_default ncol, const solver_int32_default *colidx, const solver_int32_default *row, FORCESNLPsolver_callback_float *data, FORCESNLPsolver_float *out)
{
    solver_int32_default i, j;
    
    /* copy data into dense matrix */
    for(i=0; i<ncol; i++)
    {
        for(j=colidx[i]; j<colidx[i+1]; j++)
        {
            out[i*nrow + row[j]] = ((FORCESNLPsolver_float) data[j]);
        }
    }
}




/* AD tool to FORCESPRO interface */
extern solver_int32_default FORCESNLPsolver_adtool2forces(FORCESNLPsolver_float *x,        /* primal vars                                         */
                                 FORCESNLPsolver_float *y,        /* eq. constraint multiplers                           */
                                 FORCESNLPsolver_float *l,        /* ineq. constraint multipliers                        */
                                 FORCESNLPsolver_float *p,        /* parameters                                          */
                                 FORCESNLPsolver_float *f,        /* objective function (scalar)                         */
                                 FORCESNLPsolver_float *nabla_f,  /* gradient of objective function                      */
                                 FORCESNLPsolver_float *c,        /* dynamics                                            */
                                 FORCESNLPsolver_float *nabla_c,  /* Jacobian of the dynamics (column major)             */
                                 FORCESNLPsolver_float *h,        /* inequality constraints                              */
                                 FORCESNLPsolver_float *nabla_h,  /* Jacobian of inequality constraints (column major)   */
                                 FORCESNLPsolver_float *hess,     /* Hessian (column major)                              */
                                 solver_int32_default stage,     /* stage number (0 indexed)                           */
                                 solver_int32_default iteration, /* iteration number of solver                         */
                                 solver_int32_default threadID   /* Id of caller thread                                */)
{
    /* AD tool input and output arrays */
    const FORCESNLPsolver_callback_float *in[4];
    FORCESNLPsolver_callback_float *out[7];
    

    /* Allocate working arrays for AD tool */
    
    FORCESNLPsolver_callback_float w[33];
	
    /* temporary storage for AD tool sparse output */
    FORCESNLPsolver_callback_float this_f = (FORCESNLPsolver_callback_float) 0.0;
    FORCESNLPsolver_float nabla_f_sparse[6];
    
    
    FORCESNLPsolver_float c_sparse[13];
    FORCESNLPsolver_float nabla_c_sparse[53];
    
    
    /* pointers to row and column info for 
     * column compressed format used by AD tool */
    solver_int32_default nrow, ncol;
    const solver_int32_default *colind, *row;
    
    /* set inputs for AD tool */
    in[0] = x;
    in[1] = p;
    in[2] = l;
    in[3] = y;

	if ((0 <= stage && stage <= 9))
	{
		
		
		out[0] = &this_f;
		out[1] = nabla_f_sparse;
		FORCESNLPsolver_objective_0(in, out, NULL, w, 0);
		if( nabla_f != NULL )
		{
			nrow = FORCESNLPsolver_objective_0_sparsity_out(1)[0];
			ncol = FORCESNLPsolver_objective_0_sparsity_out(1)[1];
			colind = FORCESNLPsolver_objective_0_sparsity_out(1) + 2;
			row = FORCESNLPsolver_objective_0_sparsity_out(1) + 2 + (ncol + 1);
				
			FORCESNLPsolver_sparse2fullcopy(nrow, ncol, colind, row, nabla_f_sparse, nabla_f);
		}
		
		out[0] = c_sparse;
		out[1] = nabla_c_sparse;
		FORCESNLPsolver_dynamics_0(in, out, NULL, w, 0);
		if( c != NULL )
		{
			nrow = FORCESNLPsolver_dynamics_0_sparsity_out(0)[0];
			ncol = FORCESNLPsolver_dynamics_0_sparsity_out(0)[1];
			colind = FORCESNLPsolver_dynamics_0_sparsity_out(0) + 2;
			row = FORCESNLPsolver_dynamics_0_sparsity_out(0) + 2 + (ncol + 1);
				
			FORCESNLPsolver_sparse2fullcopy(nrow, ncol, colind, row, c_sparse, c);
		}
		if( nabla_c != NULL )
		{
			nrow = FORCESNLPsolver_dynamics_0_sparsity_out(1)[0];
			ncol = FORCESNLPsolver_dynamics_0_sparsity_out(1)[1];
			colind = FORCESNLPsolver_dynamics_0_sparsity_out(1) + 2;
			row = FORCESNLPsolver_dynamics_0_sparsity_out(1) + 2 + (ncol + 1);
				
			FORCESNLPsolver_sparse2fullcopy(nrow, ncol, colind, row, nabla_c_sparse, nabla_c);
		}
	}
	if ((10 == stage))
	{
		
		
		out[0] = &this_f;
		out[1] = nabla_f_sparse;
		FORCESNLPsolver_objective_1(in, out, NULL, w, 0);
		if( nabla_f != NULL )
		{
			nrow = FORCESNLPsolver_objective_1_sparsity_out(1)[0];
			ncol = FORCESNLPsolver_objective_1_sparsity_out(1)[1];
			colind = FORCESNLPsolver_objective_1_sparsity_out(1) + 2;
			row = FORCESNLPsolver_objective_1_sparsity_out(1) + 2 + (ncol + 1);
				
			FORCESNLPsolver_sparse2fullcopy(nrow, ncol, colind, row, nabla_f_sparse, nabla_f);
		}
	}
    
    /* add to objective */
    if (f != NULL)
    {
        *f += ((FORCESNLPsolver_float) this_f);
    }

    return 0;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
