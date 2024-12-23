/*
 * AD tool to FORCESPRO Template - missing information to be filled in by createADTool.m 
 * (C) embotech AG, Zurich, Switzerland, 2013-2023. All rights reserved.
 *
 * This file is part of the FORCESPRO client, and carries the same license.
 */ 

#ifdef __cplusplus
extern "C" {
#endif

#include "$SOLVER_HEADER_FILE$"

#ifndef NULL
#define NULL ((void *) 0)
#endif

$INCLUDES$
$EXTERN_FUNCTION$


/* copies data from sparse matrix into a dense one */
static void $SOLVER_NAME$_sparse2fullcopy(solver_int32_default nrow, solver_int32_default ncol, const solver_int32_default *colidx, const solver_int32_default *row, $SOLVER_NAME$_callback_float *data, $SOLVER_NAME$_float *out)
{
    solver_int32_default i, j;
    
    /* copy data into dense matrix */
    for(i=0; i<ncol; i++)
    {
        for(j=colidx[i]; j<colidx[i+1]; j++)
        {
            out[i*nrow + row[j]] = (($SOLVER_NAME$_float) data[j]);
        }
    }
}

$OUT_INDEXED_TRANSPOSE$
$X_U_IDS$

/* AD tool to FORCESPRO interface */
extern solver_int32_default $SOLVER_NAME$_adtool2forces($SOLVER_NAME$_float *x,        /* primal vars                                         */
                                 $SOLVER_NAME$_float *y,        /* eq. constraint multiplers                           */
                                 $SOLVER_NAME$_float *l,        /* ineq. constraint multipliers                        */
                                 $PARAM_TYPE$ *p,        /* parameters                                          */
                                 $SOLVER_NAME$_float *f,        /* objective function (scalar)                         */
                                 $SOLVER_NAME$_float *nabla_f,  /* gradient of objective function                      */
                                 $SOLVER_NAME$_float *c,        /* dynamics                                            */
                                 $SOLVER_NAME$_float *nabla_c,  /* Jacobian of the dynamics (column major)             */
                                 $SOLVER_NAME$_float *h,        /* inequality constraints                              */
                                 $SOLVER_NAME$_float *nabla_h,  /* Jacobian of inequality constraints (column major)   */
                                 $SOLVER_NAME$_float *hess,     /* Hessian (column major)                              */
                                 solver_int32_default stage,     /* stage number (0 indexed)                           */
                                 solver_int32_default iteration, /* iteration number of solver                         */
                                 solver_int32_default threadID   /* Id of caller thread                                */)
{
    /* AD tool input and output arrays */
    $CASADI_ARG_ARRAY$
    $CASADI_RES_ARRAY$
    $EXTRA_INTEGERS$

    /* Allocate working arrays for AD tool */
    $CASADI_IW_ARRAY$
    $CASADI_W_ARRAY$
	
    /* temporary storage for AD tool sparse output */
    $SOLVER_NAME$_callback_float this_f = ($SOLVER_NAME$_callback_float) 0.0;
    $nabla_f_sparse$
    $hdef$
    $hjacdef$
    $cdef$
    $cjacdef$
    $laghessdef$
    
    /* pointers to row and column info for 
     * column compressed format used by AD tool */
    solver_int32_default nrow, ncol;
    const solver_int32_default *colind, *row;
    
    $INPUT_ASSIGNMENT$

$SWITCH_BODY$
    
    /* add to objective */
    if (f != NULL)
    {
        *f += (($SOLVER_NAME$_float) this_f);
    }

    return 0;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
