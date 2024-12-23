/*
FORCES_NLP_solver : A fast customized optimization solver.

Copyright (C) 2013-2023 EMBOTECH AG [info@embotech.com]. All rights reserved.


This software is intended for simulation and testing purposes only. 
Use of this software for any commercial purpose is prohibited.

This program is distributed in the hope that it will be useful.
EMBOTECH makes NO WARRANTIES with respect to the use of the software 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE. 

EMBOTECH shall not have any liability for any damage arising from the use
of the software.

This Agreement shall exclusively be governed by and interpreted in 
accordance with the laws of Switzerland, excluding its principles
of conflict of laws. The Courts of Zurich-City shall have exclusive 
jurisdiction in case of any dispute.

*/

#include "mex.h"
#include "math.h"
#include "../include/FORCES_NLP_solver.h"
#include "../include/FORCES_NLP_solver_memory.h"
#ifndef SOLVER_STDIO_H
#define SOLVER_STDIO_H
#include <stdio.h>
#endif



/* copy functions */

void copyCArrayToM_double(double *src, double *dest, solver_int32_default dim) 
{
    solver_int32_default i;
    for( i = 0; i < dim; i++ ) 
    {
        *dest++ = (double)*src++;
    }
}

void copyMArrayToC_double(double *src, double *dest, solver_int32_default dim) 
{
    solver_int32_default i;
    for( i = 0; i < dim; i++ ) 
    {
        *dest++ = (double) (*src++) ;
    }
}

void copyMValueToC_double(double * src, double * dest)
{
	*dest = (double) *src;
}

/* copy functions */

void copyCArrayToM_FORCES_NLP_solver_int(FORCES_NLP_solver_int *src, double *dest, solver_int32_default dim) 
{
    solver_int32_default i;
    for( i = 0; i < dim; i++ ) 
    {
        *dest++ = (double)*src++;
    }
}

void copyMArrayToC_FORCES_NLP_solver_int(double *src, FORCES_NLP_solver_int *dest, solver_int32_default dim) 
{
    solver_int32_default i;
    for( i = 0; i < dim; i++ ) 
    {
        *dest++ = (FORCES_NLP_solver_int) (*src++) ;
    }
}

void copyMValueToC_FORCES_NLP_solver_int(double * src, FORCES_NLP_solver_int * dest)
{
	*dest = (FORCES_NLP_solver_int) *src;
}

/* copy functions */

void copyCArrayToM_solver_int32_default(solver_int32_default *src, double *dest, solver_int32_default dim) 
{
    solver_int32_default i;
    for( i = 0; i < dim; i++ ) 
    {
        *dest++ = (double)*src++;
    }
}

void copyMArrayToC_solver_int32_default(double *src, solver_int32_default *dest, solver_int32_default dim) 
{
    solver_int32_default i;
    for( i = 0; i < dim; i++ ) 
    {
        *dest++ = (solver_int32_default) (*src++) ;
    }
}

void copyMValueToC_solver_int32_default(double * src, solver_int32_default * dest)
{
	*dest = (solver_int32_default) *src;
}

/* copy functions */

void copyCArrayToM_FORCES_NLP_solver_float(FORCES_NLP_solver_float *src, double *dest, solver_int32_default dim) 
{
    solver_int32_default i;
    for( i = 0; i < dim; i++ ) 
    {
        *dest++ = (double)*src++;
    }
}

void copyMArrayToC_FORCES_NLP_solver_float(double *src, FORCES_NLP_solver_float *dest, solver_int32_default dim) 
{
    solver_int32_default i;
    for( i = 0; i < dim; i++ ) 
    {
        *dest++ = (FORCES_NLP_solver_float) (*src++) ;
    }
}

void copyMValueToC_FORCES_NLP_solver_float(double * src, FORCES_NLP_solver_float * dest)
{
	*dest = (FORCES_NLP_solver_float) *src;
}



extern solver_int32_default (FORCES_NLP_solver_float *x, FORCES_NLP_solver_float *y, FORCES_NLP_solver_float *l, FORCES_NLP_solver_float *p, FORCES_NLP_solver_float *f, FORCES_NLP_solver_float *nabla_f, FORCES_NLP_solver_float *c, FORCES_NLP_solver_float *nabla_c, FORCES_NLP_solver_float *h, FORCES_NLP_solver_float *nabla_h, FORCES_NLP_solver_float *hess, solver_int32_default stage, solver_int32_default iteration, solver_int32_default threadID);
FORCES_NLP_solver_extfunc pt2function_FORCES_NLP_solver = &;


/* Some memory for MEX function */
static FORCES_NLP_solver_params params;
static FORCES_NLP_solver_output output;
static FORCES_NLP_solver_info info;
static FORCES_NLP_solver_mem * mem;

/* Main MEX function */
void mexFunction( solver_int32_default nlhs, mxArray *plhs[], solver_int32_default nrhs, const mxArray *prhs[] )  
{
	/* file pointer for printing */
	FILE *fp = NULL;

	/* define variables */	
	mxArray *par;
	mxArray *outvar;
	const mxArray *PARAMS = prhs[0]; 
	double *pvalue;
	solver_int32_default i;
	solver_int32_default exitflag;
	const solver_int8_default *fname;
	const solver_int8_default *outputnames[21] = {"x01", "x02", "x03", "x04", "x05", "x06", "x07", "x08", "x09", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21"};
	const solver_int8_default *infofields[10] = { "it", "res_eq", "rsnorm", "pobj", "solvetime", "fevalstime", "QPtime", "QPit", "QPexitflag", "solver_id"};
	
	/* Check for proper number of arguments */
    if (nrhs != 1)
	{
		mexErrMsgTxt("This function requires exactly 1 input: PARAMS struct.\nType 'help FORCES_NLP_solver_mex' for details.");
	}    
	if (nlhs > 3) 
	{
        mexErrMsgTxt("This function returns at most 3 outputs.\nType 'help FORCES_NLP_solver_mex' for details.");
    }

	/* Check whether params is actually a structure */
	if( !mxIsStruct(PARAMS) ) 
	{
		mexErrMsgTxt("PARAMS must be a structure.");
	}
	 

    /* initialize memory */
    if (mem == NULL)
    {
        mem = FORCES_NLP_solver_internal_mem(0);
    }

	/* copy parameters into the right location */
	par = mxGetField(PARAMS, 0, "xinit");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	
	{
        mexErrMsgTxt("PARAMS.xinit not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.xinit must be a double.");
    }
    if( mxGetM(par) != 6 || mxGetN(par) != 1 ) 
	{
    mexErrMsgTxt("PARAMS.xinit must be of size [6 x 1]");
    }
#endif	 
	if ( (mxGetN(par) != 0) && (mxGetM(par) != 0) )
	{
		copyMArrayToC_double(mxGetPr(par), params.xinit,6);

	}
	par = mxGetField(PARAMS, 0, "x0");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	
	{
        mexErrMsgTxt("PARAMS.x0 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.x0 must be a double.");
    }
    if( mxGetM(par) != 168 || mxGetN(par) != 1 ) 
	{
    mexErrMsgTxt("PARAMS.x0 must be of size [168 x 1]");
    }
#endif	 
	if ( (mxGetN(par) != 0) && (mxGetM(par) != 0) )
	{
		copyMArrayToC_double(mxGetPr(par), params.x0,168);

	}
	par = mxGetField(PARAMS, 0, "all_parameters");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	
	{
        mexErrMsgTxt("PARAMS.all_parameters not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.all_parameters must be a double.");
    }
    if( mxGetM(par) != 21 || mxGetN(par) != 1 ) 
	{
    mexErrMsgTxt("PARAMS.all_parameters must be of size [21 x 1]");
    }
#endif	 
	if ( (mxGetN(par) != 0) && (mxGetM(par) != 0) )
	{
		copyMArrayToC_double(mxGetPr(par), params.all_parameters,21);

	}
	par = mxGetField(PARAMS, 0, "reinitialize");
	if ( (par != NULL) && (mxGetN(par) != 0) && (mxGetM(par) != 0) )
	{
		copyMValueToC_FORCES_NLP_solver_int(mxGetPr(par), &params.reinitialize);

	}




	#if SET_PRINTLEVEL_FORCES_NLP_solver > 0
		/* Prepare file for printfs */
        fp = fopen("stdout_temp","w+");
		if( fp == NULL ) 
		{
			mexErrMsgTxt("freopen of stdout did not work.");
		}
		rewind(fp);
	#endif

	/* call solver */

	exitflag = FORCES_NLP_solver_solve(&params, &output, &info, mem, fp, pt2function_FORCES_NLP_solver);
	
	#if SET_PRINTLEVEL_FORCES_NLP_solver > 0
		/* Read contents of printfs printed to file */
		rewind(fp);
		while( (i = fgetc(fp)) != EOF ) 
		{
			mexPrintf("%c",i);
		}
		fclose(fp);
	#endif

	/* copy output to matlab arrays */
	plhs[0] = mxCreateStructMatrix(1, 1, 21, outputnames);
		/* column vector of length 8 */
	outvar = mxCreateDoubleMatrix(8, 1, mxREAL);
	copyCArrayToM_double((&(output.x01[0])), mxGetPr(outvar), 8);
	mxSetField(plhs[0], 0, "x01", outvar);


	/* column vector of length 8 */
	outvar = mxCreateDoubleMatrix(8, 1, mxREAL);
	copyCArrayToM_double((&(output.x02[0])), mxGetPr(outvar), 8);
	mxSetField(plhs[0], 0, "x02", outvar);


	/* column vector of length 8 */
	outvar = mxCreateDoubleMatrix(8, 1, mxREAL);
	copyCArrayToM_double((&(output.x03[0])), mxGetPr(outvar), 8);
	mxSetField(plhs[0], 0, "x03", outvar);


	/* column vector of length 8 */
	outvar = mxCreateDoubleMatrix(8, 1, mxREAL);
	copyCArrayToM_double((&(output.x04[0])), mxGetPr(outvar), 8);
	mxSetField(plhs[0], 0, "x04", outvar);


	/* column vector of length 8 */
	outvar = mxCreateDoubleMatrix(8, 1, mxREAL);
	copyCArrayToM_double((&(output.x05[0])), mxGetPr(outvar), 8);
	mxSetField(plhs[0], 0, "x05", outvar);


	/* column vector of length 8 */
	outvar = mxCreateDoubleMatrix(8, 1, mxREAL);
	copyCArrayToM_double((&(output.x06[0])), mxGetPr(outvar), 8);
	mxSetField(plhs[0], 0, "x06", outvar);


	/* column vector of length 8 */
	outvar = mxCreateDoubleMatrix(8, 1, mxREAL);
	copyCArrayToM_double((&(output.x07[0])), mxGetPr(outvar), 8);
	mxSetField(plhs[0], 0, "x07", outvar);


	/* column vector of length 8 */
	outvar = mxCreateDoubleMatrix(8, 1, mxREAL);
	copyCArrayToM_double((&(output.x08[0])), mxGetPr(outvar), 8);
	mxSetField(plhs[0], 0, "x08", outvar);


	/* column vector of length 8 */
	outvar = mxCreateDoubleMatrix(8, 1, mxREAL);
	copyCArrayToM_double((&(output.x09[0])), mxGetPr(outvar), 8);
	mxSetField(plhs[0], 0, "x09", outvar);


	/* column vector of length 8 */
	outvar = mxCreateDoubleMatrix(8, 1, mxREAL);
	copyCArrayToM_double((&(output.x10[0])), mxGetPr(outvar), 8);
	mxSetField(plhs[0], 0, "x10", outvar);


	/* column vector of length 8 */
	outvar = mxCreateDoubleMatrix(8, 1, mxREAL);
	copyCArrayToM_double((&(output.x11[0])), mxGetPr(outvar), 8);
	mxSetField(plhs[0], 0, "x11", outvar);


	/* column vector of length 8 */
	outvar = mxCreateDoubleMatrix(8, 1, mxREAL);
	copyCArrayToM_double((&(output.x12[0])), mxGetPr(outvar), 8);
	mxSetField(plhs[0], 0, "x12", outvar);


	/* column vector of length 8 */
	outvar = mxCreateDoubleMatrix(8, 1, mxREAL);
	copyCArrayToM_double((&(output.x13[0])), mxGetPr(outvar), 8);
	mxSetField(plhs[0], 0, "x13", outvar);


	/* column vector of length 8 */
	outvar = mxCreateDoubleMatrix(8, 1, mxREAL);
	copyCArrayToM_double((&(output.x14[0])), mxGetPr(outvar), 8);
	mxSetField(plhs[0], 0, "x14", outvar);


	/* column vector of length 8 */
	outvar = mxCreateDoubleMatrix(8, 1, mxREAL);
	copyCArrayToM_double((&(output.x15[0])), mxGetPr(outvar), 8);
	mxSetField(plhs[0], 0, "x15", outvar);


	/* column vector of length 8 */
	outvar = mxCreateDoubleMatrix(8, 1, mxREAL);
	copyCArrayToM_double((&(output.x16[0])), mxGetPr(outvar), 8);
	mxSetField(plhs[0], 0, "x16", outvar);


	/* column vector of length 8 */
	outvar = mxCreateDoubleMatrix(8, 1, mxREAL);
	copyCArrayToM_double((&(output.x17[0])), mxGetPr(outvar), 8);
	mxSetField(plhs[0], 0, "x17", outvar);


	/* column vector of length 8 */
	outvar = mxCreateDoubleMatrix(8, 1, mxREAL);
	copyCArrayToM_double((&(output.x18[0])), mxGetPr(outvar), 8);
	mxSetField(plhs[0], 0, "x18", outvar);


	/* column vector of length 8 */
	outvar = mxCreateDoubleMatrix(8, 1, mxREAL);
	copyCArrayToM_double((&(output.x19[0])), mxGetPr(outvar), 8);
	mxSetField(plhs[0], 0, "x19", outvar);


	/* column vector of length 8 */
	outvar = mxCreateDoubleMatrix(8, 1, mxREAL);
	copyCArrayToM_double((&(output.x20[0])), mxGetPr(outvar), 8);
	mxSetField(plhs[0], 0, "x20", outvar);


	/* column vector of length 8 */
	outvar = mxCreateDoubleMatrix(8, 1, mxREAL);
	copyCArrayToM_double((&(output.x21[0])), mxGetPr(outvar), 8);
	mxSetField(plhs[0], 0, "x21", outvar);


	/* copy exitflag */
	if( nlhs > 1 )
	{
	plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
	*mxGetPr(plhs[1]) = (double)exitflag;
	}

	/* copy info struct */
	if( nlhs > 2 )
	{
	plhs[2] = mxCreateStructMatrix(1, 1, 10, infofields);
				/* scalar: iteration number */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		copyCArrayToM_solver_int32_default((&(info.it)), mxGetPr(outvar), 1);
		mxSetField(plhs[2], 0, "it", outvar);


		/* scalar: inf-norm of equality constraint residuals */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		copyCArrayToM_FORCES_NLP_solver_float((&(info.res_eq)), mxGetPr(outvar), 1);
		mxSetField(plhs[2], 0, "res_eq", outvar);


		/* scalar: norm of stationarity condition */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		copyCArrayToM_FORCES_NLP_solver_float((&(info.rsnorm)), mxGetPr(outvar), 1);
		mxSetField(plhs[2], 0, "rsnorm", outvar);


		/* scalar: primal objective */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		copyCArrayToM_FORCES_NLP_solver_float((&(info.pobj)), mxGetPr(outvar), 1);
		mxSetField(plhs[2], 0, "pobj", outvar);


		/* scalar: total solve time */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		copyCArrayToM_FORCES_NLP_solver_float((&(info.solvetime)), mxGetPr(outvar), 1);
		mxSetField(plhs[2], 0, "solvetime", outvar);


		/* scalar: time spent in function evaluations */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		copyCArrayToM_FORCES_NLP_solver_float((&(info.fevalstime)), mxGetPr(outvar), 1);
		mxSetField(plhs[2], 0, "fevalstime", outvar);


		/* scalar: time spent solving inner QPs */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		copyCArrayToM_FORCES_NLP_solver_float((&(info.QPtime)), mxGetPr(outvar), 1);
		mxSetField(plhs[2], 0, "QPtime", outvar);


		/* scalar: iterations spent in solving inner QPs */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		copyCArrayToM_solver_int32_default((&(info.QPit)), mxGetPr(outvar), 1);
		mxSetField(plhs[2], 0, "QPit", outvar);


		/* scalar: last exitflag of inner QP solver */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		copyCArrayToM_solver_int32_default((&(info.QPexitflag)), mxGetPr(outvar), 1);
		mxSetField(plhs[2], 0, "QPexitflag", outvar);


		/* column vector of length 8: solver ID of FORCESPRO solver */
		outvar = mxCreateDoubleMatrix(8, 1, mxREAL);
		copyCArrayToM_solver_int32_default((&(info.solver_id[0])), mxGetPr(outvar), 8);
		mxSetField(plhs[2], 0, "solver_id", outvar);

	}
}
