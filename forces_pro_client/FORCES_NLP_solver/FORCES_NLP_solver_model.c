/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) FORCES_NLP_solver_model_ ## ID
#endif

#include <math.h> 
#include "FORCES_NLP_solver_model.h"

#ifndef casadi_real
#define casadi_real FORCES_NLP_solver_float
#endif

#ifndef casadi_int
#define casadi_int solver_int32_default
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_f1 CASADI_PREFIX(f1)
#define casadi_f2 CASADI_PREFIX(f2)
#define casadi_f3 CASADI_PREFIX(f3)
#define casadi_f4 CASADI_PREFIX(f4)
#define casadi_f5 CASADI_PREFIX(f5)
#define casadi_f6 CASADI_PREFIX(f6)
#define casadi_f7 CASADI_PREFIX(f7)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_s7 CASADI_PREFIX(s7)
#define casadi_s8 CASADI_PREFIX(s8)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#if 0
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[12] = {8, 1, 0, 8, 0, 1, 2, 3, 4, 5, 6, 7};
static const casadi_int casadi_s1[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s2[19] = {1, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0};
static const casadi_int casadi_s3[4] = {0, 1, 0, 0};
static const casadi_int casadi_s4[10] = {6, 1, 0, 6, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s5[19] = {8, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7};
static const casadi_int casadi_s6[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s7[11] = {2, 6, 0, 0, 0, 0, 0, 1, 2, 0, 1};
static const casadi_int casadi_s8[23] = {6, 6, 0, 1, 7, 8, 14, 14, 14, 1, 0, 1, 2, 3, 4, 5, 3, 0, 1, 2, 3, 4, 5};

/* FORCES_NLP_solver_objective_0:(i0[8],i1)->(o0) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2, a3, a4, a5, a6, a7;
  a0=5.0000000000000000e-01;
  a1=3.1622776601683793e+01;
  a2=arg[0]? arg[0][2] : 0;
  a3=1.2000000000000000e+00;
  a4=arg[1]? arg[1][0] : 0;
  a5=(a3*a4);
  a5=(a2-a5);
  a5=(a1*a5);
  a6=(a3*a4);
  a2=(a2-a6);
  a2=(a1*a2);
  a5=(a5*a2);
  a2=3.1622776601683794e-01;
  a6=arg[0]? arg[0][3] : 0;
  a6=(a2*a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a6=arg[0]? arg[0][4] : 0;
  a7=(a3*a4);
  a7=(a6+a7);
  a7=(a1*a7);
  a3=(a3*a4);
  a6=(a6+a3);
  a1=(a1*a6);
  a7=(a7*a1);
  a5=(a5+a7);
  a7=arg[0]? arg[0][5] : 0;
  a2=(a2*a7);
  a2=casadi_sq(a2);
  a5=(a5+a2);
  a2=1.0000000000000001e-01;
  a7=arg[0]? arg[0][6] : 0;
  a7=(a2*a7);
  a7=casadi_sq(a7);
  a5=(a5+a7);
  a7=arg[0]? arg[0][7] : 0;
  a7=(a2*a7);
  a7=casadi_sq(a7);
  a5=(a5+a7);
  a7=arg[0]? arg[0][0] : 0;
  a7=(a2*a7);
  a7=casadi_sq(a7);
  a5=(a5+a7);
  a7=arg[0]? arg[0][1] : 0;
  a2=(a2*a7);
  a2=casadi_sq(a2);
  a5=(a5+a2);
  a0=(a0*a5);
  if (res[0]!=0) res[0][0]=a0;
  return 0;
}

int FORCES_NLP_solver_objective_0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

int FORCES_NLP_solver_objective_0_alloc_mem(void) {
  return 0;
}

int FORCES_NLP_solver_objective_0_init_mem(int mem) {
  return 0;
}

void FORCES_NLP_solver_objective_0_free_mem(int mem) {
}

int FORCES_NLP_solver_objective_0_checkout(void) {
  return 0;
}

void FORCES_NLP_solver_objective_0_release(int mem) {
}

void FORCES_NLP_solver_objective_0_incref(void) {
}

void FORCES_NLP_solver_objective_0_decref(void) {
}

casadi_int FORCES_NLP_solver_objective_0_n_in(void) { return 2;}

casadi_int FORCES_NLP_solver_objective_0_n_out(void) { return 1;}

casadi_real FORCES_NLP_solver_objective_0_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

const char* FORCES_NLP_solver_objective_0_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    default: return 0;
  }
}

const char* FORCES_NLP_solver_objective_0_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

const casadi_int* FORCES_NLP_solver_objective_0_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

const casadi_int* FORCES_NLP_solver_objective_0_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s1;
    default: return 0;
  }
}

int FORCES_NLP_solver_objective_0_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* FORCES_NLP_solver_dobjective_0:(i0[8],i1)->(o0[1x8]) */
static int casadi_f1(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2, a3, a4, a5, a6, a7;
  a0=1.0000000000000001e-01;
  a1=5.0000000000000000e-01;
  a2=arg[0]? arg[0][0] : 0;
  a2=(a0*a2);
  a2=(a2+a2);
  a2=(a1*a2);
  a2=(a0*a2);
  if (res[0]!=0) res[0][0]=a2;
  a2=arg[0]? arg[0][1] : 0;
  a2=(a0*a2);
  a2=(a2+a2);
  a2=(a1*a2);
  a2=(a0*a2);
  if (res[0]!=0) res[0][1]=a2;
  a2=3.1622776601683793e+01;
  a3=arg[0]? arg[0][2] : 0;
  a4=1.2000000000000000e+00;
  a5=arg[1]? arg[1][0] : 0;
  a6=(a4*a5);
  a6=(a3-a6);
  a6=(a2*a6);
  a6=(a1*a6);
  a6=(a2*a6);
  a7=(a4*a5);
  a3=(a3-a7);
  a3=(a2*a3);
  a3=(a1*a3);
  a3=(a2*a3);
  a6=(a6+a3);
  if (res[0]!=0) res[0][2]=a6;
  a6=3.1622776601683794e-01;
  a3=arg[0]? arg[0][3] : 0;
  a3=(a6*a3);
  a3=(a3+a3);
  a3=(a1*a3);
  a3=(a6*a3);
  if (res[0]!=0) res[0][3]=a3;
  a3=arg[0]? arg[0][4] : 0;
  a7=(a4*a5);
  a7=(a3+a7);
  a7=(a2*a7);
  a7=(a1*a7);
  a7=(a2*a7);
  a4=(a4*a5);
  a3=(a3+a4);
  a3=(a2*a3);
  a3=(a1*a3);
  a2=(a2*a3);
  a7=(a7+a2);
  if (res[0]!=0) res[0][4]=a7;
  a7=arg[0]? arg[0][5] : 0;
  a7=(a6*a7);
  a7=(a7+a7);
  a7=(a1*a7);
  a6=(a6*a7);
  if (res[0]!=0) res[0][5]=a6;
  a6=arg[0]? arg[0][6] : 0;
  a6=(a0*a6);
  a6=(a6+a6);
  a6=(a1*a6);
  a6=(a0*a6);
  if (res[0]!=0) res[0][6]=a6;
  a6=arg[0]? arg[0][7] : 0;
  a6=(a0*a6);
  a6=(a6+a6);
  a1=(a1*a6);
  a0=(a0*a1);
  if (res[0]!=0) res[0][7]=a0;
  return 0;
}

int FORCES_NLP_solver_dobjective_0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f1(arg, res, iw, w, mem);
}

int FORCES_NLP_solver_dobjective_0_alloc_mem(void) {
  return 0;
}

int FORCES_NLP_solver_dobjective_0_init_mem(int mem) {
  return 0;
}

void FORCES_NLP_solver_dobjective_0_free_mem(int mem) {
}

int FORCES_NLP_solver_dobjective_0_checkout(void) {
  return 0;
}

void FORCES_NLP_solver_dobjective_0_release(int mem) {
}

void FORCES_NLP_solver_dobjective_0_incref(void) {
}

void FORCES_NLP_solver_dobjective_0_decref(void) {
}

casadi_int FORCES_NLP_solver_dobjective_0_n_in(void) { return 2;}

casadi_int FORCES_NLP_solver_dobjective_0_n_out(void) { return 1;}

casadi_real FORCES_NLP_solver_dobjective_0_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

const char* FORCES_NLP_solver_dobjective_0_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    default: return 0;
  }
}

const char* FORCES_NLP_solver_dobjective_0_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

const casadi_int* FORCES_NLP_solver_dobjective_0_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

const casadi_int* FORCES_NLP_solver_dobjective_0_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    default: return 0;
  }
}

int FORCES_NLP_solver_dobjective_0_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* FORCES_NLP_solver_hessian_0:(i0[8],i1,i2[0],i3[6])->(o0[8x8,8nz]) */
static int casadi_f2(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2;
  a0=1.0000000000000002e-02;
  if (res[0]!=0) res[0][0]=a0;
  if (res[0]!=0) res[0][1]=a0;
  a1=1000.;
  if (res[0]!=0) res[0][2]=a1;
  a2=1.0000000000000001e-01;
  if (res[0]!=0) res[0][3]=a2;
  if (res[0]!=0) res[0][4]=a1;
  if (res[0]!=0) res[0][5]=a2;
  if (res[0]!=0) res[0][6]=a0;
  if (res[0]!=0) res[0][7]=a0;
  return 0;
}

int FORCES_NLP_solver_hessian_0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f2(arg, res, iw, w, mem);
}

int FORCES_NLP_solver_hessian_0_alloc_mem(void) {
  return 0;
}

int FORCES_NLP_solver_hessian_0_init_mem(int mem) {
  return 0;
}

void FORCES_NLP_solver_hessian_0_free_mem(int mem) {
}

int FORCES_NLP_solver_hessian_0_checkout(void) {
  return 0;
}

void FORCES_NLP_solver_hessian_0_release(int mem) {
}

void FORCES_NLP_solver_hessian_0_incref(void) {
}

void FORCES_NLP_solver_hessian_0_decref(void) {
}

casadi_int FORCES_NLP_solver_hessian_0_n_in(void) { return 4;}

casadi_int FORCES_NLP_solver_hessian_0_n_out(void) { return 1;}

casadi_real FORCES_NLP_solver_hessian_0_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

const char* FORCES_NLP_solver_hessian_0_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

const char* FORCES_NLP_solver_hessian_0_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

const casadi_int* FORCES_NLP_solver_hessian_0_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s3;
    case 3: return casadi_s4;
    default: return 0;
  }
}

const casadi_int* FORCES_NLP_solver_hessian_0_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s5;
    default: return 0;
  }
}

int FORCES_NLP_solver_hessian_0_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* FORCES_NLP_solver_cdyn_0:(i0[6],i1[2],i2)->(o0[6],o1[2x6,2nz],o2[6x6,14nz]) */
static int casadi_f3(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a4, a5, a6, a7, a8, a9;
  a0=arg[0]? arg[0][1] : 0;
  if (res[0]!=0) res[0][0]=a0;
  a1=2.5000000000000000e-01;
  a2=3.;
  a3=1.;
  a4=5.0000000000000000e-01;
  a5=arg[0]? arg[0][2] : 0;
  a6=cos(a5);
  a6=(a4*a6);
  a6=(a3+a6);
  a6=(a2*a6);
  a6=(a1+a6);
  a7=3.7500000000000001e-04;
  a6=(a6+a7);
  a8=3.2503750000000000e+00;
  a9=(a6/a8);
  a10=2.9430000000000000e+01;
  a11=arg[0]? arg[0][0] : 0;
  a12=(a11+a5);
  a13=cos(a12);
  a13=(a10*a13);
  a14=1.5000000000000000e+00;
  a15=sin(a5);
  a15=(a14*a15);
  a16=(a15*a0);
  a17=(a16*a0);
  a17=(a13+a17);
  a18=arg[0]? arg[0][5] : 0;
  a17=(a17-a18);
  a19=(a9*a17);
  a20=-3.;
  a21=sin(a5);
  a21=(a20*a21);
  a22=(a21*a0);
  a23=arg[0]? arg[0][3] : 0;
  a24=(a22*a23);
  a19=(a19-a24);
  a24=-1.5000000000000000e+00;
  a25=sin(a5);
  a25=(a24*a25);
  a26=(a25*a23);
  a19=(a19-a26);
  a26=4.5616500000000009e+01;
  a27=cos(a11);
  a27=(a26*a27);
  a28=(a11+a5);
  a29=cos(a28);
  a29=(a10*a29);
  a27=(a27+a29);
  a19=(a19-a27);
  a27=arg[0]? arg[0][4] : 0;
  a19=(a19+a27);
  a27=3.5003750000000000e+00;
  a29=1.2500000000000000e+00;
  a30=cos(a5);
  a29=(a29+a30);
  a29=(a29+a7);
  a30=7.4999999999999997e-02;
  a29=(a29+a30);
  a29=(a2*a29);
  a27=(a27+a29);
  a29=cos(a5);
  a29=(a4*a29);
  a29=(a3+a29);
  a29=(a2*a29);
  a1=(a1+a29);
  a1=(a1+a7);
  a7=(a6*a1);
  a7=(a7/a8);
  a27=(a27-a7);
  a19=(a19/a27);
  if (res[0]!=0) res[0][1]=a19;
  if (res[0]!=0) res[0][2]=a23;
  a7=3.0765680882975044e-01;
  a8=(a1*a19);
  a18=(a18-a8);
  a8=(a15*a0);
  a29=(a8*a0);
  a18=(a18-a29);
  a18=(a18-a13);
  a18=(a7*a18);
  if (res[0]!=0) res[0][3]=a18;
  a18=arg[1]? arg[1][0] : 0;
  if (res[0]!=0) res[0][4]=a18;
  a18=arg[1]? arg[1][1] : 0;
  if (res[0]!=0) res[0][5]=a18;
  if (res[1]!=0) res[1][0]=a3;
  if (res[1]!=0) res[1][1]=a3;
  if (res[2]!=0) res[2][0]=a3;
  a11=sin(a11);
  a26=(a26*a11);
  a28=sin(a28);
  a11=(a10*a28);
  a26=(a26+a11);
  a12=sin(a12);
  a11=(a10*a12);
  a18=(a9*a11);
  a26=(a26-a18);
  a26=(a26/a27);
  if (res[2]!=0) res[2][1]=a26;
  a18=(a0*a15);
  a18=(a18+a16);
  a18=(a9*a18);
  a21=(a23*a21);
  a18=(a18-a21);
  a18=(a18/a27);
  if (res[2]!=0) res[2][2]=a18;
  a21=cos(a5);
  a14=(a14*a21);
  a21=(a0*a14);
  a21=(a0*a21);
  a12=(a10*a12);
  a21=(a21-a12);
  a21=(a9*a21);
  a16=sin(a5);
  a16=(a4*a16);
  a16=(a2*a16);
  a13=(a7*a16);
  a17=(a17*a13);
  a21=(a21-a17);
  a17=cos(a5);
  a20=(a20*a17);
  a20=(a0*a20);
  a20=(a23*a20);
  a21=(a21-a20);
  a20=cos(a5);
  a24=(a24*a20);
  a23=(a23*a24);
  a21=(a21-a23);
  a10=(a10*a28);
  a21=(a21+a10);
  a21=(a21/a27);
  a10=(a19/a27);
  a16=(a1*a16);
  a28=sin(a5);
  a4=(a4*a28);
  a4=(a2*a4);
  a6=(a6*a4);
  a16=(a16+a6);
  a16=(a7*a16);
  a5=sin(a5);
  a2=(a2*a5);
  a16=(a16-a2);
  a10=(a10*a16);
  a21=(a21-a10);
  if (res[2]!=0) res[2][3]=a21;
  a22=(a22+a25);
  a22=(a22/a27);
  a25=(-a22);
  if (res[2]!=0) res[2][4]=a25;
  a25=(1./a27);
  if (res[2]!=0) res[2][5]=a25;
  a9=(a9/a27);
  a25=(-a9);
  if (res[2]!=0) res[2][6]=a25;
  if (res[2]!=0) res[2][7]=a3;
  a26=(a1*a26);
  a11=(a11-a26);
  a11=(a7*a11);
  if (res[2]!=0) res[2][8]=a11;
  a18=(a1*a18);
  a15=(a0*a15);
  a15=(a15+a8);
  a18=(a18+a15);
  a18=(a7*a18);
  a18=(-a18);
  if (res[2]!=0) res[2][9]=a18;
  a21=(a1*a21);
  a19=(a19*a4);
  a21=(a21-a19);
  a14=(a0*a14);
  a0=(a0*a14);
  a21=(a21+a0);
  a12=(a12-a21);
  a12=(a7*a12);
  if (res[2]!=0) res[2][10]=a12;
  a22=(a1*a22);
  a22=(a7*a22);
  if (res[2]!=0) res[2][11]=a22;
  a27=(a1/a27);
  a27=(a7*a27);
  a27=(-a27);
  if (res[2]!=0) res[2][12]=a27;
  a1=(a1*a9);
  a3=(a3+a1);
  a7=(a7*a3);
  if (res[2]!=0) res[2][13]=a7;
  return 0;
}

int FORCES_NLP_solver_cdyn_0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f3(arg, res, iw, w, mem);
}

int FORCES_NLP_solver_cdyn_0_alloc_mem(void) {
  return 0;
}

int FORCES_NLP_solver_cdyn_0_init_mem(int mem) {
  return 0;
}

void FORCES_NLP_solver_cdyn_0_free_mem(int mem) {
}

int FORCES_NLP_solver_cdyn_0_checkout(void) {
  return 0;
}

void FORCES_NLP_solver_cdyn_0_release(int mem) {
}

void FORCES_NLP_solver_cdyn_0_incref(void) {
}

void FORCES_NLP_solver_cdyn_0_decref(void) {
}

casadi_int FORCES_NLP_solver_cdyn_0_n_in(void) { return 3;}

casadi_int FORCES_NLP_solver_cdyn_0_n_out(void) { return 3;}

casadi_real FORCES_NLP_solver_cdyn_0_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

const char* FORCES_NLP_solver_cdyn_0_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    default: return 0;
  }
}

const char* FORCES_NLP_solver_cdyn_0_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    default: return 0;
  }
}

const casadi_int* FORCES_NLP_solver_cdyn_0_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    case 1: return casadi_s6;
    case 2: return casadi_s1;
    default: return 0;
  }
}

const casadi_int* FORCES_NLP_solver_cdyn_0_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    case 1: return casadi_s7;
    case 2: return casadi_s8;
    default: return 0;
  }
}

int FORCES_NLP_solver_cdyn_0_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 3;
  if (sz_res) *sz_res = 3;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* FORCES_NLP_solver_cdyn_0rd_0:(i0[6],i1[2],i2)->(o0[6]) */
static int casadi_f4(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a2, a3, a4, a5, a6, a7, a8, a9;
  a0=arg[0]? arg[0][1] : 0;
  if (res[0]!=0) res[0][0]=a0;
  a1=2.5000000000000000e-01;
  a2=3.;
  a3=1.;
  a4=5.0000000000000000e-01;
  a5=arg[0]? arg[0][2] : 0;
  a6=cos(a5);
  a6=(a4*a6);
  a6=(a3+a6);
  a6=(a2*a6);
  a6=(a1+a6);
  a7=3.7500000000000001e-04;
  a6=(a6+a7);
  a8=3.2503750000000000e+00;
  a9=(a6/a8);
  a10=2.9430000000000000e+01;
  a11=arg[0]? arg[0][0] : 0;
  a12=(a11+a5);
  a12=cos(a12);
  a12=(a10*a12);
  a13=1.5000000000000000e+00;
  a14=sin(a5);
  a13=(a13*a14);
  a14=(a13*a0);
  a14=(a14*a0);
  a14=(a12+a14);
  a15=arg[0]? arg[0][5] : 0;
  a14=(a14-a15);
  a9=(a9*a14);
  a14=-3.;
  a16=sin(a5);
  a14=(a14*a16);
  a14=(a14*a0);
  a16=arg[0]? arg[0][3] : 0;
  a14=(a14*a16);
  a9=(a9-a14);
  a14=-1.5000000000000000e+00;
  a17=sin(a5);
  a14=(a14*a17);
  a14=(a14*a16);
  a9=(a9-a14);
  a14=4.5616500000000009e+01;
  a17=cos(a11);
  a14=(a14*a17);
  a11=(a11+a5);
  a11=cos(a11);
  a10=(a10*a11);
  a14=(a14+a10);
  a9=(a9-a14);
  a14=arg[0]? arg[0][4] : 0;
  a9=(a9+a14);
  a14=3.5003750000000000e+00;
  a10=1.2500000000000000e+00;
  a11=cos(a5);
  a10=(a10+a11);
  a10=(a10+a7);
  a11=7.4999999999999997e-02;
  a10=(a10+a11);
  a10=(a2*a10);
  a14=(a14+a10);
  a5=cos(a5);
  a4=(a4*a5);
  a3=(a3+a4);
  a2=(a2*a3);
  a1=(a1+a2);
  a1=(a1+a7);
  a6=(a6*a1);
  a6=(a6/a8);
  a14=(a14-a6);
  a9=(a9/a14);
  if (res[0]!=0) res[0][1]=a9;
  if (res[0]!=0) res[0][2]=a16;
  a16=3.0765680882975044e-01;
  a1=(a1*a9);
  a15=(a15-a1);
  a13=(a13*a0);
  a13=(a13*a0);
  a15=(a15-a13);
  a15=(a15-a12);
  a16=(a16*a15);
  if (res[0]!=0) res[0][3]=a16;
  a16=arg[1]? arg[1][0] : 0;
  if (res[0]!=0) res[0][4]=a16;
  a16=arg[1]? arg[1][1] : 0;
  if (res[0]!=0) res[0][5]=a16;
  return 0;
}

int FORCES_NLP_solver_cdyn_0rd_0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f4(arg, res, iw, w, mem);
}

int FORCES_NLP_solver_cdyn_0rd_0_alloc_mem(void) {
  return 0;
}

int FORCES_NLP_solver_cdyn_0rd_0_init_mem(int mem) {
  return 0;
}

void FORCES_NLP_solver_cdyn_0rd_0_free_mem(int mem) {
}

int FORCES_NLP_solver_cdyn_0rd_0_checkout(void) {
  return 0;
}

void FORCES_NLP_solver_cdyn_0rd_0_release(int mem) {
}

void FORCES_NLP_solver_cdyn_0rd_0_incref(void) {
}

void FORCES_NLP_solver_cdyn_0rd_0_decref(void) {
}

casadi_int FORCES_NLP_solver_cdyn_0rd_0_n_in(void) { return 3;}

casadi_int FORCES_NLP_solver_cdyn_0rd_0_n_out(void) { return 1;}

casadi_real FORCES_NLP_solver_cdyn_0rd_0_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

const char* FORCES_NLP_solver_cdyn_0rd_0_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    default: return 0;
  }
}

const char* FORCES_NLP_solver_cdyn_0rd_0_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

const casadi_int* FORCES_NLP_solver_cdyn_0rd_0_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    case 1: return casadi_s6;
    case 2: return casadi_s1;
    default: return 0;
  }
}

const casadi_int* FORCES_NLP_solver_cdyn_0rd_0_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    default: return 0;
  }
}

int FORCES_NLP_solver_cdyn_0rd_0_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 3;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* FORCES_NLP_solver_objective_1:(i0[8],i1)->(o0) */
static int casadi_f5(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2, a3, a4, a5, a6, a7;
  a0=5.0000000000000000e-01;
  a1=3.1622776601683793e+01;
  a2=arg[0]? arg[0][2] : 0;
  a3=1.2000000000000000e+00;
  a4=arg[1]? arg[1][0] : 0;
  a5=(a3*a4);
  a5=(a2-a5);
  a5=(a1*a5);
  a6=(a3*a4);
  a2=(a2-a6);
  a2=(a1*a2);
  a5=(a5*a2);
  a2=3.1622776601683794e-01;
  a6=arg[0]? arg[0][3] : 0;
  a6=(a2*a6);
  a6=casadi_sq(a6);
  a5=(a5+a6);
  a6=arg[0]? arg[0][4] : 0;
  a7=(a3*a4);
  a7=(a6+a7);
  a7=(a1*a7);
  a3=(a3*a4);
  a6=(a6+a3);
  a1=(a1*a6);
  a7=(a7*a1);
  a5=(a5+a7);
  a7=arg[0]? arg[0][5] : 0;
  a2=(a2*a7);
  a2=casadi_sq(a2);
  a5=(a5+a2);
  a2=1.0000000000000001e-01;
  a7=arg[0]? arg[0][6] : 0;
  a7=(a2*a7);
  a7=casadi_sq(a7);
  a5=(a5+a7);
  a7=arg[0]? arg[0][7] : 0;
  a7=(a2*a7);
  a7=casadi_sq(a7);
  a5=(a5+a7);
  a7=arg[0]? arg[0][0] : 0;
  a7=(a2*a7);
  a7=casadi_sq(a7);
  a5=(a5+a7);
  a7=arg[0]? arg[0][1] : 0;
  a2=(a2*a7);
  a2=casadi_sq(a2);
  a5=(a5+a2);
  a0=(a0*a5);
  if (res[0]!=0) res[0][0]=a0;
  return 0;
}

int FORCES_NLP_solver_objective_1(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f5(arg, res, iw, w, mem);
}

int FORCES_NLP_solver_objective_1_alloc_mem(void) {
  return 0;
}

int FORCES_NLP_solver_objective_1_init_mem(int mem) {
  return 0;
}

void FORCES_NLP_solver_objective_1_free_mem(int mem) {
}

int FORCES_NLP_solver_objective_1_checkout(void) {
  return 0;
}

void FORCES_NLP_solver_objective_1_release(int mem) {
}

void FORCES_NLP_solver_objective_1_incref(void) {
}

void FORCES_NLP_solver_objective_1_decref(void) {
}

casadi_int FORCES_NLP_solver_objective_1_n_in(void) { return 2;}

casadi_int FORCES_NLP_solver_objective_1_n_out(void) { return 1;}

casadi_real FORCES_NLP_solver_objective_1_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

const char* FORCES_NLP_solver_objective_1_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    default: return 0;
  }
}

const char* FORCES_NLP_solver_objective_1_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

const casadi_int* FORCES_NLP_solver_objective_1_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

const casadi_int* FORCES_NLP_solver_objective_1_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s1;
    default: return 0;
  }
}

int FORCES_NLP_solver_objective_1_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* FORCES_NLP_solver_dobjective_1:(i0[8],i1)->(o0[1x8]) */
static int casadi_f6(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2, a3, a4, a5, a6, a7;
  a0=1.0000000000000001e-01;
  a1=5.0000000000000000e-01;
  a2=arg[0]? arg[0][0] : 0;
  a2=(a0*a2);
  a2=(a2+a2);
  a2=(a1*a2);
  a2=(a0*a2);
  if (res[0]!=0) res[0][0]=a2;
  a2=arg[0]? arg[0][1] : 0;
  a2=(a0*a2);
  a2=(a2+a2);
  a2=(a1*a2);
  a2=(a0*a2);
  if (res[0]!=0) res[0][1]=a2;
  a2=3.1622776601683793e+01;
  a3=arg[0]? arg[0][2] : 0;
  a4=1.2000000000000000e+00;
  a5=arg[1]? arg[1][0] : 0;
  a6=(a4*a5);
  a6=(a3-a6);
  a6=(a2*a6);
  a6=(a1*a6);
  a6=(a2*a6);
  a7=(a4*a5);
  a3=(a3-a7);
  a3=(a2*a3);
  a3=(a1*a3);
  a3=(a2*a3);
  a6=(a6+a3);
  if (res[0]!=0) res[0][2]=a6;
  a6=3.1622776601683794e-01;
  a3=arg[0]? arg[0][3] : 0;
  a3=(a6*a3);
  a3=(a3+a3);
  a3=(a1*a3);
  a3=(a6*a3);
  if (res[0]!=0) res[0][3]=a3;
  a3=arg[0]? arg[0][4] : 0;
  a7=(a4*a5);
  a7=(a3+a7);
  a7=(a2*a7);
  a7=(a1*a7);
  a7=(a2*a7);
  a4=(a4*a5);
  a3=(a3+a4);
  a3=(a2*a3);
  a3=(a1*a3);
  a2=(a2*a3);
  a7=(a7+a2);
  if (res[0]!=0) res[0][4]=a7;
  a7=arg[0]? arg[0][5] : 0;
  a7=(a6*a7);
  a7=(a7+a7);
  a7=(a1*a7);
  a6=(a6*a7);
  if (res[0]!=0) res[0][5]=a6;
  a6=arg[0]? arg[0][6] : 0;
  a6=(a0*a6);
  a6=(a6+a6);
  a6=(a1*a6);
  a6=(a0*a6);
  if (res[0]!=0) res[0][6]=a6;
  a6=arg[0]? arg[0][7] : 0;
  a6=(a0*a6);
  a6=(a6+a6);
  a1=(a1*a6);
  a0=(a0*a1);
  if (res[0]!=0) res[0][7]=a0;
  return 0;
}

int FORCES_NLP_solver_dobjective_1(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f6(arg, res, iw, w, mem);
}

int FORCES_NLP_solver_dobjective_1_alloc_mem(void) {
  return 0;
}

int FORCES_NLP_solver_dobjective_1_init_mem(int mem) {
  return 0;
}

void FORCES_NLP_solver_dobjective_1_free_mem(int mem) {
}

int FORCES_NLP_solver_dobjective_1_checkout(void) {
  return 0;
}

void FORCES_NLP_solver_dobjective_1_release(int mem) {
}

void FORCES_NLP_solver_dobjective_1_incref(void) {
}

void FORCES_NLP_solver_dobjective_1_decref(void) {
}

casadi_int FORCES_NLP_solver_dobjective_1_n_in(void) { return 2;}

casadi_int FORCES_NLP_solver_dobjective_1_n_out(void) { return 1;}

casadi_real FORCES_NLP_solver_dobjective_1_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

const char* FORCES_NLP_solver_dobjective_1_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    default: return 0;
  }
}

const char* FORCES_NLP_solver_dobjective_1_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

const casadi_int* FORCES_NLP_solver_dobjective_1_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

const casadi_int* FORCES_NLP_solver_dobjective_1_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    default: return 0;
  }
}

int FORCES_NLP_solver_dobjective_1_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* FORCES_NLP_solver_hessian_1:(i0[8],i1,i2[0],i3[0])->(o0[8x8,8nz]) */
static int casadi_f7(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2;
  a0=1.0000000000000002e-02;
  if (res[0]!=0) res[0][0]=a0;
  if (res[0]!=0) res[0][1]=a0;
  a1=1000.;
  if (res[0]!=0) res[0][2]=a1;
  a2=1.0000000000000001e-01;
  if (res[0]!=0) res[0][3]=a2;
  if (res[0]!=0) res[0][4]=a1;
  if (res[0]!=0) res[0][5]=a2;
  if (res[0]!=0) res[0][6]=a0;
  if (res[0]!=0) res[0][7]=a0;
  return 0;
}

int FORCES_NLP_solver_hessian_1(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f7(arg, res, iw, w, mem);
}

int FORCES_NLP_solver_hessian_1_alloc_mem(void) {
  return 0;
}

int FORCES_NLP_solver_hessian_1_init_mem(int mem) {
  return 0;
}

void FORCES_NLP_solver_hessian_1_free_mem(int mem) {
}

int FORCES_NLP_solver_hessian_1_checkout(void) {
  return 0;
}

void FORCES_NLP_solver_hessian_1_release(int mem) {
}

void FORCES_NLP_solver_hessian_1_incref(void) {
}

void FORCES_NLP_solver_hessian_1_decref(void) {
}

casadi_int FORCES_NLP_solver_hessian_1_n_in(void) { return 4;}

casadi_int FORCES_NLP_solver_hessian_1_n_out(void) { return 1;}

casadi_real FORCES_NLP_solver_hessian_1_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

const char* FORCES_NLP_solver_hessian_1_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

const char* FORCES_NLP_solver_hessian_1_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

const casadi_int* FORCES_NLP_solver_hessian_1_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s3;
    case 3: return casadi_s3;
    default: return 0;
  }
}

const casadi_int* FORCES_NLP_solver_hessian_1_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s5;
    default: return 0;
  }
}

int FORCES_NLP_solver_hessian_1_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
