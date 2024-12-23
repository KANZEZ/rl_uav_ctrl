%% Interface between adaptive MPC and FORCESPRO QP Solver (internal)
%
%   The method generates "params" structure used by FORCESPRO QP solver at run
%   time.

% data is a core data structure generated by "mpcToForces"
% onlinedata is a structure where 
%   yref is a matrix of p-by-ny values for output references
%   uref is a matrix pf p-by-nmv values for MV reference
%   md is a matrix of (p+1)-by-nmd values for MD
%   externalMV is a nmv-by-1 vector for true previous MV used in the plant 
%
% params(1) is "c", a column vector stacked from step 1 to p
% params(2) is "f", a column vector stacked from step 1 to p
% optional parameters:
%   "H" when "data.UseOnlineWeight" is true;

% FORCESPRO low-level API supports "convex multi-stage linearly constrained
% problem":
%
%   min sum(i=1..p) 0.5*z(i)'*H(i)*z(i) + f(i)'*z(i)
%
%   s.t.    DD(1)*z(1) = cc(1)                  initial equality (i=1)
%           CC(i-1)*z(i-1) + DD(i)*z(i) = cc(i) inter-stage equality (i=2..p)
%           lower and upper bounds of z(i)      decision variable bounds (1..p)
%           AA(i)*z(i) <= bb(i)                 polytopic inequalities (1..p)
%
% Assume time k is the current time and time k+1 to k+p correspond to p
% prediction steps, we define the decision variables for p stages as:
%   z[1] = [mv[k]; dmv[k]; y[k+1]; x[k+1]; slack[k]] for stage 1
%   z[2] = [mv[k+1]; dmv[k+1]; y[k+2]; x[k+2]; slack[k+1]] for stage 2
%   ...
%   z[p] = [mv[k+p-1]; dmv[k+p-1]; y[k+p]; x[k+p]; slack[k+p-1]] for stage p
% where dmv[i] = mv[i] - mv[i-1] and mv[0] is last mv provided at run time.
%
% For discrete-time prediction model defined as
%   x[k+1] = A[k]*x[k] + Bmv[k]*mv[k] + Bmd[k]*md[k] + Bdx[k]*1
%   y[k] = C[k]*x[k] + 0*mv[k] + Dmd[k]*md[k] + Ddx[k]*1
% where x/mv/md/y are deviation variables and Bdx[k], Ddx[k] are the lumped
% offsets occurring due to changing nominal conditions
%
% Model needs to be translated into FORCESPRO equality constraints.
% Therefore, we have for i=1 (as prediction step k+1)
%   DD[1] = [0 0 -I C; Bmv[k] 0 0 -I; I -I 0 0]
%   cc[1] = [-(Dmd[k+1]*md[k+1]+Ddx[k+1]*1); -(A[k]*x[k]+Bmd[k]*md[k]+Bdx[k]*1); trueLastMV]
% and for i=2,..p (as prediction steps k+2..k+p)
%   CC[i-1] = [0 0 0 0; 0 0 0 A[k+i-1]; -I 0 0 0] 
%   DD[i] = [0 0 -I C[k+i]; Bmv[k+i-1] 0 0 -I; I -I 0 0]
%   cc[i] = [-(Dmd[k+i]*md(k+i)+Ddx[k+i]*1); -Bmd*md[k+i-1]-Bdx[k+i-1]*1; 0]
% CC and DD change due to updated prediction model. "cc" is a function
% of run time MD signal and thus passed in as parameter.

%   Author(s): Rong Chen, MathWorks Inc.
%              Nivethan Yogarajah, Embotech AG
%
%   Copyright 2019-2023 The MathWorks, Inc. and Embotech AG
