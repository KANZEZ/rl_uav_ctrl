% ForcesInterpolation is class providing continuous interpolations from
% data for use within FORCESPRO.
%
% FINTERP = FORCESINTERPOLATION(BREAKS, COEFS) creates an interpolation 
% object for use with FORCESPRO. BREAKS is a row vector of size M+1 and 
% defines M pieces for the interpolation. The value of the interpolation 
% on the i-th piece is given by
%
%   FINTERP(x) = COEFS(i,1)*x^(N-1) + COEFS(i,2)*x^(N-2) + ... + COEFS(i,N)
%
% where x is the curent point to evaluate. For multi-valued interpolations 
% with an output dimension K, COEFS is a (K*M)-by-N matrix, and the 
% evaluation is as follows for output k (1 <= k <= K):
%
%   FINTERP(x)_k = COEFS(i+(k-1),1)*x^(N-1) + ... + COEFS(i+(k-1),N)
%
% that is, for each piece the coefficients in COEFS are concatenated
% vertically into blocks of K rows, of which there are M in total.
%
% FINTERP = FORCESINTERPOLATION(BREAKS, COEFS, 'pp') creates an 
% interpolation object as above, but the polynomial on the i-th piece is 
% defined as in the Matlab functions MKPP and as used by PPVAL:
%
%   FINTERP(x) = COEFS(i,1)*dx^(N-1) + COEFS(i,2)*dx^(N-2) + ... + COEFS(i,N)
%
% where dx := x - BREAKS(i). This is for direct compatibility with
% Matlab-created interpolations. Multi-valued interpolations are handled 
% accordingly as described above.
%
% FINTERP = FORCESINTERPOLATION(PP) as above, directly using the struct 
% returned by MKPP.
%
% In general, COEFS can also be parametric when used in FORCESPRO.
%
% See also ForcesInterpolationFit, mkpp, ppval
%
% This file is part of the FORCESPRO client software for Matlab.
% (c) embotech AG, 2013-2023, Zurich, Switzerland. All rights reserved.
