% Returns a ForcesInterpolation object created by calling interp1 on the
% provided data.
%
% FINTERP = FORCESINTERPOLATIONFIT(X,V) returns a piecewise cubic 
% interpolation of the values V at nodes X using Matlab's interp1 method.
% X and V must be vectors of the same length (of at least 2). The node 
% values X must be strictly monotonically increasing.
%
% FINTERP = FORCESINTERPOLATIONFIT(X,V,METHOD) uses the interpolation 
% algorithm specified by METHOD to generate a piecewise-polynomial form
% of the value V.
%
% See also ForcesInterpolation, interp1
%
% This file is part of the FORCESPRO client software for Matlab.
% (c) embotech AG, 2013-2023, Zurich, Switzerland. All rights reserved.
