% Generates a 2D spline object compatible with CasADi
% In particular, it uses casadi.Function.bspline
%
% Inputs:
%     tx: 1D sequence of ordered knot x-coordinates (interior + additional)
%
%     ty: 1D sequence of ordered knot y-coordinates (interior + additional)
%
%     coeffs: 2D array of spline coefficients, which are matrix-indexed
%             (i.e. rows correspond to x and columns correspond to y)
%
%     kx: degree of bivariate spline
%
%     ky: degree of bivariate spline
% 
%     m (optional): CasADi interpolant specific m-value. Defaults to 1.
% 
%     casadiOptions (optional): Struct containing options for the
%                               CasADi interpolant
% 
% Outputs:
%     spline: differentiable B-spline CasADi object function handle 
%             z = f(x, y)
% 
% This file is part of the FORCESPRO client software for Matlab.
% (c) embotech AG, 2022-2023, Zurich, Switzerland. All rights reserved.
