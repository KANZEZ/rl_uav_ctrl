% Generates a 2D spline object compatible with CasADi and parametric
% coefficient matrix.
% In particular, it uses the parametric bispline class casadi.bspline
%
% 
% IMPORTANT: The knots {tx, ty} and the degrees [kx, ky] cannot be
%            changed. Only the coefficient matrix is parametric. 
% 
% Inputs:
%     numel_coeffs: number of elements in the coefficient matrix
% 
%     tx: 1D sequence of ordered knot x-coordinates (interior + additional)
%
%     ty: 1D sequence of ordered knot y-coordinates (interior + additional)
%
%     kx: degree of bivariate spline
%
%     ky: degree of bivariate spline
% 
%     m (optional): CasADi interpolant specific m-value. Defaults to 1.
% 
%     casadiOptions (optional): Struct containing options for the
%                               parametric
%                               CasADi interpolant
% 
% Outputs:
%     spline: differentiable B-spline CasADi object function handle with 
%             parametric coefficients z = f(x, y, coeffs)
% 
%             IMPORTANT: The inputs to the resulting fcn handle are 
%                        (x, y, coeffs), where the coeffs are the 
%                        matrix-indexed coefficients matrix flattened in
%                        column-major order
%
% This file is part of the FORCESPRO client software for Matlab.
% (c) embotech AG, 2022-2023, Zurich, Switzerland. All rights reserved.
