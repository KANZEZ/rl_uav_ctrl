% Fits a 2D-spline using Scipy's scipy.interpolate.SmoothBivariateSpline
% and creates a CasADi B-spline object using the fitted coefficients.
% 
% It additionally supports scaling of the I/O, which affects the numerical
% stability of Scipy's spline fitting routine.
% The scaling is automatically taken care for, so that the I/O to the
% resulting spline are still the physical quantities.
%
% IMPORTANT: This functions requires a valid Python installation as well as 
% the packages Scipy and Numpy on the system.
% 
% Inputs:
%     x: 1D sequence of data points
% 
%     y: 1D sequence of data points
% 
%     z: 1D sequence of data pionts
% 
%     w (optional): Positive 1D sequence of weights with same length as x, 
%                   y, z. If not or empty list is provided, defaults to 
%                   unit gains.
% 
%     bbox (optional): Sequence of length 4 specifying the boundary of the
%                      rectangular approximation domain. If not or empty
%                      list is provided, defaults to 
%                      [min(x), max(x), min(y), max(y)].
% 
%     kx (optional): Degree of bivariate spline. If not or empty list is 
%                    provided, defaults to 3.
% 
%     ky (optional): Degree of bivariate spline. If not or empty list is 
%                    provided, defaults to 3.
% 
%     s (optional): Positive smoothing factor defined for estimation
%                   condition: 
%                   sum((w(i)*(z(i) - spline(x(i), y(i)))).^2) <= s.
%                   If not or empty list is provided, defaults to len(w).
% 
%     rankTol (optional): Threshold for determining the effective rank of 
%                         an over-determined linear system of equations. 
%                         Needs to be within open interval (0, 1). If not 
%                         or empty list is provided, defaults to 1e-16. It
%                         corresponds to eps argument in the function
%                         scipy.interpolate.SmoothBivariateSpline
%
%     scaling (optional): Optional array for scaling input data sequences
%                         [x, y, z] (provided in this order).
%                         Choosing it such that the 1D sequences fall in a
%                         similar numerical range increases robustness of
%                         the fitting routine. If not or empty list is
%                         provided, defaults to [1, 1, 1].
%
%     casadiOptions (optional): Struct containing options for the
%                               CasADi interpolant
% 
% Outputs:
%     spline: differentiable B-spline CasADi object function handle
%             z = f(x, y)
%     
% For more details on the SmoothBivariateSpline fitting and/or 
% Scipy-specific input arguments (i.e. all the arguments up to scaling), see 
% https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.SmoothBivariateSpline.html#scipy.interpolate.SmoothBivariateSpline
% 
% This file is part of the FORCESPRO client software for Matlab.
% (c) embotech AG, 2022-2023, Zurich, Switzerland. All rights reserved.
