% Generates a 2D spline object compatible with CasADi from the output
% object of the scipy interpolation and the corresponding scaling vector
%
% Inputs: 
%     scipyFit: 2D spline object, fitted by
%               scipy.interpolate.(Smooth|LSQ)BivariateSpline
% 
%     scaling: Utilized scaling vector for data x, y, z
% 
%     casadiOptions: Struct containing options for the CasADi interpolant
% 
% Outputs:
%     spline: differentiable B-spline CasADi object function handle
%             z = f(x, y)
% 
% This file is part of the FORCESPRO client software for Matlab.
% (c) embotech AG, 2022-2023, Zurich, Switzerland. All rights reserved.
