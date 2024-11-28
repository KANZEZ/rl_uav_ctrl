% Generates a 2D spline object compatible with CasADi for gridded data.
% In particular, it uses the 'bspline' plugin of CasADi interpolant
% (i.e. casadi.interpolant('spline', 'bspline', ...))
%
% It additionally supports scaling of the I/O, which affects the numerical 
% stability of the spline fitting routine. The scaling is automatically 
% taken care for, so that the I/O to the resulting spline are still the 
% physical quantities.
% 
% Inputs:
%     xgrid: 1D sequence of grid points in strictly ascending order
% 
%     ygrid: 1D sequence of grid points in strictly ascending order
%
%     Z: 2D array representing matrix-indexed meshgrid of z values
%        corresponding to xgrid and ygrid data sequences. Hence, Z has the
%        size [length(xgrid), length(ygrid)]
% 
%        Example with 2D polynomial function:
%            xgrid = -5:1:5;
%            ygrid = -4:1:4;
%            [X, Y] = meshgrid(xgrid, ygrid) % Cartesian-indexed mesh-grid
%            ZCartesian = X.^2 + Y.^3 + 3;
%            Z = ZCartesian.'; % switch indexing: Cartesian -> matrix
% 
%     scaling (optional): Optional list for scaling input data sequences 
%                         [x, y, z] (provided in this order). 
%                         Choosing it such that the 1D sequences fall in a 
%                         similar numerical range increases robustness of 
%                         the fitting routine. Defaults to [1, 1, 1].
% 
%     casadiOptions (optional): Struct containing non-default options for
%                               the 'bspline' CasADi interpolant plugin
% 
%                               Example to change the default
%                               degree of the bivariate spline:
%                               casadiOptions = struct('degree', [kx, ky])
% 
% Outputs: 
%     spline: differentiable B-spline CasADi object function handle 
%             z = f(x, y)
%
% This file is part of the FORCESPRO client software for Matlab.
% (c) embotech AG, 2022-2023, Zurich, Switzerland. All rights reserved.
