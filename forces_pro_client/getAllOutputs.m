% Shortcut code to generate list of outputs such that all variables are
% included. 
%
%    OUTPUTS = getAllOutputs(STAGES) creates an output struct for use with
%    FORCESPRO that returns all variables as outputs, separated according
%    to stage. The default name of the variables is "z", appended by the
%    stage number.
%
%    OUTPUTS = getAllOutputs(STAGES, NAME) as above, but instead of "z" use
%    the string NAME to define the output. The stage number is
%    automatically appended.
%
% For a detailed explanation for declaring different outputs with FORCESPRO 
% please consult the documentation at
% https://forces.embotech.com/Documentation/high_level_interface/index.html#declaring-outputs
%
% See also NEWOUTPUT
%
% 
% This file is part of the FORCESPRO client software for Matlab.
% (c) embotech AG, 2013-2023, Zurich, Switzerland. All rights reserved.
