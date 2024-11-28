% Returns options structure with default settings for generating code with 
% FORCESPRO.
% 
%    OPTS = FORCESGETDEFAULTOPTIONS returns a codeoptions struct with 
%    default solver name and generic default settings.
%
%    OPTS = FORCESGETDEFAULTOPTIONS(NAME,ALGORITHM,FLOATTYPE) returns 
%    a codeoptions struct with the solver named NAME, as well as default
%    settings tailored to the specified ALGORITHM and FLOATTYPE.
%
% For a detailed explanation of all possible options with FORCESPRO please
% consult the documentation at
% https://forces.embotech.com/Documentation/solver_options/index.html
%
% See also FORCES_NLP, GENERATECODE
%
%
% This file is part of the FORCESPRO client software for Matlab.
% (c) embotech AG, 2013-2023, Zurich, Switzerland. All rights reserved.
