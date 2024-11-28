% Computes a unique hash of a dumped FORCESPRO formulation.
%
%   ForcesComputeDumpHash(FORMULATION, OPTIONS) returns a unique hash 
%   based on the problem formulation and codeoptions.
%
%   ForcesComputeDumpHash(FORMULATION, OPTIONS, OUTPUTS, PARAMS) returns a 
%   unique hash based on the following input arguments:
%
%       FORMULATION:   formulation or stages struct as returned by FORCES_NLP or generateCode
%       OPTIONS:       codeoptions as provided to FORCES_NLP or generateCode
%       OUTPUTS:       outputs as provided to FORCES_NLP or generateCode
%       PARAMS:        parameters as provided to generateCode
%
% See also ForcesDumpFormulation, ForcesDumpProblem
%   
% This file is part of the FORCESPRO client software for Matlab.
% (c) embotech AG, 2013-2023, Zurich, Switzerland. All rights reserved.
