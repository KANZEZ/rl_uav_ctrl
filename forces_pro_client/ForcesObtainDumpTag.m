% Obtains a unique tag to be used to store dumped FORCESPRO formulations 
% and problems.
%
%   ForcesObtainDumpTag(FORMULATION, OPTIONS) returns a unique tag based
%   on the problem formulation and codeoptions.
%
%   ForcesObtainDumpTag(FORMULATION, OPTIONS, OUTPUTS, PARAMS, LABEL) returns 
%   a unique tag based on the following input arguments:
%
%       FORMULATION:   formulation or stages struct as returned by FORCES_NLP or generateCode
%       OPTIONS:       codeoptions as provided to FORCES_NLP or generateCode
%       OUTPUTS:       outputs as provided to FORCES_NLP or generateCode
%       PARAMS:        parameters as provided to generateCode
%       LABEL:         optional, a custom label used inside the filename
%
% See also ForcesDumpFormulation, ForcesDumpFormulationLowLevel, ForcesDumpProblem
%   
% This file is part of the FORCESPRO client software for Matlab.
% (c) embotech AG, 2013-2023, Zurich, Switzerland. All rights reserved.
