% Loads a FORCESPRO low-level dump from given json files.
%
%   [STAGES, OPTIONS, OUTPUTS, PARAMS, PROBLEMS] = ForcesLoadLowLevelDump(FORMULATIONFILENAME, PROBLEMFILENAMES)
%   loads all objects needed for generating and running a dumped solver from json file(s).
%
%       FORMULATIONFILENAME: file created by ForcesDumpFormulationLowLevel or ForcesDumpAllLowLevel
%       PROBLEMFILENAMES:    file(s) created by ForcesDumpProblemLowLevel
%
%       STAGES:              stages struct as provided to generateCode
%       OPTIONS:             codeoptions as provided to generateCode
%       OUTPUTS:             outputs as provided to generateCode
%       PARAMS:              parameters as provided to generateCode
%       PROBLEMS:            a struct array of problem structs
%
% Backwards compatibility: low-level dumps created with FORCESPRO
% version 6.1.0 or higher are supported.
%
% See also generateCode, ForcesDumpFormulationLowLevel, ForcesDumpProblemLowLevel, ForcesDumpAllLowLevel
%
% This file is part of the FORCESPRO client software for Matlab.
% (c) embotech AG, 2013-2023, Zurich, Switzerland. All rights reserved.
