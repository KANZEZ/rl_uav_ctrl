% Loads a FORCESPRO symbolic dump from given json files.
%
%   [MODEL, OPTIONS, OUTPUTS, ADDITIONALDATA, PROBLEMS] = ForcesLoadSymbolicDump(FORMULATIONFILENAME, PROBLEMFILENAMES)
%
%       FORMULATIONFILENAME: file created by ForcesDumpFormulation or ForcesDumpAll
%       PROBLEMFILENAMES:    file(s) created by ForcesDumpProblem
%
%       MODEL:               model as provided to FORCES_NLP
%                            (CasADi MX expressions not yet supported)
%       OPTIONS:             codeoptions as provided to FORCES_NLP
%       OUTPUTS:             outputs as provided to FORCES_NLP
%       ADDITIONALDATA:      any additional structs that may have been dumped.
%       PROBLEMS:            a struct array of problem structs
%
% This file is part of the FORCESPRO client software for Matlab.
% (c) embotech AG, 2013-2023, Zurich, Switzerland. All rights reserved.
