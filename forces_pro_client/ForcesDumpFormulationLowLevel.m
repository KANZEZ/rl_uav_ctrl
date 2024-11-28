% Dumps a FORCESPRO low-level problem formulation into a file to allow to
% exactly reproduce the behaviour of a generated solver.
%
%   [TAG, FULLFILENAME] = ForcesDumpFormulationLowLevel(STAGES, OPTIONS, OUTPUTS, PARAMS) stores
%   the problem formulation into a json file with standardized naming.
%   It returns a unique label TAG as a string that should be passed to ForcesDumpProblemLowLevel
%   when dumping actual problem instances. Besides, the returned
%   FULLFILENAME string consists of the dump directory and the dump
%   filename.
%
%   [TAG, FULLFILENAME] = ForcesDumpFormulationLowLevel(STAGES, OPTIONS, OUTPUTS, PARAMS, LABEL, DUMPDIRECTORY)
%   provides additional options.
%
%       STAGES:              stages struct as provided to generateCode
%       OPTIONS:             codeoptions as provided to generateCode
%       OUTPUTS:             outputs as provided to generateCode
%       PARAMS:              parameters as provided to generateCode
%       LABEL:               optional, a custom label used inside the filename
%       DUMPDIRECTORY:       directory used to store the dumped problem formulation
%
% See also generateCode, ForcesDumpProblemLowLevel, ForcesDumpAllLowLevel, ForcesLoadLowLevelDump
% 
% This file is part of the FORCESPRO client software for Matlab.
% (c) embotech AG, 2013-2023, Zurich, Switzerland. All rights reserved.
