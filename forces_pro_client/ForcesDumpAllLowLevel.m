% Dumps a FORCESPRO low-level problem formulation and/or problem instance 
% into a file to allow to exactly reproduce the behaviour of a generated solver.
%
%   [TAG, FULLFILENAME] = ForcesDumpAllLowLevel(STAGES, OPTIONS, OUTPUTS, PARAMS, LABEL, DUMPDIRECTORY, PROBLEMS, VARARGIN) 
%   stores the problem formulation and problems into a json file with 
%   standardized naming. It returns a unique label TAG as a string for identifying the dump 
%   and the FULLFILENAME string consisting of the dump directory and the 
%   dump filename.
%
%       STAGES:             stages struct as provided to generateCode
%       OPTIONS:            codeoptions as provided to FORCES_NLP
%       OUTPUTS:            outputs as provided to FORCES_NLP
%       PARAMS:             parameters as provided to generateCode
%       LABEL:              optional, a custom label used inside the filename
%       DUMPDIRECTORY:      directory used to store the dumped problem formulation
%       PROBLEMS:           a cell array of problem structs
%       VARARGIN:           any additional data to be stored in the dump
%
% See also generateCode, ForcesDumpFormulationLowLevel, ForcesDumpProblemLowLevel, ForcesLoadLowLevelDump
%
% This file is part of the FORCESPRO client software for Matlab.
% (c) embotech AG, 2013-2023, Zurich, Switzerland. All rights reserved.
