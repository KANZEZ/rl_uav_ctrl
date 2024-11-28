% Dumps a FORCESPRO problem formulation and/or problem instance into a file 
% to allow to exactly reproduce the behaviour of a generated solver.
%
%   [TAG, FULLFILENAME] = ForcesDumpAll(MODELORFORMULATION, OPTIONS, OUTPUTS, LABEL, DUMPDIRECTORY, PROBLEMS, DUMPTYPE, VARARGIN)
%   stores the problem formulation and problems into a json file with 
%   standardized naming. It returns a unique label TAG as a string for identifying the dump 
%   and the FULLFILENAME string consisting of the dump directory and the 
%   dump filename.
%
%       MODELORFORMULATION: For ForcesDumpType.DumpSymbolics (default):
%                           model as provided to FORCES_NLP
%                           For ForcesDumpType.LegacyDumpGeneratedC: 
%                           formulation struct as returned as third argument by FORCES_NLP                    
%                           Note that CasADi MX expressions are not yet supported!
%       OPTIONS:            codeoptions as provided to FORCES_NLP
%       OUTPUTS:            outputs as provided to FORCES_NLP
%       LABEL:              optional, a custom label used inside the filename
%       DUMPDIRECTORY:      directory used to store the dumped problem formulation
%       PROBLEMS:           a cell array of problem structs
%       DUMPTYPE:           any ForcesDumpType specifying the information to be dumped.
%                           Default: ForcesDumpType.DumpSymbolics
%       VARARGIN:           any additional data to be stored in the dump
%
% See also FORCES_NLP, ForcesDumpFormulation, ForcesDumpProblem, ForcesDumpType
%
% This file is part of the FORCESPRO client software for Matlab.
% (c) embotech AG, 2013-2023, Zurich, Switzerland. All rights reserved.
