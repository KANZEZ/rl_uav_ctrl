% Dumps a FORCESPRO problem formulation into a file to allow to exactly 
% reproduce the behaviour of a generated solver.
%
%   [TAG, FULLFILENAME] = ForcesDumpFormulation(MODELORFORMULATION, OPTIONS, OUTPUTS) stores the 
%   problem formulation into a json file with standardized naming. 
%   It returns a unique label TAG as a string that should be passed to ForcesDumpProblem  
%   when dumping actual problem instances. Besides, the returned
%   fullFilename string consists of the dump directory and the dump
%   filename.
%
%   [TAG, FULLFILENAME] = ForcesDumpFormulation(MODELORFORMULATION, OPTIONS, OUTPUTS, LABEL, DUMPDIRECTORY, DUMPTYPE)
%   provides addtional options.
%
%       MODELORFORMULATION:  For ForcesDumpType.DumpSymbolics (default):
%                            model as provided to FORCES_NLP or formulation struct.
%                            For ForcesDumpType.LegacyDumpGeneratedC: 
%                            formulation struct as returned as third argument by FORCES_NLP                    
%                            Note that CasADi MX expressions are not yet supported!
%       OPTIONS:             codeoptions as provided to FORCES_NLP
%       OUTPUTS:             outputs as provided to FORCES_NLP
%       LABEL:               optional, a custom label used inside the filename
%       DUMPDIRECTORY:       directory used to store the dumped problem formulation
%       DUMPTYPE:            any ForcesDumpType specifying the information to be dumped
%
% See also FORCES_NLP, ForcesDumpProblem, ForcesDumpType
% 
% This file is part of the FORCESPRO client software for Matlab.
% (c) embotech AG, 2013-2023, Zurich, Switzerland. All rights reserved.
