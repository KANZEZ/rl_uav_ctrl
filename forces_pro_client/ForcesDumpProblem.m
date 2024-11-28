% Dumps a FORCESPRO problem instance into a file to allow to exactly 
% reproduce the behaviour of a generated solver.
%
%   FULLFILENAME = ForcesDumpProblem(PROBLEMS, TAG) stores the problem struct
%   into a json file with standardized naming. The returned string
%   FULLFILENAME consists of the dump directory and the dump filename.
%
%   FULLFILENAME = ForcesDumpProblem(PROBLEMS, TAG, DUMPDIRECTORY, DUMPTYPE) provides 
%   additional options.
%
%       PROBLEMS:       a cell array of problem structs
%       TAG:            optional, a unique label used inside the filename
%       DUMPDIRECTORY:  directory used to store the dumped problem instance
%       DUMPTYPE:       any ForcesDumpType specifying the information to be dumped.
%                       Default: ForcesDumpType.DumpSymbolics
%
% See also FORCES_NLP, ForcesDumpFormulation, ForcesDumpAll, ForcesDumpType
%   
% This file is part of the FORCESPRO client software for Matlab.
% (c) embotech AG, 2013-2023, Zurich, Switzerland. All rights reserved.
