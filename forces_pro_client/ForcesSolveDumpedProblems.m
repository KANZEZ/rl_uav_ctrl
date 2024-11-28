% Solves certain dumped FORCESPRO problems by first generating a solver 
% based on the dumped formulation and then solving the dumped problems.
%
%   ForcesSolveDumpedProblems(TAG, DUMPDIRECTORY) returns arrays with outputs, 
%   exitflags and infos obtained when solving dumped problems identified by 
%   the given tag.
%
%       TAG:            optional, a unique label used inside the filename
%       DUMPDIRECTORY:  directory used to look for dumped problem instances
%
% See also ForcesDumpFormulation, ForcesDumpProblem, FORCES_NLP
%   
% This file is part of the FORCESPRO client software for Matlab.
% (c) embotech AG, 2013-2023, Zurich, Switzerland. All rights reserved.
