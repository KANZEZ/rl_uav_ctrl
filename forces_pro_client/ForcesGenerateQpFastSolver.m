% Generate a fast QP solver for multistage problems using FORCESPRO.
%
%    success = ForcesGenerateQpFastSolver(stages,params,codeoptions,tuningoptions) 
%    generates, downloads and compiles your custom solver for the multistage 
%    problem stages. Parameters are as specified by the params struct, code 
%    settings are set through the codeoptions struct and options for tuning
%    the fast QP solver is set via the tuningoptions struct.
% 
%    success = ForcesGenerateQpFastSolver(stages,params,codeoptions,tuningoptions,outputs) 
%    does the above but with user defined outputs. Outputs are
%    defined by an array of structs obtained by newOutput, or you can also
%    define all variables by using getAllOutputs.
%
%    success = ForcesGenerateQpFastSolver(stages,params,codeoptions,tuningoptions,outputs,extra)   
%    does the above but with additional information about the method and 
%    the interface.
%
% SEE ALSO MULTISTAGEPROBLEM NEWPARAM NEWOUTPUT GETOPTIONS GENERATECODE
%
%
% This file is part of the FORCESPRO client software for Matlab.
% (c) embotech AG, 2013-2023, Zurich, Switzerland. All rights reserved.