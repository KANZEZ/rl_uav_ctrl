% Shortcut code to define an output of the FORCESPRO solver.
% 
%    OUTVAR = NEWOUTPUT(NAME, FROMSTAGE, IDXWITHINSTAGE) returns a struct
%    that defines a valid output to be used with the FORCESPRO code generator,
%    where NAME is a label of the output, FROMSTAGE defines from which
%    stage variable the output is retrieved, and IDXWITHINSTAGE defines the
%    indices within that stage variable.
%
%    Example: 1) to have variables 5 to 8 from stage variable 11 as outputs 
%                of the generated solver, call
%                    
%                   output = newOutput('myoutput', 11, 5:8);
%
%             2) to collect all variables with index 3 and 4 from all
%                stages N, call
%
%                   output = newOutput('myoutput', 1:N, 3:4);
%
% For a detailed explanation for declaring different outputs with FORCESPRO 
% please consult the documentation
%
% % For the "low-level" interface:
% https://forces.embotech.com/Documentation/low_level_interface/index.html#declaring-solver-outputs
%
% For the "high-level" interface:
% https://forces.embotech.com/Documentation/high_level_interface/index.html#declaring-outputs
%
% See also GETALLOUTPUTS FORCESVERSION
%
%
% This file is part of the FORCESPRO client software for Matlab.
% (c) embotech AG, 2013-2023, Zurich, Switzerland. All rights reserved.
