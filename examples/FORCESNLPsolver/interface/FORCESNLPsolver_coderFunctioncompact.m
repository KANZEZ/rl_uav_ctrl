% FORCESNLPsolver : A fast customized optimization solver.
% 
% Copyright (C) 2013-2023 EMBOTECH AG [info@embotech.com]. All rights reserved.
% 
% 
% This software is intended for simulation and testing purposes only. 
% Use of this software for any commercial purpose is prohibited.
% 
% This program is distributed in the hope that it will be useful.
% EMBOTECH makes NO WARRANTIES with respect to the use of the software 
% without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
% PARTICULAR PURPOSE. 
% 
% EMBOTECH shall not have any liability for any damage arising from the use
% of the software.
% 
% This Agreement shall exclusively be governed by and interpreted in 
% accordance with the laws of Switzerland, excluding its principles
% of conflict of laws. The Courts of Zurich-City shall have exclusive 
% jurisdiction in case of any dispute.
% 
% [OUTPUTS] = FORCESNLPsolver(INPUTS) solves an optimization problem where:
% Inputs:
% - xinit - matrix of size [13x1]
% - x0 - matrix of size [51x1]
% Outputs:
% - outputs - column vector of length 51
function [outputs] = FORCESNLPsolver(xinit, x0)
    
    [output, ~, ~] = FORCESNLPsolverBuildable.forcesCall(xinit, x0);
    outputs = coder.nullcopy(zeros(51,1));
    outputs(1:17) = output.x1;
    outputs(18:34) = output.x2;
    outputs(35:51) = output.x3;
end
