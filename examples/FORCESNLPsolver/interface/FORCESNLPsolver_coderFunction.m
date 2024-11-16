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
% - x0 - matrix of size [170x1]
% Outputs:
% - x01 - column vector of length 17
% - x02 - column vector of length 17
% - x03 - column vector of length 17
% - x04 - column vector of length 17
% - x05 - column vector of length 17
% - x06 - column vector of length 17
% - x07 - column vector of length 17
% - x08 - column vector of length 17
% - x09 - column vector of length 17
% - x10 - column vector of length 17
function [x01, x02, x03, x04, x05, x06, x07, x08, x09, x10] = FORCESNLPsolver(xinit, x0)
    
    [output, ~, ~] = FORCESNLPsolverBuildable.forcesCall(xinit, x0);
    x01 = output.x01;
    x02 = output.x02;
    x03 = output.x03;
    x04 = output.x04;
    x05 = output.x05;
    x06 = output.x06;
    x07 = output.x07;
    x08 = output.x08;
    x09 = output.x09;
    x10 = output.x10;
end
