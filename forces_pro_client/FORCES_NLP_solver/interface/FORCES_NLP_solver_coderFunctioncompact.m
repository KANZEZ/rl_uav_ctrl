% FORCES_NLP_solver : A fast customized optimization solver.
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
% [OUTPUTS] = FORCES_NLP_solver(INPUTS) solves an optimization problem where:
% Inputs:
% - xinit - matrix of size [6x1]
% - x0 - matrix of size [168x1]
% - all_parameters - matrix of size [21x1]
% - reinitialize - scalar
% Outputs:
% - outputs - column vector of length 168
function [outputs] = FORCES_NLP_solver(xinit, x0, all_parameters, reinitialize)
    
    [output, ~, ~] = FORCES_NLP_solverBuildable.forcesCall(xinit, x0, all_parameters, reinitialize);
    outputs = coder.nullcopy(zeros(168,1));
    outputs(1:8) = output.x01;
    outputs(9:16) = output.x02;
    outputs(17:24) = output.x03;
    outputs(25:32) = output.x04;
    outputs(33:40) = output.x05;
    outputs(41:48) = output.x06;
    outputs(49:56) = output.x07;
    outputs(57:64) = output.x08;
    outputs(65:72) = output.x09;
    outputs(73:80) = output.x10;
    outputs(81:88) = output.x11;
    outputs(89:96) = output.x12;
    outputs(97:104) = output.x13;
    outputs(105:112) = output.x14;
    outputs(113:120) = output.x15;
    outputs(121:128) = output.x16;
    outputs(129:136) = output.x17;
    outputs(137:144) = output.x18;
    outputs(145:152) = output.x19;
    outputs(153:160) = output.x20;
    outputs(161:168) = output.x21;
end
