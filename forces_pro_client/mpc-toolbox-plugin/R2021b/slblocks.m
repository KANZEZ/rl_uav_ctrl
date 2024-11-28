function [ blkStruct ] = slblocks()
%% FORCESPRO "mpc-toolbox-plugin" Simulink block utility

%   Author(s): Rong Chen, MathWorks Inc.
%
%   Copyright 2019-2023 The MathWorks, Inc.

    Browser.Library = 'forceslib';
    Browser.Name = 'FORCESPRO MPC Blocks';
    blkStruct.Browser = Browser; 
    
end
