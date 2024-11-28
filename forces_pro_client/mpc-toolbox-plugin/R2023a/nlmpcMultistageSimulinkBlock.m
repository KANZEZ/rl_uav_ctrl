function [mv, cost, mvseq, xseq, nlpstatus, nlpsolvetime, solverid, z0next] = nlmpcMultistageSimulinkBlock(x, lastMV, md, stateparam, stageparam, MVMin, MVMax, MVRateMin, MVRateMax, StateMin, StateMax, xTerminal, z0ext, z0, blkData)
%% Interface between MPC Toolbox and FORCESPRO NLP Solver (internal)
%
%  This function contains the functionality of the nlmpc Multistage 
%  Simulink Block.
    
%   Author(s): Rong Chen, MathWorks Inc.
%
%   Copyright 2020-2023 The MathWorks, Inc.

    %#codegen
    coder.allowpcode('plain');

    %% Interface to solver
    returninfo = blkData.cost_enabled || blkData.mvseq_enabled || blkData.stateseq_enabled || blkData.status_enabled || blkData.solvetime_enabled || blkData.solverid_enabled || ~blkData.nlp_initialize;
    onlinedata = struct;
    if blkData.md_enabled
        onlinedata.md = md;
    end
    if blkData.stateparam_enabled
        onlinedata.StateFcnParameter = stateparam;
    end
    if blkData.stageparam_enabled
        onlinedata.StageParameter = stageparam;
    end
    if blkData.mv_min
        onlinedata.MVMin = MVMin;
    end
    if blkData.mv_max
        onlinedata.MVMax = MVMax;
    end
    if blkData.state_min
        onlinedata.StateMin = StateMin;
    end
    if blkData.state_max
        onlinedata.StateMax = StateMax;
    end
    if blkData.mvrate_min
        onlinedata.MVRateMin = MVRateMin;
    end
    if blkData.mvrate_max
        onlinedata.MVRateMax = MVRateMax;
    end
    if blkData.nlp_initialize
        onlinedata.InitialGuess = z0ext;
    else   
        onlinedata.InitialGuess = z0;
    end
    % call FORCES MEX solver
    % coder require pre-define outputs for fixed-size data code generation
    %mv = lastMV;
    %newonlinedata = onlinedata;
    % info structure must be the same as "nlmpcmoveForces" produces
    if strcmp(blkData.SolverType,'SQP')
        info = struct('MVopt',zeros(blkData.N,blkData.nmv),...
                      'Xopt',zeros(blkData.N,blkData.nx),...
                      'Topt',zeros(blkData.N,1),...
                      'ExitFlag',1,...
                      'Iterations',1,...
                      'Cost',0,...
                      'EqualityResidual',0,...
                      'SolveTime',0,...
                      'FcnEvalTime',0,...
                      'QPTime',0,...
                      'SolverId',zeros(8,1));
    else
        info = struct('MVopt',zeros(blkData.N,blkData.nmv),...
                      'Xopt',zeros(blkData.N,blkData.nx),...
                      'Topt',zeros(blkData.N,1),...
                      'ExitFlag',1,...
                      'Iterations',1,...
                      'Cost',0,...
                      'EqualityResidual',0,...
                      'InequalityResidual',0,...
                      'SolveTime',0,...
                      'FcnEvalTime',0,...
                      'SolverId',zeros(8,1));
    end
    if returninfo
        [mv,newonlinedata,info] = nlmpcmoveForcesMultistage(blkData,x,lastMV,onlinedata);
    else
        [mv,newonlinedata] = nlmpcmoveForcesMultistage(blkData,x0,u0,onlinedata);
    end
    cost = info.Cost;
    mvseq = info.MVopt;
    xseq = info.Xopt;
    nlpstatus = info.ExitFlag;
    nlpsolvetime = info.SolveTime;
    solverid = info.SolverId;
    z0next = newonlinedata.InitialGuess;
end
