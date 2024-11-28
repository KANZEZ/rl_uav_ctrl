function [mv, cost, mvseq, xseq, yseq, nlpstatus, nlpsolvetime, solverid, X0, MV0] = nlmpcSimulinkBlock(x, ref, lastMV, md, MVTarget, MVMin, MVMax, OutputMin, OutputMax, MVRateMin, MVRateMax, StateMin, StateMax, OutputWeights, MVWeights, MVRateWeights, ECRWeight, Parameter, MV0, X0, blkData)
%% Interface between MPC Toolbox and FORCESPRO NLP Solver (internal)
%
%  This function contains the functionality of the nlmpc Simulink 
%  Block.
    
%   Author(s): Rong Chen, MathWorks Inc.
%
%   Copyright 2020-2023 The MathWorks, Inc.

    %#codegen
    coder.allowpcode('plain');

    %% Interface to solver
    returninfo = blkData.cost_enabled || blkData.mvseq_enabled || blkData.stateseq_enabled || blkData.ovseq_enabled || blkData.status_enabled || blkData.solvetime_enabled || blkData.solverid_enabled || ~blkData.nlp_initialize;
    onlinedata = struct;
    onlinedata.ref = ref;
    if blkData.mvtarget_enabled
        onlinedata.MVTarget = MVTarget;
    else
        onlinedata.MVTarget = zeros(1,blkData.nmv);
    end
    if blkData.md_enabled
        onlinedata.md = md;
    end
    if blkData.HasParameter
        onlinedata.Parameter = Parameter;
    end
    if blkData.mv_min
        onlinedata.MVMin = MVMin;
    end
    if blkData.mv_max
        onlinedata.MVMax = MVMax;
    end
    if blkData.ov_min
        onlinedata.OutputMin = OutputMin;
    end
    if blkData.ov_max
        onlinedata.OutputMax = OutputMax;
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
    if blkData.ov_weight
        onlinedata.OutputWeights = OutputWeights;
    end
    if blkData.mv_weight
        onlinedata.MVWeights = MVWeights;
    end
    if blkData.mvrate_weight
        onlinedata.MVRateWeights = MVRateWeights;
    end
    if blkData.ecr_weight
        onlinedata.ECRWeight = ECRWeight;
    end
    % Inputs xinit, mvinit, einit contain the solution from the previous sample
    % time.  Use as warm start.  At t=0, these are initialized to x0, mv0,
    % slack0. User can provide custom initialization signals.  Here xinit is
    % [1:p]-by-nx from k+1, mvinit is [1:p]-by-nmv from k, einit is a scalar.
    onlinedata.X0 = X0;
    onlinedata.MV0 = MV0;
    % call FORCES MEX solver
    % coder require pre-define outputs for fixed-size data code generation
    %mv = lastMV;
    %newonlinedata = onlinedata;
    % info structure must be the same as "nlmpcmoveForces" produces
    if strcmp(blkData.SolverType,'SQP')
        info = struct('MVopt',zeros(blkData.p+1,blkData.nmv),...
                      'Xopt',zeros(blkData.p+1,blkData.nx),...
                      'Yopt',zeros(blkData.p+1,blkData.ny),...
                      'Topt',zeros(blkData.p+1,1),...
                      'ExitFlag',1,...
                      'Iterations',1,...
                      'Cost',0,...
                      'EqualityResidual',0,...
                      'SolveTime',0,...
                      'FcnEvalTime',0,...
                      'QPTime',0,...
                      'SolverId',zeros(8,1));
    else
        info = struct('MVopt',zeros(blkData.p+1,blkData.nmv),...
                      'Xopt',zeros(blkData.p+1,blkData.nx),...
                      'Yopt',zeros(blkData.p+1,blkData.ny),...
                      'Topt',zeros(blkData.p+1,1),...
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
        [mv,newonlinedata,info] = nlmpcmoveForces(blkData,x,lastMV,onlinedata);
    else
        [mv,newonlinedata] = nlmpcmoveForces(blkData,x0,u0,onlinedata);
    end
    cost = info.Cost;
    mvseq = info.MVopt;
    xseq = info.Xopt;
    yseq = info.Yopt;
    nlpstatus = info.ExitFlag;
    nlpsolvetime = info.SolveTime;
    solverid = info.SolverId;
    X0 = newonlinedata.X0;
    MV0 = newonlinedata.MV0;
end
