function [xkkp1,mv,cost,mvseq,xseq,yseq,exitflag,solvetime,solverid,xkk,Pkk] = mpcSparseSimulinkBlock(xkkm1,lastmv,moorx,yref,md,extmv,umin,umax,dumin,dumax,ymin,ymax,ywt,uwt,duwt,ecrwt,uref,A,B,C,D,U,Y,X,DX,Pkk,MPCstruct,isEmptyMPC,mpcVariant)
    %% Interface between MPC Toolbox and FORCESPRO Sparse QP Solver (internal)
    %
    %  This function contains the functionality of the mpc Sparse QP
    %  Simulink Block.

    % The input "mpcVariant" is an integer flag specifying, which of the 
    % linear mpc plugin is calling this script. The mapping is as follows
    %       0: linear 
    %       1: adaptive 
    %       2: ltv

    %   Author(s): Rong Chen, MathWorks Inc.
    %              Konstantinos Lekkas, Embotech AG
    %              Nivethan Yogarajah, Embotech AG
    %
    %   Copyright 2020-2023 The MathWorks, Inc. and Embotech AG


    %#codegen
    coder.allowpcode('plain');

    if isEmptyMPC
        [xkkp1,mv,cost,mvseq,xseq,yseq,exitflag,solvetime,solverid,xkk,Pkk] = deal(0);
        return;
    end

    %% states
    statedata.Plant = reshape(xkkm1(1:MPCstruct.nxp),MPCstruct.nxp,1);
    statedata.Disturbance = reshape(xkkm1(MPCstruct.nxp+1:MPCstruct.nxp+MPCstruct.nxdist),MPCstruct.nxdist,1);
    statedata.Noise = reshape(xkkm1(MPCstruct.nxp+MPCstruct.nxdist+1:MPCstruct.nxp+MPCstruct.nxdist+MPCstruct.nxnoise),MPCstruct.nxnoise,1);
    statedata.Covariance = Pkk;
    statedata.LastMove = reshape(lastmv,MPCstruct.nmv,1);
    statedata.warmstart = int32(0); 
    %% ref
    onlinedata.signals.ref = reshape(yref,numel(yref)/MPCstruct.ny,MPCstruct.ny);
    %% md if requested
    if MPCstruct.nmd>0
        onlinedata.signals.md = reshape(md,numel(md)/MPCstruct.nmd,MPCstruct.nmd);
    end
    %% ym if built-in estimation is used
    if ~MPCstruct.IsCustomEstimation
        onlinedata.signals.ym = reshape(moorx,MPCstruct.nmy,1);
    end
    %% externalMV if requested
    if MPCstruct.UseExternalMV
        onlinedata.signals.externalMV = reshape(extmv,MPCstruct.nmv,1);
    end
    %% mvTarget if requested
    if MPCstruct.UseMVTarget
        onlinedata.signals.mvTarget = reshape(uref,numel(uref)/MPCstruct.nmv,MPCstruct.nmv);
    end
    %% weights
    if MPCstruct.UseOnlineWeightOV
        onlinedata.weights.y = appendRows(ywt,MPCstruct.p,MPCstruct.ny);
    end
    if MPCstruct.UseOnlineWeightMV
        onlinedata.weights.u = appendRows(uwt,MPCstruct.p,MPCstruct.nmv);
    end
    if MPCstruct.UseOnlineWeightMVRate
        onlinedata.weights.du = appendRows(duwt,MPCstruct.p,MPCstruct.nmv);
    end
    if MPCstruct.UseOnlineWeightECR
        onlinedata.weights.ecr = ecrwt;
    end
    %% constraints
    if MPCstruct.UseOnlineConstraintOVMax
        onlinedata.limits.ymax = appendRows(ymax,MPCstruct.p,MPCstruct.ny);
    end
    if MPCstruct.UseOnlineConstraintOVMin
        onlinedata.limits.ymin = appendRows(ymin,MPCstruct.p,MPCstruct.ny);
    end
    if MPCstruct.UseOnlineConstraintMVMax
        onlinedata.limits.umax = appendRows(umax,MPCstruct.p,MPCstruct.nmv);
    end
    if MPCstruct.UseOnlineConstraintMVMin
        onlinedata.limits.umin = appendRows(umin,MPCstruct.p,MPCstruct.nmv);
    end
    if MPCstruct.UseOnlineConstraintMVRateMax
        onlinedata.limits.dumax = appendRows(dumax,MPCstruct.p,MPCstruct.nmv);
    end
    if MPCstruct.UseOnlineConstraintMVRateMin
        onlinedata.limits.dumin = appendRows(dumin,MPCstruct.p,MPCstruct.nmv);
    end
    %% Plant and Nominal
    if mpcVariant > 0
        onlinedata.model.A = A;
        onlinedata.model.B = B;
        onlinedata.model.C = C;
        onlinedata.model.D = D;
    
        onlinedata.model.U = U;
        onlinedata.model.Y = Y;
        onlinedata.model.X = X;
        onlinedata.model.DX = DX;
    end

    %% Solve
    mv = lastmv;
    newstatedata = statedata;
    info = struct('Uopt',zeros(MPCstruct.p,MPCstruct.nmv),...
                  'DUopt',zeros(MPCstruct.p,MPCstruct.nmv),...
                  'Yopt',zeros(MPCstruct.p,MPCstruct.ny),...
                  'Xopt',zeros(MPCstruct.p,MPCstruct.nxQP),...
                  'Slack',zeros(MPCstruct.p,1),...
                  'ExitFlag',0,...
                  'Iterations',0,...
                  'SolveTime',0,...
                  'SolverId',zeros(8,1),...
                  'Cost',0);
    if mpcVariant == 0
        [mv, newstatedata, info] = mpcmoveForces(MPCstruct, statedata, onlinedata);
    else
        [mv, newstatedata, info] = mpcmoveAdaptiveForces(MPCstruct, statedata, onlinedata);
    end
    xkkp1 = [newstatedata.Plant;newstatedata.Disturbance;newstatedata.Noise];
    cost = info.Cost;
    mvseq = info.Uopt;
    xseq = info.Xopt;
    yseq = info.Yopt;
    if info.ExitFlag > 0
        exitflag = info.Iterations;
    elseif info.ExitFlag == 0
        exitflag = 0;
    else
        exitflag = info.ExitFlag;
    end
    solvetime = info.SolveTime;
    solverid = info.SolverId;
    xkk = info.CurrentState;
    Pkk = newstatedata.Covariance;
end

function val = appendRows(data,p,cols)
    rows0 = numel(data)/cols;
    val0 = reshape(data,rows0,cols);
    val = zeros(p,cols);
    for ct=1:p
        val(ct,:) = val0(min(rows0,ct),:);
    end
end
