function [] = forcesmpc_timevarying(SkipSolverGeneration)
%% Time-Varying MPC Control of a Time-Varying Plant
% This example shows how time-varying prediction models can be used to
% achieve better performance when controlling a time-varying plant. It also
% shows how to use the FORCESPRO plugin for the MathWorks Model Predictive
% Control Toolbox.
%
% If the Matlab Coder from MathWorks is installed, this script will use the
% automatically generated MEX function versions of mpmoveForces and
% mpcmoveAdaptiveForces respectively.
%
% If Simulink is installed, this script will additionally simulate the
% system inside Simulink using the FORCESPRO plugin Simulink blocks.
% 
% This example requires the Model Predictive Control Toolbox from
% MathWorks.
%
% The following MPC controllers are compared:
% 1. Linear MPC controller based on a time-invariant average model
% 2. Linear MPC controller based on a time-invariant model, which is
%    updated at each time step.
% 3. Linear MPC controller based on a time-varying prediction model.
 
% Copyright 2012-2023 The MathWorks, Inc. and Embotech AG, Zurich, Switzerland
    
    coderIsInstalled = mpcchecktoolboxinstalled('matlabcoder');
    simulinkIsInstalled = mpcchecktoolboxinstalled('simulink');

    if nargin < 1
        % (1) Skip the FORCESPRO solver generation step (assumes the 
        % solvers have already been generated in a previous function call
        % (0) (Re-)generate the FORCESPRO solvers
        SkipSolverGeneration = 0;
    end

    close all; clc;
    
    cd(fileparts(which(mfilename())));
    curDir = pwd();
    
    %% Initialization
    % Set the simulation duration in seconds.
    Ts = 0.1;
    Tstop = 25;
    Nsim = round(Tstop/Ts) + 1;

    % MPC controller object
    p = 3;                                  % prediction horizon
    m = 3;                                  % control horizon
    mpcobj = setupMPC(Ts, p, m);

    % Time-varying model
    Models = setupPlant(Ts, Tstop, p);

    % Set the initial plant states to zero.
    nx = length(mpcobj.Model.Nominal.X);
    ny = length(mpcobj.Model.Nominal.Y);
    nu = length(mpcobj.Model.Nominal.U);
    x0 = zeros(nx, 1);
    
    %% Solver Generation                                                     
    % LTI
    solverDirLTI = fullfile(curDir, 'LTI');
    setupSolverDir(solverDirLTI, SkipSolverGeneration);
    
    optionsLTI = mpcToForcesOptions(); 
    optionsLTI.SolverName = 'LTICustomForcesSparseQP';
    optionsLTI.SkipSolverGeneration = SkipSolverGeneration;
    
    cd(solverDirLTI)
    [coredataLTI, statedataLTI, onlinedataLTI] = mpcToForces(mpcobj, optionsLTI);
    
    % Adaptive
    solverDirAdaptive = fullfile(curDir, 'Adaptive');
    setupSolverDir(solverDirAdaptive, SkipSolverGeneration);
    
    optionsAdaptive = mpcToForcesOptions(); 
    optionsAdaptive.SolverName = 'AdaptiveCustomForcesSparseQP';
    optionsAdaptive.SkipSolverGeneration = SkipSolverGeneration;
    optionsAdaptive.UseAdaptive = 1;
    
    cd(solverDirAdaptive)
    [coredataAdaptive, statedataAdaptive, onlinedataAdaptive] = mpcToForces(mpcobj, optionsAdaptive);
    
    % LTV
    solverDirLTV = fullfile(curDir, 'LTV');
    setupSolverDir(solverDirLTV, SkipSolverGeneration);
    
    optionsLTV = mpcToForcesOptions();
    optionsLTV.SolverName = 'LTVCustomForcesSparseQP';
    optionsLTV.SkipSolverGeneration = SkipSolverGeneration;
    optionsLTV.UseLTV = 1;
    
    cd(solverDirLTV)
    [coredataLTV, statedataLTV, onlinedataLTV] = mpcToForces(mpcobj, optionsLTV);
    
    %% References
    stepSwitch = round(Nsim/3);
    references = ones(Nsim+p, 1);
    references(stepSwitch:2*stepSwitch-1) = 0.6;
    references(2*stepSwitch:end) = 0.25;

    %% Simulation   
    % LTI
    yyMPCFORCES = zeros(ny, Nsim);
    uuMPCFORCES = zeros(nu, Nsim);
    x = x0;
    fprintf('Simulating FORCESPRO MPC controller based on average LTI model.\n');
    for ct = 1:Nsim
        % Get the real plant.
        real_plant = Models(:,:,ct);
        % Update and store the plant output.
        y = real_plant.C*x;
        yyMPCFORCES(ct) = y;
        % Compute and store the MPC optimal move.
        onlinedataLTI.signals.ym = y;
        onlinedataLTI.signals.ref = references(ct:ct+p-1);
        if coderIsInstalled
            % MEX function name is mpcmove_<optionsLTI.SolverName>
            % It is not necessary to supply the constant coredataLTI (i.e.
            % it was already considered during code generation)
            [mv, statedataLTI, ~] = mpcmove_LTICustomForcesSparseQP(statedataLTI, onlinedataLTI);
        else
            [mv, statedataLTI, ~] = mpcmoveForces(coredataLTI, statedataLTI, onlinedataLTI);
        end
        uuMPCFORCES(ct) = mv;
        % Update the plant state.
        x = real_plant.A*x + real_plant.B*mv;
    end
    
    % Adaptive
    yyAMPCFORCES = zeros(ny, Nsim);
    uuAMPCFORCES = zeros(nu, Nsim);
    x = x0;
    nominal = mpcobj.Model.Nominal; % Nominal conditions are constant over the prediction horizon.
    fprintf('Simulating FORCESPRO MPC controller based on LTI model, updated at each time step t.\n');
    for ct = 1:Nsim
        % Get the real plant.
        real_plant = Models(:,:,ct);
        % Update and store the plant output.
        y = real_plant.C*x;
        yyAMPCFORCES(ct) = y;    
        % Compute and store the MPC optimal move.
        onlinedataAdaptive.signals.ym = y;
        onlinedataAdaptive.signals.ref = references(ct:ct+p-1);
    
        onlinedataAdaptive.model.X = nominal.X;
        onlinedataAdaptive.model.U = nominal.U; 
        onlinedataAdaptive.model.Y = nominal.Y;
        onlinedataAdaptive.model.DX = nominal.DX;
    
        onlinedataAdaptive.model.A = real_plant.A;
        onlinedataAdaptive.model.B = real_plant.B;
        onlinedataAdaptive.model.C = real_plant.C;
        onlinedataAdaptive.model.D = real_plant.D;
        
        if coderIsInstalled
            % MEX function name is mpcmoveAdaptive_<optionsAdaptive.SolverName>
            % It is not necessary to supply the constant coredataAdaptive 
            % (i.e. it was already considered during code generation)
            [mv, statedataAdaptive, ~] = mpcmoveAdaptive_AdaptiveCustomForcesSparseQP(statedataAdaptive, onlinedataAdaptive);
        else
            [mv, statedataAdaptive, ~] = mpcmoveAdaptiveForces(coredataAdaptive, statedataAdaptive, onlinedataAdaptive);
        end
        uuAMPCFORCES(ct) = mv;
        % Update the plant state.
        x = real_plant.A*x + real_plant.B*mv;
    end
    
    % LTV
    yyLTVMPCFORCES = zeros(ny, Nsim);
    uuLTVMPCFORCES = zeros(nu, Nsim);
    x = x0;
    Nominals = repmat(mpcobj.Model.Nominal,p+1,1); % Nominal conditions are constant over the prediction horizon.
    fprintf('Simulating FORCESPRO MPC controller based on time-varying model, updated at each time step t.\n');
    for ct = 1:Nsim
        % Get the real plant.
        real_plant = Models(:,:,ct);
        % Update and store the plant output.
        y = real_plant.C*x;
        yyLTVMPCFORCES(ct) = y;
        % Compute and store the MPC optimal move.
        onlinedataLTV.signals.ym = y;
        onlinedataLTV.signals.ref = references(ct:ct+p-1);        
        
        for k=1:p+1
            onlinedataLTV.model.X(:, 1, k) = Nominals(k).X; 
            onlinedataLTV.model.DX(:, 1, k) = Nominals(k).DX; 
            onlinedataLTV.model.U(:, 1, k) = Nominals(k).U; 
            onlinedataLTV.model.Y(:, 1, k) = Nominals(k).Y; 
    
            model = Models(:, :, ct+k-1);
            onlinedataLTV.model.A(:, :, k) = model.A;
            onlinedataLTV.model.B(:, :, k) = model.B;
            onlinedataLTV.model.C(:, :, k) = model.C;
            onlinedataLTV.model.D(:, :, k) = model.D;
        end
        
        if coderIsInstalled
            % MEX function name is mpcmoveAdaptive_<optionsLTV.SolverName>
            % It is not necessary to supply the constant coredataLTV (i.e.
            % it was already considered during code generation)
            [mv, statedataLTV, ~] = mpcmoveAdaptive_LTVCustomForcesSparseQP(statedataLTV, onlinedataLTV);
        else
            [mv, statedataLTV, ~] = mpcmoveAdaptiveForces(coredataLTV, statedataLTV, onlinedataLTV);
        end
        uuLTVMPCFORCES(ct) = mv;
        % Update the plant state.
        x = real_plant.A*x + real_plant.B*mv;
    end
    
    %% Plot                                                                  
    figure(1);
    fontSize = 15;
    t = 0:Ts:Tstop;

    blue = [0, 0.4470, 0.7410];
    orange = [0.8500, 0.3250, 0.0980];
    yellow = [0.9290, 0.6940, 0.1250];
    
    hold on;
    plot(t, yyMPCFORCES, 'Color', blue, 'DisplayName', 'LTI');
    plot(t, yyAMPCFORCES, 'Color', orange, 'DisplayName', 'Adaptive');
    plot(t, yyLTVMPCFORCES, 'Color', yellow, 'DisplayName', 'LTV');
    plot(t, references(1:Nsim), 'k--', 'DisplayName', 'Reference');
    hold off;
    title('FORCESPRO: Closed-Loop Output', 'FontSize', fontSize);
    xlabel('t [s]', 'FontSize', fontSize);
    ylabel('y', 'FontSize', fontSize);
    xlim([t(1), t(end)]);
    grid on;
    legend('Location','NorthEast');

    %% Simulink
    if simulinkIsInstalled
        % We can use the solvers generated previously, but we need to use 
        % the original initial statedata for the kalman filter to work 
        % properly. Therefore, we call again mpcToForces.m without
        % generating a new solver.
        optionsLTI.SkipSolverGeneration = 1;
        optionsAdaptive.SkipSolverGeneration = 1;
        optionsLTV.SkipSolverGeneration = 1;
        
        [~, statedataLTI, ~] = mpcToForces(mpcobj, optionsLTI);
        [~, statedataAdaptive, ~] = mpcToForces(mpcobj, optionsAdaptive);
        [~, statedataLTV, ~] = mpcToForces(mpcobj, optionsLTV);

        % Run the Simulink simulation
        cd(curDir);
               
        mdlForces = 'mpc_forces';
        open_system(mdlForces);
        
        xmpc = mpcstate(mpcobj);    
        simulinkOptions = simset('srcworkspace','current');
        sim(mdlForces, Tstop, simulinkOptions);
        open_system([mdlForces, '/Scope']);
        
        fprintf('Simulating MPC controller using Matlab LTI/Adaptive/LTV MPC Simulink Blocks as well as FORCESPRO LTI/Adaptive/LTV MPC Simulink Blocks.\n');
    end
        
    %% Clean Up  
    cd(curDir);
    rmpath(solverDirLTI);
    rmpath(solverDirAdaptive);
    rmpath(solverDirLTV);
end

%% Utility functions
function [Models] = setupPlant(Ts, Tstop, p)
% Time-Varying Linear Plant   
%
% In this example, the plant is a single-input-single-output 3rd order
% time-varying linear system with poles, zeros and gain that vary
% periodically with time.
% 
% $$G = \frac{{5s + 5 + 2\cos \left( {2.5t} \right)}}{{{s^3} 
%   + 3{s^2} + 2s + 6 + \sin \left( {5t} \right)}}$$
% 
% The plant poles move between being stable and unstable at run time,
% which leads to a challenging control problem.
%
% The plant is taken from MathWorks' example: 
% https://ch.mathworks.com/help/mpc/ug/time-varying-mpc-control-of-a-time-varying-linear-system.html
    
    % Generate an array of plant models at |t| = |0|, |Ts|, |2*Ts|, ...,
    % |Tstop + p*Ts| seconds.
    Models = tf;
    ct = 1;
    for t = 0:Ts:(Tstop + p*Ts)
        Models(:,:,ct) = tf([5 5+2*cos(2.5*t)],[1 3 2 6+sin(5*t)]);
        ct = ct + 1;
    end
    
    % Convert the models to state-space format and discretize them with a
    % sample time of Ts second.
    Models = ss(c2d(Models,Ts));
end

function [mpcobj] = setupMPC(Ts, p, m)
% MPC Controller Design 
%
% The control objective is to track a step change in the reference signal.
% First, design an MPC controller for the average plant model. The
% controller sample time is Ts second.
    sys = ss(c2d(tf([5 5],[1 3 2 6]),Ts));  % prediction model
    mpcobj = mpc(sys,Ts,p,m);
    
    % Set hard constraints on the manipulated variable and specify tuning
    % weights.
    mpcobj.MV = struct('Min',-2,'Max',2);
    mpcobj.Weights = struct('MV',0,'MVRate',0.01,'Output',1);
end

function [] = setupSolverDir(solverDirPath, SkipSolverGeneration)
% Set up directory for the FORCESPRO solver generation
    if nargin < 2
        SkipSolverGeneration = 0;
    end
    if SkipSolverGeneration
        addpath(solverDirPath)
    else
        if isfolder(solverDirPath)
            try
                rmdir(solverDirPath, 's');
            catch
                clear('mex');
                rmdir(solverDirPath, 's');
            end
        end
        mkdir(solverDirPath)
        addpath(solverDirPath)
    end
end
