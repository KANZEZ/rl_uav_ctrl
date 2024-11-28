function [] = forcesnlmpc_cstr(useMex, useSQP)
%% Design nonlinear MPC Controller
% This example shows how to create and test a nonlinear model predictive 
% controller for an exothermic chemical reactor through the FORCESPRO plugin
% for the MPC Toolbox. It illustrates how to ude the generic MPC Toolbox to
% FORCESPRO interface as well as the generated mex function.
%
% This example requires Model Predictive Control Toolbox from MathWorks.
%
% Copyright 2020-2023 The MathWorks, Inc. and Embotech AG, Zurich, Switzerland
%
% The example can be run without inputs, but to run specific settings one
% can also set the following inputs:
%
%       forcesnlmpc_cstr( useMex )
%
% where useMex is a boolean. If useMex = true, then the generated MEX
% interface during the closed-loop simulation (for optimal performance). If
% useMex = false (default) then nlmpcmoveForces is used to generate optimal
% control inputs.
%
%       forcesnlmpc_cstr( useMex, useSQP )
% 
% where useSQP is a boolean. If useSQP = true (default), then a SQP 
% algorithm is used and if useSQP = false then an interior point algorithm
% is used.
%
% To run this example using the fast SQP solver from FORCESPRO, do as
% follows:
%
%   1.  Run 'forcesMpcCache reset' in the MATLAB terminal to enable 
%       simulation cache.
%   2.  Run this script to generate a general QP solver and collect tuning
%       data by running a simulation (the script must be run with default
%       input values).
%   3.  Run 'forcesMpcCache off' in the MATLAB terminal to save the
%       simulation data.
%   4.  Run this script again with useSQP=true (default). This time a fast 
%       SQP solver will be generated used for the simulation.
    clc;
    close all;

    if nargin < 1
        useMex = false;
    end
    if nargin < 2
        useSQP = true;
    end

    %% Model plant using @nlmpc object
    [nlobj, options] = getSolverGenerationData(useSQP);
    options.Server = 'https://forces.embotech.com';

    %% Generate FORCESPRO NLP solver through MPC Toolbox interface
    [coredata, onlinedata] = nlmpcToForces(nlobj,options);
    onlinedata.md = [10 298.15];

    %% Closed loop simulation
    [time, cost, solvetime, recorded_concentration, reference_concentration] = runSimulation(nlobj, options, coredata, onlinedata, useMex);

    %% Plot results
    plotResults(time, cost, solvetime, recorded_concentration, reference_concentration);

end

function [nlobj, options] = getSolverGenerationData( useSQP )
% This fcn constructs the nlmpc object as well as the options needed to
% generate a FORCESPRO solver.
% The control objective in this example is to maintain the concentration of
% reagent A in the exitstream of the plant. As a consequence the output
% function of our plant model outputs the concentration of A.
    nx = 2;
    ny = 1;
    %nu = 3;
    nlobj = nlmpc(nx,ny,'MV',1,'MD',[2 3]);
    Ts = 0.5;
    nlobj.Ts = Ts;
    nlobj.PredictionHorizon = 6; 
    nlobj.ControlHorizon = [2 2 2];
    nlobj.MV.RateMin = -5;
    nlobj.MV.RateMax = 5;
    % See model functions below
    nlobj.Model.StateFcn = 'exocstrStateFcnCT';
    nlobj.Model.OutputFcn = 'exocstrOutputFcn';

    %% Set options to use with FORCESPRO
    options = nlmpcToForcesOptions();
    options.SolverName = 'CstrSolver';
    options.SolverType = 'InteriorPoint';
    options.IntegrationNodes = 5;
    options.IP_MaxIteration = 500;
    options.x0 = [311.2639; 8.5698];
    options.mv0 = 298.15;

    if useSQP
        options.SolverType = 'SQP';
        options.SQP_MaxQPS = 5;
        options.SQP_MaxIteration = 500; 
    end
end

function [time, cost, solvetime, recorded_concentration, reference_concentration] = runSimulation(nlobj, options, coredata, onlinedata, useMex)
% Run a closed-loop simulation
            
    Tstop = 200; % Number of seconds to simulate
    simulationLength = Tstop/nlobj.Ts; % Number of simulation steps
    x = options.x0; % initial state vector
    mv = options.mv0; % initial manipulated variable
    %md = onlinedata.md; % initial measured disturbance

    % data to store simulation results
    time = zeros(simulationLength,1);
    recorded_concentration = zeros(simulationLength,1);
    reference_concentration = zeros(simulationLength,1);
    cost = zeros(simulationLength,1);
    solvetime = zeros(simulationLength,1);

    for k = 1:simulationLength
        
        % set reference trajectory
        onlinedata.ref = exocstrReferenceTrajectory( Tstop, k);
        
        % call generated FORCESPRO solver through MPC Toolbox interface or
        % call mex for speed-up
        if useMex
            [mv, onlinedata, info] = nlmpcmove_CstrSolver(x,mv,onlinedata); % generated mex function has the name strcat('nlmpcmove_', options.SolverName)
        else
            [mv, onlinedata, info] = nlmpcmoveForces(coredata,x,mv,onlinedata);
        end
        
        % always check that solve was successfull before applying result
        assert(info.ExitFlag == 1, 'FORCESPRO solver failed to find solution');
        
        % simulate dynamics
        x = RK4(x,[mv;onlinedata.md'],@(x,u) exocstrStateFcnCT(x,u),nlobj.Ts);
        
        % store simulation data
        time(k) = k * nlobj.Ts;
        recorded_concentration(k) = x(2);
        reference_concentration(k) = onlinedata.ref(2);
        cost(k) = info.Cost;
        solvetime(k) = info.SolveTime * 1e3; % store solvetime in ms
    end
end

function [ ] = plotResults(time, cost, solvetime, recorded_concentration, reference_concentration)
% Plot the results obtained from the closed-loop simulation
    % Cost
    figure('Name','Cost');clc;
    plot(time, cost, 'b', 'LineWidth', 2); hold on;
    legend('cost');
    xlabel('Simulation time (s)'); ylabel('Cost');grid on;

    % Solve time
    figure('Name','Solvetime');clc;
    plot(time, solvetime, 'b', 'LineWidth', 2); hold on;
    legend('solve time');
    xlabel('Simulation time (s)'); ylabel('Solve time (ms)');grid on;

    % Concentration of A
    figure('Name','Concentration of A vs time');clc;
    plot(time, recorded_concentration, 'b', 'LineWidth', 2); hold on;
    plot(time, reference_concentration, 'r', 'LineWidth', 2); hold on;
    legend('measured concentration','reference concentration');
    xlabel('Simulation time (s)'); ylabel('Concentration of A');grid on;
end

function [ ref ] = exocstrReferenceTrajectory( Tstop, k )
% reference trajectory 
    Ts = 0.5;
    PredictionHorizon = 6;
    yHigh = 8.5698;
    yLow = 2;
    time = (0:Ts:(Tstop+PredictionHorizon*Ts))';
    len = length(time);
    r = [yHigh*ones(5,1);linspace(yHigh,yLow,len-10)';yLow*ones(5,1)];
    ref = r(k:k+PredictionHorizon-1);
end
