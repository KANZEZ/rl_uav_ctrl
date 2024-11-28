function [] = EvChargeAndSpeedProfile2D()
% Example script demonstrating how to use the NLP solver to determine the
% optimal speed and charging profile of an electric vehicle (EV) in order
% to minimize the total trip time. 

% The main difference to the previous example EvChargeAndSpeedProfile() is
% that it demonstrates the usage of 2D-splines (i.e. the previous example
% made a cruder approximation of the motor efficiency using 1D-splines).
% Furthermore, this updated example shows how to scale the optimization
% variables for higher numerical stability.

% The controller plans the car's speed trajectory such that the given route
% is traversed in the shortest time possible, while simultaneously 
% respecting speed limits as well as vehicle's and battery's technical 
% requirements.
%
% We consider a single electric vehicle and a fixed route with given road 
% slopes (i.e. angles) and speed limits. Optionally, some charging stations
% could be available at predefined locations on the road. The user can
% define whether or not the vehicle has to stop at every charging point or 
% the optimizer can decide internally. This is achieved using the 
% `trip.reduceSpeed` option. However, the accuracy and feasibility of the 
% solution are not guaranteed in the latter case, as the formulation does 
% not explicitly cover discrete decisions.

% Due to the spatial characteristic of the problem, the formulation is 
% discretized in space domain. Vehicle dynamics are described discretely 
% using kinetic energy laws and longitudinal velocity and force action. In 
% terms of vehicle energetics, the SoC is regulated through charging and 
% energy dissipation due to driving. 

% The objective is to minimize time as terminal state as well as to reduce
% cost associated with slack variable used to relax the min/max SoC bounds.
% 
% The problem is currently solved as a full-horizon MPC snapshot.
%
% Variables are collected stage-wise into 
% 
%     z = [slack Ft Fb deltaTch v t SoC].
% 
% See also FORCES_NLP.
% 
% This file is part of the FORCESPRO client software for Matlab.
% (c) embotech AG, 2013-2023, Zurich, Switzerland. All rights reserved. 

    clc;
    close all;

    %% Trip parameters (provided by the user)
    % Note: Accuracy and feasibility of the solution are not guaranteed when
    % the `trip.reduceSpeed` flag is disabled due to inherent mixed-integer
    % nature of the problem!

    % Trip type (0 - long (Munich - Cologne); 1 - short)
    trip.type = 1;
    
    switch trip.type 
        case 1    
            % Spatial discretization step (km)
            trip.Ls = 1;
            % Trip distance (km) 
            trip.dist = 50;    
            % EV charging station locations (km)
            trip.kmPosCh = [8, 18, 30, 40, 45];
            % Force vehicle to reduce speed at charging locations (1 - yes; 0 - no)
            trip.reduceSpeed = 0;
            % Initial vehicle speed (km/h)
            trip.initSpeed = 30;
            % Initial SoC [0, 1]
            trip.initSoC = 0.25;
        case 0
            % Spatial discretization step (km)
            trip.Ls = 1;
            % Trip distance (km) 
            trip.dist = 573;    
            % EV charging station locations (km)
            trip.kmPosCh = [110, 150, 250, 375];    
            % Force vehicle to reduce speed at charging locations (1 - yes; 0 - no)
            trip.reduceSpeed = 1;
            % Initial vehicle speed (km/h)
            trip.initSpeed = 30;
            % Initial SoC [0, 1]
            trip.initSoC = 0.75;
    end    
    
    %% Check trip setup
    trip = checkTripSetup(trip);

    %% Define vehicle parameters
    param = defineVehicleParameters(trip);

    %% Formulate the problem and generate the solver
    [model, scalingVec] = generateSpeedProfile(trip, param);    

    %% Simulate the model
    sim = runSimulation(trip, param, model, scalingVec);

    %% Plot simulation results
    plotResults(trip, param, sim);

end

%% Auxilary functions
function [ trip ] = checkTripSetup(trip)
% Checks validity of provided trip setup and modifies it if needed.
    % Trip type has to be 0 or 1
    if trip.type~=0 && trip.type~=1  
        error('Incorrect input: trip type should be either 0 or 1.');
    end 
    % Force vehicle to reduce speed at charging locations when running the
    % long trip scenario
    if trip.type==0
        trip.reduceSpeed = 1;
    end
end

function [ params ] = defineVehicleParameters(trip)
% Returns vehicle parameters
    % Constant vehicle parameters (not to be changed)
    % The data is obtained from "Technical specifications of the BMW i3
    % (120 Ah)" valid from 11/2018 and available at 
    % https://www.press.bmwgroup.com/global/article/detail/T0285608EN/
    
    % Gravitational acceleration (m/s^2)
    params.g = 9.81;
    % Vehicle mass (kg)
    params.mv = 1345;
    % Mass factor (-)
    params.eI = 1.06;
    % Equivalent mass (kg)
    params.meq = (1+params.eI)*params.mv;
    % Projected frontal area (m^2)
    params.Af = 2.38;
    % Air density (kg/m^3)
    params.rhoa = 1.206;
    % Air drag coefficient (-)
    params.ca = 0.29;
    % Rolling resistance coefficient (-)
    params.cr = 0.01;    
    % Maximum traction force (N) 
    params.FtMax = 5e3;    
    % Minimum traction force (N) 
    params.FtMin = 0;
    % param.FtMin = -1.11e3;
    % Maximum breaking force (N)
    params.FbMax = 10e3;
    % Battery's energy capacity (Wh) 
    % BMW i3 60Ah - 18.2kWh; BMW i3 94Ah - 27.2kWh; BMW i3 120Ah - 37.9kWh
    params.Ecap = 37.9e3;
    % Maximum permissible charging time (s)
    params.TchMax = 3600;
    % Charging power (W) 
    % 7.4 kW on-board charger on IEC Combo AC, optional 50 kW Combo DC
    params.PchMax = 50e3;
    % Power dissipation factor (-) (adjusted for Munich - Cologne trip in
    % order to achieve realistic BMW i3 range of ~260km)
    if trip.type == 1
        params.pf = 1;
    else
        params.pf = 0.4;
    end
    % Minimum vehicle velocity (km/h)    
    params.vMin = 30;
    % Maximum vehicle velocity (km/h)
    params.vMax = 150; 
    % Minimum SoC (-)   
    params.SoCmin = 0.1;
    % Maximum SoC (-)    
    params.SoCmax = 0.9;
    
    % Variable vehicle parameters
    % Motor efficiency (-)
    params.eta = setupMotorEfficiency();
    % Charging power (W)
    params.Pch = setupChargingPower(params.PchMax);
    % Traction force hyperbola (upper limit) (N)
    params.FtMaxHyp = setupTractionForceHyperbola(kmh2ms(params.vMax), params.FtMax);
end

function [ etaMotorSpline ] = setupMotorEfficiency()
% Returns a cubic bivariate spline representing motor efficiency as a 
% function of velocity and traction force. 
%
% The data is taken from "Jia, Y.; Jibrin, R.; Goerges, D.: Energy-Optimal
% Adaptive Cruise Control for Electric Vehicles Based on Linear and 
% Nonlinear Model Predictive Control. In: IEEE Transactions on Vehicular
% Technology, vol. 69, no. 12, pp. 14173-14187, Dec. 2020."
    
    vMaxMEff = 180;     % maximum velocity in the lookup table [km/h] 
    FtMaxMEff = 5e3;    % maximum traction force in the lookup table [N]
    vMaxMEff = kmh2ms(vMaxMEff);
    vSampleOrig = 0:vMaxMEff/10:vMaxMEff;
    FtSampleOrig = 0:FtMaxMEff/5:FtMaxMEff;
    
    % denser grid point selection near the edges with the sharp transitions
    vSampleOrig = [vSampleOrig(1), mean(vSampleOrig(1:2)), vSampleOrig(2:end)];
    FtSampleOrig = [FtSampleOrig(1), mean(FtSampleOrig(1:2)), FtSampleOrig(2:end)];

    %     v [m/s] =        0,  2.5,    5,   10,   15,   20,   25,   30,   35,   40,   45,   50
    etaSampleOrig = [   
                        0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50; ... % F [N] = 0
                        0.50, 0.68, 0.80, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.83, 0.81; ... % F [N] = 500
                        0.50, 0.78, 0.81, 0.88, 0.88, 0.88, 0.88, 0.88, 0.85, 0.83, 0.81, 0.80; ... % F [N] = 1000
                        0.50, 0.75, 0.81, 0.85, 0.88, 0.85, 0.83, 0.81, 0.78, 0.68, 0.65, 0.63; ... % F [N] = 2000
                        0.50, 0.68, 0.80, 0.83, 0.83, 0.81, 0.78, 0.68, 0.65, 0.63, 0.60, 0.57; ... % F [N] = 3000
                        0.50, 0.67, 0.78, 0.81, 0.80, 0.78, 0.65, 0.63, 0.60, 0.57, 0.55, 0.55; ... % F [N] = 4000
                        0.50, 0.67, 0.75, 0.78, 0.75, 0.65, 0.63, 0.60, 0.57, 0.55, 0.52, 0.52; ... % F [N] = 5000
                    ];
    
    % spline
    useScipy = false;
    if useScipy && validPythonPackages()
        % Option 1: fitting with Scipy's SmoothBivariateSpline
        % convert to 1-D sequences of data points
        [vTemp, FtTemp] = meshgrid(vSampleOrig, FtSampleOrig);
        vSample = reshape(vTemp.', 1, []);
        FtSample = reshape(FtTemp.', 1, []);
        etaSample = reshape(etaSampleOrig', [1, numel(etaSampleOrig)]);
     
        s = 0.025;
        w = ones(1, length(vSample));
        bbox = [min(vSample), max(vSample), min(FtSample), max(FtSample)];
        kx = 3;
        ky = 3;
        rankTol = 1e-8;
        scaling = [1e1, 1e3, 1];
        etaMotorSpline = ForcesInterpolationFit_SmoothBivariateSpline(vSample, FtSample, etaSample, w, bbox, kx, ky, s, rankTol, scaling);
    else
        % Option 2: providing 2D-spline coefficients
        kx = 3;
        ky = 3;
        tx = [0, 0, 0, 0, 6.3993742001266, 24.1805094335686, 50, 50, 50, 50];
        ty = [0, 0, 0, 0, 760.320551795358, 1503.23831410745, 5000, 5000, 5000, 5000];
        coeffs = [0.498727471092637   0.511037494098402   0.524901875945660   0.548511907792580   0.475607487652389   0.513135686437586
                  0.498465143841756   0.654046675851965   0.783006359203673   0.713858506960411   0.676705972846789   0.681154627267599
                  0.510442836827170   0.854158083414314   1.007125597517271   0.847628255554849   1.018658592375758   0.878826758082995
                  0.495093430686428   0.735511289747710   0.857756122056489   0.863078750613390   0.548595365620131   0.497927393614425
                  0.510240152112442   0.835399001169469   0.952683895958243   0.536982511482952   0.563982968586042   0.577416237725016
                  0.501474795696226   0.773879473939183   0.878143062979889   0.444534437467682   0.615960539904494   0.508404545245928];
        etaMotorSpline = ForcesInterpolation2D(tx, ty, coeffs, kx, ky);
    end

    % plotting
    vMesh = 0:1:vMaxMEff;
    FtMesh = 0:100:FtMaxMEff;
    [vMesh, FtMesh] = meshgrid(vMesh, FtMesh);
    
    etaMesh = full(etaMotorSpline(vMesh(:), FtMesh(:)));
    etaMesh = reshape(etaMesh, size(vMesh, 1), size(vMesh, 2));
    
    figure;
    surf(vMesh, FtMesh, etaMesh);
    xlabel("v [m/s]")
    ylabel("Ft [N]")
    title("Motor Efficiency")

    figure;
    contourf(vMesh, FtMesh, etaMesh, unique(etaSampleOrig)', 'ShowText',true)
    xlabel("v [m/s]")
    ylabel("Ft [N]")
    title("Motor Efficiency")
end

function [ packagesInstalled ] = validPythonPackages()
% Checks if Python3 loads with the SciPy package for 2D spline fitting
    try
        py.importlib.import_module('scipy');
        packagesInstalled = true;
    catch
        warning(['No valid Python3 and/or SciPy package installation in the system.', newline, 'Option 2 will be used for the motor efficiency 2D spline.']);
        packagesInstalled = false;
    end
end

function [ Pch ] = setupChargingPower(PchMax)
% Returns piecewise linear representation of charging power as a 
% function of vehicle's SoC.
%
% The data is taken from Fastned charging chart available at 
% https://support.fastned.nl/hc/en-gb/articles/204784718-Charging-with-a-BMW-i3
%
% NOTE: Python example uses shape-preserving piecewise cubic spline
% approximation ('pchip') which might lead to negligible result 
% discrepancies between the two clients when directly compared.
    socSample = [0.15, 0.85, 1.0];
    PchSample = [0.88, 1, 0.2]*PchMax;    
    Pch = ForcesInterpolationFit(socSample,PchSample,'linear');    
end

function [ FtMaxHypSpline ] = setupTractionForceHyperbola(vMax,FtMax)
% Returns cubic spline representing maximum traction force as a function of
% vehicle's speed    
%
% The data is generated based on the technical article provided at 
% https://x-engineer.org/need-gears/
    vSample = [0.25, 0.4, 0.6, 0.8, 1.0]*vMax;
    FtMaxHypSample = [1.0, 0.67, 0.43, 0.32, 0.28]*FtMax;     
    FtMaxHypSpline = ForcesInterpolationFit(vSample,FtMaxHypSample);    
end

function [model, scalingVec] = generateSpeedProfile(trip, param)
% Formulates the optimization problem and generates a solver for the optimal 
% speed & charging profile by calling the FORCESPRO code generation
%    
% Assume variable ordering zi = [u{i}; x{i}] for i=1...N
% zi = [slack(i); Ft(i); Fb(i); deltaTch(i); v(i); t(i); SoC(i)]
% pi = [vMax(i); vMin(i); alpha(i); TchMax(i); Ls(k)] #k - iteration No.

    %% Scaling Vector
    % Scaling factors approximately set equal to order of magnitude of 
    % the corresponding physical states and inputs
    scalingVec = [1, 1e4, 1e4, 1e4, 1e2, 1e4, 1].';  

    %% Problem dimensions
    nx = 3;    
    nu = 4;
    np = 5;
    nh = 6;
    
    model.N = round(trip.dist/trip.Ls); % horizon length
    model.nvar = nx + nu;               % number of variables
    model.neq  = nx;                    % number of equality constraints
    model.nh   = nh;                    % number of inequality constraints
    model.npar = np;                    % number of runtime parameters

    %% Objective function
    costScaling = 1e3;
    model.objective = @(z) objective(z, scalingVec, costScaling);
    model.objectiveN = @(z) objectiveN(z, scalingVec, costScaling);

    %% Dynamics, i.e. equality constraints 
    model.eq = @(z,p) dynamics(z, p, param, scalingVec, nu, nx);

    model.E = [zeros(nx,nu),eye(nx)];

    %% Inequality constraints
    % Upper/lower variable bounds lb <= z <= ub
    %                             inputs                       |                states
    %           slack       Ft       Fb            Tch                     v           t      SoC
    model.lb = [0,     param.FtMin,  0.,           0.,            kmh2ms(param.vMin),  0.,    0.]./scalingVec.'; 
    model.ub = [0.1,   param.FtMax,  param.FbMax,  param.TchMax,  kmh2ms(param.vMax),  +inf,  1.]./scalingVec.';

    % Nonlinear inequalities hl <= h(z,p) <= hu
    model.ineq = @(z,p) ineq(z, p, param, scalingVec);

    % Upper/lower bounds for inequalities                 
    model.hu = [0,    0,    0,    0,    0,    0];
    model.hl = [-inf, -inf, -inf, -inf, -inf, -inf];    

    %% Initial and final conditions
    model.xinitidx = nu+1:nu+nx;  

    %% Generate FORCESPRO solver
    % Define solver options
    codeoptions = getOptions('FORCESNLPsolver');
    codeoptions.printlevel = 0;
    codeoptions.nlp.compact_code = 1;
    codeoptions.maxit = 300;

    % 2D-splines require the MX CasADi variables
    codeoptions.nlp.ad_expression_class = 'MX';

    % Generate code
    FORCES_NLP(model, codeoptions);
end

function [ xNext ] = dynamics(z, p, param, scalingVec, nu, nx)
    z = z.*scalingVec;

    xNext = [sqrt(ForcesMax(2*p(5)/param.meq*(z(2) - z(3) - param.cr*param.mv*param.g*cos(p(3)) - param.mv*param.g*sin(p(3)) - 0.5*param.ca*param.Af*param.rhoa*z(5)^2) + z(5)^2, 1e-5));                    
             z(6) + p(5)/z(5) + z(4); 
             z(7) - param.pf*p(5)*z(2)/(3600*param.eta(z(5), z(2))*param.Ecap) + param.Pch(z(7))*z(4)/(3600*param.Ecap)];
    
    xNext = xNext./scalingVec(nu+1:nu+nx);
end

function [ h ] = ineq(z, p, param, scalingVec)
    z = z.*scalingVec;

    h = [z(2) - param.FtMaxHyp(z(5));
         z(4) - p(4);
         z(7) - param.SoCmax - z(1);
         param.SoCmin - z(7) - z(1);
         2*p(5)/param.meq*(z(2) - z(3) - param.cr*param.mv*param.g*cos(p(3)) - param.mv*param.g*sin(p(3)) - 0.5*param.ca*param.Af*param.rhoa*z(5)^2) + z(5)^2 - kmh2ms(ForcesMax((1-z(4))*p(1), param.vMin+1))^2;
         kmh2ms(p(2))^2 - (2*p(5)/param.meq*(z(2) - z(3) - param.cr*param.mv*param.g*cos(p(3)) - param.mv*param.g*sin(p(3)) - 0.5*param.ca*param.Af*param.rhoa*z(5)^2) + z(5)^2)];
    
    h = h./[scalingVec(2); scalingVec(4); scalingVec(7); scalingVec(7); scalingVec(5); scalingVec(5)];
end

function [ stageCost ] = objective(z, scalingVec, costScaling)
    z = z.*scalingVec;

    R = [ 1e-7,  0; 
          0,    1e-6];
    slackCostFactor = 1e6;
    stageCost = slackCostFactor*z(1) + [z(2);z(3)]'*R*[z(2);z(3)];

    stageCost = stageCost/costScaling;
end

function [ terminalCost ] = objectiveN(z, scalingVec, costScaling)
    z = z.*scalingVec;

    terminalCost = z(6);

    terminalCost = terminalCost/costScaling;
end

function sim = runSimulation(trip, param, model, scalingVec)
% Defines initial and stage-dependent runtime parameters and solves the
% problem
    %% Define spatially distributed (i.e. stage-dependent) parameters 
    % Road speed limits and slope angles
    [vMaxRoad, ~, vMinRoad, roadSlope] = setupRoadParameters(param.vMin, trip);

    % Maximum permissible charging time
    deltaTchMax = setupChargingTime(param.TchMax, trip);  
    
    %% Initialize the problem        
    problem.x0 = zeros(model.N*model.nvar,1); 
    sim.kMax = trip.dist/trip.Ls;    
    nx = model.neq;
    nu = model.nvar - model.neq;
    np = model.npar;
    
    if trip.initSpeed < param.vMin || trip.initSpeed > param.vMax
        v1 = kmh2ms(param.vMin);
        warning('Initial vehicle speed is outside of the speed limits. It will be set to the minimum speed limit at t=0.')
    else
        v1 = kmh2ms(trip.initSpeed);
    end

    if trip.initSoC < param.SoCmin || trip.initSoC > param.SoCmax
        SoC1 = 0.5;
        warning('Initial vehicle state-of-charge is outside of the limits. It will be set to 50% at t=0.')
    else
        SoC1 = trip.initSoC;
    end
        
    % X(1) = [v(1); t(1); SoC(1)]
    problem.xinit = [v1; 0; SoC1]./scalingVec(nu+1:nu+nx); 
   
    %% Solve the problem
    
    % Set runtime parameters
    problem.all_parameters = zeros(np*model.N,1);
    for i = 1:model.N
        problem.all_parameters((i-1)*np+1:i*np) = [vMaxRoad(i); vMinRoad(i); roadSlope(i*trip.Ls*1e3); deltaTchMax(i); trip.Ls*1e3];  
    end    

    % Call solver
    [solverout,exitflag,info] = FORCESNLPsolver(problem); 
    sim.exitflag = exitflag;        

    % Extract state and control vector
    if exitflag == 1               
        sim.Z = unpackStruct(solverout,model.nvar).*scalingVec.'; 
        sim.solvetime = info.solvetime;
        sim.iters = info.it;   
    else
        error(['Solver terminated with exitflag ', num2str(exitflag)]);
    end    
    
    sim = displayResults(sim);
end

function [upperSpeedLimit, upperSpeedLimitStrict, lowerSpeedLimit, alpha] = setupRoadParameters(vMin, trip)
% Returns vectors of upper and lower speed limits and road slope as a
% function of spatial position
    % Define maximum speed limit
    switch trip.type
        case 1
            % Arbitrarily generated short trip
            vLim = [50.0, 80.0, 100.0, 120.0, 80.0];
            L = [0.1; 0.1; 0.2; 0.4; 0.2]*trip.dist/trip.Ls;    
            upperSpeedLimitStrict = [vLim(1)*ones(L(1),1);  vLim(2)*ones(L(2),1); vLim(3)*ones(L(3),1); vLim(4)*ones(L(4),1); vLim(5)*ones(L(5),1)];
        case 0
            % Munich - Cologne trip with simplified speed limits
            vLim = [50; 100; 130; 150; 130; 100; 130; 150; 130; 150; 130; 100; 130; 100; 130; 100; 130; 50];
            L = [0; 7; 15; 45; 105; 145; 157; 217; 257; 307; 377; 397; 408; 478; 489; 543; 552; 565; 573];
            upperSpeedLimitStrict = zeros(round(L(end)/trip.Ls),1);
            for i=1:length(vLim)        
                upperSpeedLimitStrict(L(i)/trip.Ls+1:L(i+1)/trip.Ls) = vLim(i);                 
            end       
    end
    upperSpeedLimit = upperSpeedLimitStrict;
    
    % Define minimum speed limit
    lowerSpeedLimit = vMin*ones(size(upperSpeedLimitStrict));    
        
    % Define road slope    
    switch trip.type
        case 1
            sSample = (0:trip.dist/20:trip.dist)*1e3;
            alphaSample = [0.05, 0.08, 0.14, 0.14, 0.2, 0.1, 0.025, 0.015, 0.05, -0.02, -0.08, -0.035, -0.015, 0.0, 0.05, 0.025, 0, 0.05, 0.08, 0.1, 0.15];  
            alpha = ForcesInterpolationFit(sSample,alphaSample);
        case 0
            sSample = (0:trip.dist/10:trip.dist)*1e3;
            alphaSample = [0.04, 0.03, 0.05, 0.02, 0.04, 0.02, 0.03, 0.04, 0.03, 0.05, 0.03];  
            alpha = ForcesInterpolationFit(sSample,alphaSample);
    end
    
    % (Optional) Reduce maximum speed limit at charging locations
    if trip.reduceSpeed
        chInd = trip.kmPosCh/trip.Ls;
        upperSpeedLimit(chInd) = vMin + 0.1;
    end
end

function [ deltaTchMax ] = setupChargingTime(TchMax, trip)
% Returns maximum permissible charging time as a 
% function of spatial position
    chInd = trip.kmPosCh/trip.Ls;    
    deltaTchMax = 1e-2*ones(trip.dist/trip.Ls,1);    
    deltaTchMax(chInd) = TchMax;  
end

function [ sim ] = displayResults(sim)
% Computes and displays relevant simulation metrics
    sim.triptime = sim.Z(end,6);
    % Only consider meaningful charging stops (charging time > 5sec)
    chInd = sim.Z(:,4) > 5;
    sim.chargetime = sum(sim.Z(chInd,4));
    sim.numstops = sum(chInd);
    disp(['Total trip time: ',num2str(round(sim.triptime/60,2)),' min']);        
    disp(['Total charging time: ',num2str(round(sim.chargetime/60,2)),' min']);
    disp(['Number of charging stops: ',num2str(sim.numstops)]);
end

function [] = plotResults(trip, param, sim)
% Plot simulation results
    ind = 1:sim.kMax;
    xkm = trip.Ls*ind;     
    xm = 1:100:trip.dist*1e3;

    % colors
    blue = [0, 0.4470, 0.7410];
    yellow = [0.9290, 0.6940, 0.1250];
    brown = [0.6350, 0.0780, 0.1840];
    
    % Road speed limits and slope angles
    [~, vMaxRoadStrict, vMinRoad, roadSlope] = setupRoadParameters(param.vMin, trip);

    screensize = get(0,'screensize');
    screenWidth = screensize(3);
    screenHeight = screensize(4);
    f = figure; f.Position = [0.1*screenWidth 0.1*screenHeight 0.8*screenWidth 0.8*screenHeight]; clf;

    % Speed profile
    subplot(4,1,1); grid on; title('Speed profile'); hold on;
    stairs(xkm,vMaxRoadStrict(ind),'-.','color',brown); stairs(xkm,vMinRoad(ind),'-.','color',brown); 
    stairs(xkm,ms2kmh(sim.Z(:,5)), 'color',blue);
    xlim([trip.Ls trip.dist]);
    ylim([0 param.vMax]);
    xlabel('Distance [km]');
    ylabel('Vehicle speed [km/h]');
    legend('Speed limits','Location','southeast');

    % Vehicle forces
    subplot(4,1,2);  grid on; title('Vehicle Forces'); hold on;   
    y1 = stairs(xkm,min(param.FtMax,param.FtMaxHyp(sim.Z(:,5)))*1e-3,'-.','color',brown);
    y2 = stairs(xkm,sim.Z(:,2)*1e-3,'color',blue);        
    y3 = stairs(xkm,sim.Z(:,3)*1e-3,'color',yellow);         
    xlim([trip.Ls trip.dist]);
    ylim([param.FtMin param.FtMax]*1e-3);
    xlabel('Distance [km]');
    ylabel('Force [kN]');
    legend([y2,y3,y1],'Traction force','Breaking force','Traction force limits');
    
    % Battery profile
    subplot(4,1,3); grid on; title('Battery profile'); hold on;
    yyaxis left
    y1 = plot(xkm,100*param.SoCmin*ones(size(ind)),'-.','color',brown); 
    plot(xkm,100*param.SoCmax*ones(size(ind)),'-.','color',brown); 
    stairs(xkm,100*sim.Z(:,7)','color',blue);
    xlim([trip.Ls trip.dist]);
    ylim([0 100]);
    xlabel('Distance [km]');
    ylabel('State of charge [%]'); 
    yyaxis right
    stairs(xkm,sim.Z(:,4)/60);
    ylim([0 max(40, max(sim.Z(:,4)/60))]);
    ylabel('Charging time [min]')
    legend(y1,'SoC limits')

    % Road profile
    subplot(4,1,4);  grid on; title('Road profile'); hold on;   
    yyaxis left
    stairs(xkm,vMaxRoadStrict(ind)); stairs(xkm,vMinRoad(ind),'-','color','#0072BD'); 
    xlim([trip.Ls trip.dist]);
    ylim([0 param.vMax]);
    xlabel('Distance [km]')
    ylabel('Speed limits [km/h]')
    yyaxis right
    stairs(xm*1e-3,rad2deg(roadSlope(xm)));
    stairs(xm*1e-3,zeros(size(xm)),'-.','color','#D95319');       
    ylim([-20 20]);    
    ylabel('Road slope [deg]')
end

function [ unpackedVector ] = unpackStruct(structure,numVar)
% Unpacks a structure into a vector
    fn = fieldnames(structure);
    unpackedVector = zeros(numel(fn),numVar);
        
    for i = 1:numel(fn)
        fni = string(fn(i));
        field = structure.(fni);
        if (isstruct(field))
            unpackStruct(field);
            continue;
        end
        unpackedVector(i,:) = field;
    end
end

function [ ms ] = kmh2ms(kmh)
% Converts km/h into m/s
    ms = kmh/3.6;
end

function [ kmh ] = ms2kmh(ms) 
% Converts m/s into km/h
    kmh = ms*3.6;
end

function [ deg ] = rad2deg(rad) 
% Converts radians into degrees
    deg = rad*180/pi;
end
