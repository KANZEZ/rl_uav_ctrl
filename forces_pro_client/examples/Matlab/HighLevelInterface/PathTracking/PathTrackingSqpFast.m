function [] = PathTrackingSqpFast()
% Example script for getting started with FORCESPRO NLP solver.
% 
% This example solves an optimization problem with the system dynamics given
% by the following continuous-time, nonlinear kinematic bicycle model:
% 
%    dxPos/dt = v*cos(theta + beta)
%    dyPos/dt = v*sin(theta + beta)
%    dv/dt = F/m
%    dtheta/dt = v/l_r*sin(beta)
%    ddelta/dt = phi
% 
%    with:
%    beta = arctan(l_r/(l_r + l_f)*tan(delta))
% 
% where xPos,yPos are the position, v the velocity in the direction of
% heading angle theta and delta the steering angle of the vehicle. The
% inputs are acceleration force F and steering rate phi. The physical
% constants m, l_r and l_f denote the car's mass and the distance from
% the car's center of gravity to the rear wheels and the front wheels.
% 
% The car starts from standstill and the optimization problem is to
% minimize the distance of the car's position to given waypoints on
% each stage of the prediction horizon.
% 
% Quadratic costs for the propulsion force and steering rate are added to
% the objective to avoid unnecessary excessive maneuvers.
% 
% There are bounds on all variables except heading angle.
% 
% Variables are collected stage-wise into 
% 
%     z = [F phi xPos yPos v theta delta].
% 
% This example models the task as a MPC problem using the SQP method.
% 
% See also FORCES_NLP
% 
% (c) Embotech AG, Zurich, Switzerland, 2013-2023.
%
%
% IMPORTANT NOTE:
%
% The racetrack map contained in track2_IfA.mat (renamed from track2.mat)
% originates from the repository  https://github.com/alexliniger/MPCC  and 
% the following license notice applies to it:
%   Copyright 2019 Alexander Liniger
%   
%   Licensed under the Apache License, Version 2.0 (the "License");
%   you may not use this file except in compliance with the License.
%   You may obtain a copy of the License at
%   
%      http://www.apache.org/licenses/LICENSE-2.0
%   
%   Unless required by applicable law or agreed to in writing, software
%   distributed under the License is distributed on an "AS IS" BASIS,
%   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
%   See the License for the specific language governing permissions and
%   limitations under the License.
%

    clc;
    close all;

    filePath = fileparts(mfilename('fullpath'));
    cd(filePath);
    addpath(filePath);


    %% Generate FORCESPRO solver
    % Settings
    timeStep = 0.1;     % Time period of one MPC discretization step. The simulation uses the same time step [s]
    horizonLength = 10; % MPC prediction horizon length [s]

    % Create a new folder for all solver related files
    solverDir = [filePath, filesep, 'codeGen'];
    if ~exist(solverDir,'dir')
        mkdir(solverDir)
    end
    addpath(solverDir);

    %% Generate and run a general SQP solver
    cd(solverDir);
    % use double precision to get most accurate cached solutions
    [model, codeoptions, I] = generatePathTrackingSolver(solverDir, timeStep, horizonLength, false, 'double');
    FORCES_NLP(model, codeoptions);
    cd(filePath);
    [ cache, sqpGeneralSolvetimes, sqpGeneralFevalstimes, sqpGeneralQptimes ] = simulate(model, I, timeStep, horizonLength, false);

    %% Generate and run a fast SQP solver
    % use single precision to get the most light-weighted solver
    [model, codeoptions, I] = generatePathTrackingSolver(solverDir, timeStep, horizonLength, true, 'float');
    
    % setup tuning options for SQP Fast
    tuningoptions = ForcesAutotuneOptions( cache );
    tuningoptions.setTuningGoal("speed");
    tuningoptions.setTuningSeed(123);
    tuningoptions.setTuningIterations(10000);
    tuningoptions.ValidationControlOutputIdx = I.inputs;    
    
    % generate a tuned SQP Fast solver
    ForcesGenerateSqpFastSolver(model, codeoptions, tuningoptions);
    
    % and run it
    cd(filePath);
    [ ~, sqpFastSolvetimes, sqpFastFevalstimes, sqpFastQptimes ] = simulate(model, I, timeStep, horizonLength, true);

    plotResults(sqpGeneralSolvetimes, sqpGeneralFevalstimes, sqpGeneralQptimes, sqpFastSolvetimes, sqpFastFevalstimes, sqpFastQptimes);

end

function [ cache, solvetimes, fevalstimes, qptimes ] = simulate(model, I, timeStep, horizonLength, useWarmstart)

    %% Simulate and call solver
    % Settings
    simLength = 360;        % Number of simulation steps at which the simulation stops
    accelLatMax = 3;        % Maximum lateral acceleration for speed profile calculation [m/s^2]
    mapName = 'racetrack';	% Current options: 'ellipse' and 'racetrack'

    % Create 2D points which the car is supposed to follow
    if strcmp(mapName,'ellipse')
        map = getEllipsePath();
    elseif strcmp(mapName,'racetrack')
        map = getIfARacetrack();
        disp(' ');
        disp('IMPORTANT NOTE:');
        disp('This simulation uses racetrack data by Alexander Liniger, licensed under the Apache License (Version 2.0).')
        disp('Please see the header of PathTracking.m for more details.');
        disp(' ');
    else
        error('Specified map type is not available!');
    end
    map = planVelocity(map, accelLatMax, model.ub(I.velocity));

    % Matrix for storing simulation data
    simulatedZ = zeros(model.nvar,simLength+1);
    simulatedZ(I.states,1) = [map.x(1), map.y(1), 0., map.heading(1), 0.]'; % Initial vehicle state

    % Set initial guess to start solver from (constant max. FLon, zero steeringAngle)
    predictedZ = zeros(model.nvar, model.N);
    predictedZ(:,1) = simulatedZ(:,1);
    predictedZ(I.FLon,1) = model.ub(I.FLon); % Set Flon to its max. value
    for k = 1:model.N-1
        predictedZ(:,k+1) = [predictedZ(I.inputs,k); model.eq( predictedZ(:,k) )];
    end
    x0 = reshape(predictedZ,[],1);

    % Initialize plot handles
    handles = [];
    cache = cell(simLength,1);
    solvetimes = zeros(simLength, 1);
    fevalstimes = zeros(simLength, 1);
    qptimes = zeros(simLength, 1);

    if useWarmstart
        problem.reinitialize = 0;
    else
        problem.reinitialize = 1;
    end

    for k = 1:simLength

        % set initial guess
        problem.x0 = x0;

        % Set initial condition
        problem.xinit = simulatedZ(I.states,k);

        % Set runtime parameters (here, the next N points on the map)
        nextPathPoints = resamplePathForTracker(simulatedZ(:,k), I, map, timeStep, horizonLength);
        problem.all_parameters = reshape(nextPathPoints,2*model.N,1);

        % Solve optimization problem
        [output,exitflag,info] = PathTrackSolver(problem);
        cache{k} = problem;
        solvetimes(k) = info.solvetime;
        fevalstimes(k) = info.fevalstime;
        qptimes(k) = info.QPtime;

        % construct initial guess for next solver call from solution
        x0 = [ output.x01; output.x02; output.x03; output.x04; output.x05; output.x06; output.x07; output.x08; output.x09; output.x10 ];
        
        % Make sure the solver has exited properly
        if (exitflag == 1)
            fprintf('FORCESPRO took %d iterations and ',info.it); 
            fprintf('%.3f milliseconds to solve the problem.\n\n',info.solvetime*1000);
        elseif (exitflag == -8)
            warning('FORCESPRO finished with exitflag=-8. The underlying QP might be infeasible.');
            fprintf('FORCESPRO took %d iterations and ',info.it); 
            fprintf('%.3f milliseconds to solve the problem.\n\n',info.solvetime*1000);
        else
            error('Some problem in solver!');
        end

        % Apply optimized input u to system and save simulation data
        simulatedZ(I.inputs,k) = output.x01(I.inputs);
        simulatedZ(I.states,k+1) = model.eq( simulatedZ(:,k) );

        % Extract output for prediction plots
        cwidth = floor(log10(model.N))+1;
        for i = 1:model.N  
            predictedZ(:,i) = output.(sprintf(sprintf('x%%0%iu',cwidth), i));
        end
        % Plot
        handles = plotPathTrackerData(k,simulatedZ,predictedZ,model,I,simLength,map,nextPathPoints,handles);

        % Pause to slow down plotting
        pause(0.02);

    end

    close(gcf);

end

%% auxiliary functions
function [ handles ] = plotPathTrackerData(k,simulatedZ,predictedZ,model,I,simLength,map,nextPathPoints,handles)
    
    narginchk(8,9)
    if nargin<9
        handles = [];
    end
    
    % Determine whether plots need to be created or only updated
    createPlots = true;
    updatePlots = false;
    
    if all(isfield(handles,{'figHandle','xyPlotHandles','velocityPlotHandles','steeringPlotHandles',...
            'accelPlotHandles','steeringRatePlotHandles'}))
        createPlots = false;
        updatePlots = true;
        figHandle = handles.figHandle;
        xyPlotHandles = handles.xyPlotHandles;
        velocityPlotHandles = handles.velocityPlotHandles;
        steeringPlotHandles = handles.steeringPlotHandles;
        accelPlotHandles = handles.accelPlotHandles;
        steeringRatePlotHandles = handles.steeringRatePlotHandles;
    end
    
    if createPlots
        figHandle = figure('units','normalized','outerposition',[0 0 1 1]); 
        clf;
        handles.figHandle = figHandle;
    elseif updatePlots
        try
            figure(figHandle);
        catch
            error('Simulation stopped as figure has been closed.');
        end
    end
    
    % xy-plot
    if createPlots
        subplot(4,2,[1,3,5,7]); hold all;
        xyPlotHandles(1) = plot(map.x, map.y,'kx');
        xyPlotHandles(2) = plot(nextPathPoints(1,:), nextPathPoints(2,:),'r');
        xyPlotHandles(3) = plot(simulatedZ(I.xPos,1),simulatedZ(I.yPos,1),'bx','LineWidth',3); 
        xyPlotHandles(4) = plot(simulatedZ(I.xPos,1),simulatedZ(I.yPos,1),'b-','LineWidth',2);
        xyPlotHandles(5) = plot(predictedZ(I.xPos,:),predictedZ(I.yPos,:),'-g','LineWidth',2);
        legend({'waypoints', 'desired path', 'initial position', 'past trajectory', 'predicted trajectory'}, ...
            'Location','southeast');
        title('xy-Plot'); 
        xlabel('x-coordinate'); ylabel('y-coordinate');
        xlim([min(map.x)-1, max(map.x)+1]); ylim([min(map.y)-1, max(map.y)+1]);
        daspect([1,1,1]); grid on;
    elseif updatePlots
        xyPlotHandles(2).XData = nextPathPoints(1,:); xyPlotHandles(2).YData = nextPathPoints(2,:);
        xyPlotHandles(4).XData = simulatedZ(I.xPos,1:k); xyPlotHandles(4).YData = simulatedZ(I.yPos,1:k); 
        xyPlotHandles(5).XData = predictedZ(I.xPos,:); xyPlotHandles(5).YData = predictedZ(I.yPos,:);
    end
    
    % Velocity plot
    if createPlots
        subplot(4,2,2); hold all;
        title('Velocity'); grid on;
        velocityPlotHandles(1) = plot(0.0,simulatedZ(I.velocity,1),'b');
        velocityPlotHandles(2) = plot(1:model.N, predictedZ(I.velocity,:),'g-');
        velocityPlotHandles(3) = plot([1 simLength], [model.ub(I.velocity) model.ub(I.velocity)]', 'r:'); 
        velocityPlotHandles(4) = plot([1 simLength], [model.lb(I.velocity) model.lb(I.velocity)]', 'r:');
        xlim([1,simLength]);
        ylabel('m/s');
    elseif updatePlots
        velocityPlotHandles(1).XData = 1:k; velocityPlotHandles(1).YData = simulatedZ(I.velocity,1:k);
        velocityPlotHandles(2).XData = k:k+model.N-1; velocityPlotHandles(2).YData = predictedZ(I.velocity,:);
    end
    
    % Steering angle plot
    if createPlots
        subplot(4,2,4); hold all; 
        title('Steering angle'); grid on;
        steeringPlotHandles(1) = plot(0.0,rad2deg(simulatedZ(I.steeringAngle,1)),'b');
        steeringPlotHandles(2) = plot(1:model.N, rad2deg(predictedZ(I.steeringAngle,:)),'g-');
        steeringPlotHandles(3) = plot([1 simLength], rad2deg([model.ub(I.steeringAngle) model.ub(I.steeringAngle)])', 'r:');
        steeringPlotHandles(4) = plot([1 simLength], rad2deg([model.lb(I.steeringAngle) model.lb(I.steeringAngle)])', 'r:');
        xlim([1,simLength]);
        ylabel('deg');
    elseif updatePlots
        steeringPlotHandles(1).XData = 1:k; steeringPlotHandles(1).YData = rad2deg(simulatedZ(I.steeringAngle,1:k));
        steeringPlotHandles(2).XData = k:k+model.N-1; steeringPlotHandles(2).YData = rad2deg(predictedZ(I.steeringAngle,:));
    end
    
    % Plot acceleration force
    if createPlots
        subplot(4,2,6); hold all;
        title('Acceleration force'); grid on;
        accelPlotHandles(1) = stairs(0.0,simulatedZ(I.FLon,1),'b');
        accelPlotHandles(2) = stairs(1:model.N, predictedZ(I.FLon,:),'g-');
        accelPlotHandles(3) = plot([1 simLength], [model.ub(I.FLon) model.ub(I.FLon)]', 'r:');
        accelPlotHandles(4) = plot([1 simLength], [model.lb(I.FLon) model.lb(I.FLon)]', 'r:');
        xlim([1,simLength]);
        ylabel('N');
    elseif updatePlots
        accelPlotHandles(1).XData = 1:k-1; accelPlotHandles(1).YData = simulatedZ(I.FLon,1:k-1);
        accelPlotHandles(2).XData = k:k+model.N-1; accelPlotHandles(2).YData = predictedZ(I.FLon,:);
    end
    
    % Plot steering rate
    if createPlots
        subplot(4,2,8); hold all;
        title('Steering rate'); grid on;
        steeringRatePlotHandles(1) = stairs(0.0,simulatedZ(I.steeringRate,1),'b');
        steeringRatePlotHandles(2) = stairs(1:model.N, predictedZ(I.steeringRate,:),'g-');
        steeringRatePlotHandles(3) = plot([1 simLength], rad2deg([model.ub(I.steeringRate) model.ub(I.steeringRate)])', 'r:');
        steeringRatePlotHandles(4) = plot([1 simLength], rad2deg([model.lb(I.steeringRate) model.lb(I.steeringRate)])', 'r:');
        xlim([1,simLength]);
        ylabel('deg/s');
    elseif updatePlots
        steeringRatePlotHandles(1).XData = 1:k-1; steeringRatePlotHandles(1).YData = rad2deg(simulatedZ(I.steeringRate,1:k-1));
        steeringRatePlotHandles(2).XData = k:k+model.N-1; steeringRatePlotHandles(2).YData = rad2deg(predictedZ(I.steeringRate,:));
    end
    
    % Collect all handles
    handles.xyPlotHandles = xyPlotHandles;
    handles.velocityPlotHandles = velocityPlotHandles;
    handles.steeringPlotHandles = steeringPlotHandles;
    handles.accelPlotHandles = accelPlotHandles;
    handles.steeringRatePlotHandles = steeringRatePlotHandles;
    
end

function [ map ] = getEllipsePath()
    % Creates a map struct with fields x, y, arclength, curvature
    
    numPoints = 200;
    dAngle = 2 * pi / numPoints;
    angleWaypoints = 0:dAngle:numPoints*dAngle;
    xWaypoints = 3.0*cos(angleWaypoints);
    yWaypoints = 5.0*sin(angleWaypoints);
    dArclength = hypot(diff(xWaypoints), diff(yWaypoints));
    arclengthWaypoints = [0,cumsum(dArclength)];
    headingWaypoints = atan2(gradient(yWaypoints), gradient(xWaypoints));
    curvatureWaypoints = wrapToPi(diff(headingWaypoints))./dArclength;
    
     
    map.x = xWaypoints;
    map.y = yWaypoints;
    map.arclength = arclengthWaypoints;
    map.heading = headingWaypoints;
    map.curvature = [curvatureWaypoints, curvatureWaypoints(1)];
end

function [ map ] = getIfARacetrack()
    % The racetrack map contained in track2_IfA.mat (renamed from track2.mat)
    % originates from the repository  https://github.com/alexliniger/MPCC  and 
    % repository  https://github.com/alexliniger/MPCC  and the following 
    % license notice applies to it:
    %
    %   Copyright 2019 Alexander Liniger
    %   
    %   Licensed under the Apache License, Version 2.0 (the "License");
    %   you may not use this file except in compliance with the License.
    %   You may obtain a copy of the License at
    %   
    %      http://www.apache.org/licenses/LICENSE-2.0
    %   
    %   Unless required by applicable law or agreed to in writing, software
    %   distributed under the License is distributed on an "AS IS" BASIS,
    %   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    %   See the License for the specific language governing permissions and
    %   limitations under the License.
    %

    data = load( fullfile('data','track2_IfA.mat') );
    scalingFactor = 7;
    xWaypoints = scalingFactor * data.track2.center(1,:);
    yWaypoints = scalingFactor * data.track2.center(2,:);

    % Calculate arclength, curvature and heading
    dArclength = hypot(diff(xWaypoints), diff(yWaypoints));
    arclengthWaypoints = [0,cumsum(dArclength)];
    headingWaypoints = atan2(gradient(yWaypoints), gradient(xWaypoints));
    curvatureWaypoints = wrapToPi(diff(headingWaypoints))./dArclength;

    map.x = xWaypoints;
    map.y = yWaypoints;
    map.arclength = arclengthWaypoints;
    map.heading = headingWaypoints;
    map.curvature = [curvatureWaypoints, curvatureWaypoints(1)];
    
end

function [ map ] = planVelocity(map, accelLatMax, velMax)
    % Calculates reference velocity and time for a given map, max. lateral acceleration and max. velocity
    
    % Calculate max. velocity from accelLatMax
    velFromLatAccel = sqrt(accelLatMax./abs(map.curvature));
    % Enforce velMax limit
    map.velocity = min(velFromLatAccel, velMax);
    % Calculate time vector
    velAvg = movmean(map.velocity,2);
    dt = diff(map.arclength)./velAvg(1:end-1);
    map.time = [0,cumsum(dt)];
end

function [ arclengthLocalization ] = localizeVehicleOnPath(xVehicle, yVehicle, map)
    % Calculates a projection of the vehicle position onto the map and outputs the corresponding arclength

    % Calculate the vehicle's distance to each point of the map 
    distanceFromWaypoints = sqrt((xVehicle - map.x).^2 + (yVehicle - map.y).^2); 
    % Find the waypoint with the smallest distance to the vehicle
    [~, idxClosest] = min(distanceFromWaypoints);   

    % Identify second closest waypoint w.r.t. the vehicle
    % idx1 is the index of the first of the two points which make up the straight onto which the vehicle's
    % position is projected
    nElements = length(map.x);
    if distanceFromWaypoints(mod(nElements + idxClosest - 2,nElements)+1) < distanceFromWaypoints(mod(nElements + idxClosest,nElements)+1)
        idx1 = mod(nElements + idxClosest - 2,nElements)+1 ;
        idx2 = idxClosest;
    else
        idx1 = idxClosest;
        idx2 = mod(nElements + idxClosest,nElements)+1;
    end

    x1 = map.x(idx1);
    y1 = map.y(idx1);
    x2 = map.x(idx2);
    y2 = map.y(idx2);

    dx = x2-x1;
    dy = y2-y1;
    ds = hypot(dx,dy);

    % Calculate the projection and the corresponding arclength
    if ds < 1e-4
        lambda = 0;
    else
        lambda = -(dx*(x1-xVehicle)+dy*(y1-yVehicle))/(dx^2+dy^2);
    end
    arclengthLocalization = mod(map.arclength(idx1) + lambda*ds,map.arclength(end));
end

function [ pointsToTrack ] = resamplePathForTracker(stageVar, I, map, timeStep, horizonLength)
    % Localizes the vehicle on the map and resamples it for the tracker's prediction horizon
    
    % Localize
    arclengthLocalization = localizeVehicleOnPath(stageVar(I.xPos), stageVar(I.yPos), map);
    % Get map time corresponding to current localization
    timeReference = interp1(map.arclength, map.time, arclengthLocalization);
    % Time vector of prediction horizon
    timeHorizon = timeReference:timeStep:timeReference+(horizonLength-1)*timeStep;
    timeHorizon = mod(timeHorizon, map.time(end));
    % Interpolate map to get current waypoints
    pointsToTrack(1,:) = interp1(map.time, map.x, timeHorizon);
    pointsToTrack(2,:) = interp1(map.time, map.y, timeHorizon);    
end

function [] = plotResults(sqpGeneralSolvetimes, sqpGeneralFevalstimes, sqpGeneralQptimes, sqpFastSolvetimes, sqpFastFevalstimes, sqpFastQptimes)
% Plot benchmark results
    figure;
    plot(sqpFastSolvetimes*1000, 'Color',[1 0 0]); hold on
    plot(sqpGeneralSolvetimes*1000, 'Color',[0 0 1]);
    plot(sqpFastFevalstimes*1000, 'Color',[1 0.4 0.4]);
    plot(sqpGeneralFevalstimes*1000, 'Color',[0.4 0.4 1]);
    plot(sqpFastQptimes*1000, 'Color',[1 0.6 0.6]);
    plot(sqpGeneralQptimes*1000, 'Color',[0.6 0.6 1]);
    title('Solvetime comparison');
    legend('SQP\_FAST solvetime','SQP\_GENERAL solvetime','SQP\_FAST fevalstime','SQP\_GENERAL fevalstime','SQP\_FAST QPtime','SQP\_GENERAL QPtime');
    xlabel('Simulation steps');
    ylabel('Solvetime [ms]');

    stats = struct();
    stats.maxSqpFastSolvetime = max(sqpFastSolvetimes);
    stats.maxSqpGeneralSolvetime = max(sqpGeneralSolvetimes);
    stats.meanSqpFastSolvetime = mean(sqpFastSolvetimes); 
    stats.meanSqpGeneralSolvetime = mean(sqpGeneralSolvetimes);
    disp(stats);
end
