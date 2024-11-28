function [model, codeoptions, I] = generatePathTrackingSolver(solverDir, timeStep, horizonLength, useFastSolver, floattype)
% This file generates the FORCESPRO solver for use within the PathTracking
% example. See PathTracking.m for more details.
% 
% See also PathTracking, FORCES_NLP
%
% (c) Embotech AG, Zurich, Switzerland, 2013-2023.
    
    if (nargin < 3)
        error('The function ''generatePathTrackingSolver'' is not meant to run standalone! Please run ''PathTracking'' instead.');
    end
    if nargin < 4
        useFastSolver = false;
    end
    if nargin < 5
        floattype = 'double';
    end

    %% Indices - struct I keeps track of variable sorting in z
    % Inputs
    I.FLon = 1; I.steeringRate = 2; 
    I.inputs = [I.FLon, I.steeringRate];
    % States
    I.xPos = 3; I.yPos = 4; I.velocity = 5; I.heading = 6; I.steeringAngle = 7;
    I.states = [I.xPos, I.yPos, I.velocity, I.heading, I.steeringAngle];
    
    %% Problem dimensions
    nInputs = numel(I.inputs);
    nStates = numel(I.states);
    model.N = horizonLength;            % Horizon length
    model.nvar = nInputs+nStates;       % Number of variables
    model.neq  = nStates;               % Number of equality constraints
    model.npar = 2;                     % Number of runtime parameters (waypoint coordinates)

    %% Objective function 
    % Definitions of LSobj and LSobjN further below in this file
    model.LSobjective = @(z,p)LSobj(z,p,I);
    model.LSobjectiveN = @(z,p)LSobjN(z,p,I); % Increased costs for the last stage

    %% Dynamics, i.e. equality constraints 
    % We use an explicit RK4 integrator here to discretize continuous dynamics:
    model.eq = @(z) RK4( z(I.states), z(I.inputs), @(x,u)continuousDynamics(x,u,I),... 
        timeStep);

    % Indices on LHS (left hand side) of dynamical constraint - for efficiency reasons
    % Make sure the matrix E has structure [0 I] where I is the identity matrix
    model.E = [zeros(nStates,nInputs), eye(nStates)];

    %% Inequality constraints
    % Upper/lower variable bounds lb <= z <= ub
    %           inputs               |  states
    %           FLon   steeringRate     xPos    yPos  velocity  heading   steeringAngle   
    model.lb = [ -5.,  deg2rad(-90),   -100.,  -100.,   0.,     -inf,     deg2rad(-50)]; 
    model.ub = [ +5.,  deg2rad(90),     100.,   100.,   5.,     +inf,     deg2rad(50)];

    %% Initial conditions
    % Initial condition on vehicle states
    model.xinitidx = I.states; % Use this to specify on which variables initial conditions are imposed

    %% Define solver options
    if useFastSolver
        codeoptions = ForcesGetDefaultOptions('PathTrackSolver','SQP_NLP_fast',floattype);
    else
        codeoptions = ForcesGetDefaultOptions('PathTrackSolver','SQP_NLP',floattype);
    end
    codeoptions.maxit = 200;        % Maximum number of iterations
    codeoptions.optlevel = 3;       % 0: No optimization, good for prototyping
    codeoptions.timing = 1;
    codeoptions.printlevel = 0;
    codeoptions.nohash = 1;         % Enforce solver regeneration
    codeoptions.overwrite = 1;      % Overwrite existing solver
    codeoptions.BuildSimulinkBlock = 0;
    codeoptions.server = 'https://forces.embotech.com';
    
    %% Generate FORCESPRO solver
    cd(solverDir);
    
end

function [xDot] = continuousDynamics(x,u,I)
    
    % Number of inputs is needed for indexing
    nu = numel(I.inputs);
    
    % Set physical constants
    l_r = 0.5; % Distance rear wheels to center of gravity of the car
    l_f = 0.5; % Distance front wheels to center of gravity of the car
    m = 1.0;   % Mass of the car
    
    % Set parameters
    beta = atan(l_r/(l_r + l_f) * tan(x(I.steeringAngle-nu)));
    
    % Calculate dx/dt
    xDot = [x(I.velocity-nu) * cos(x(I.heading-nu) + beta);    % dxPos/dt = v*cos(theta+beta)
            x(I.velocity-nu) * sin(x(I.heading-nu) + beta);    % dyPos/dt = v*cos(theta+beta)
            u(I.FLon)/m;                                       % dv/dt = F/m
            x(I.velocity-nu)/l_r * sin(beta);                  % dtheta/dt = v/l_r*sin(beta)
            u(I.steeringRate)];                                % ddelta/dt = phi

end

function [r] = LSobj(z,currentTarget, I)
% Least square costs on deviating from the path and on the inputs
% currentTarget = point on path that is tracked in this stage

    r = [sqrt(200.0)*(z(I.xPos)-currentTarget(1));	% costs for deviating from the path in x-direction
         sqrt(200.0)*(z(I.yPos)-currentTarget(2));	% costs for deviating from the path in y-direction
         sqrt(0.2)*z(I.FLon);                       % penalty on input FLon
         sqrt(10.0)*z(I.steeringRate)];             % penalty on input steeringRate
end

function [r] = LSobjN(z,currentTarget, I)
% Increased least square costs for last stage on deviating from the path and on the inputs
% currentTarget = point on path that is tracked in this stage

    r = [sqrt(400.0)*(z(I.xPos)-currentTarget(1));	% costs for deviating from the path in x-direction
        sqrt(400.0)*(z(I.yPos)-currentTarget(2));	% costs for deviating from the path in y-direction
        sqrt(0.2)*z(I.FLon);                        % penalty on input FLon
        sqrt(10.0)*z(I.steeringRate)];              % penalty on input steeringRate
end
