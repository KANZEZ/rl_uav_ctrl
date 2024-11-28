function [] = SoftConstraints()
% Simple car example illustrating the use of soft constraints.
%
% (c) Embotech AG, Zurich, Switzerland, 2020-2023.
    
    clc;
    close all;

    %% settings & dimensions
    N = 30;            % horizon length
    nvar = 7;          % number of variables
    neq  = 4;          % number of equality constraints
    nh = 1;            % number of inequality constraint functions

    deg2rad = @(x) x/180*pi;

    % initial state
    xinit = [-2, 0, 0, deg2rad(120)]; xinitidx = 1:4;

    % build matrices needed
    %       x      y       v    theta    F    s    slack
    % lbi = [ -3,    0       0     0      -5   -1];
    lbi = [ -3,    0       0       0    -5   -1     0];
    ubi = [  0,    3       2     +pi    +5   +1     5];

    x0i=lbi+(ubi-lbi)/2;
    model.x0=repmat(x0i',N,1);

    % upper/lower bounds for inequalities
    hu = 3;
    hl = 1;

    % indices on LHS of dynamical constraint
    %D = 1:4;
    E = [eye(4),zeros(4,3)];

    %% Set up and call solver
    % Set the model - multistage!
    model.N = N;
    model.nvar = nvar;
    model.neq = neq;
    model.nh = nh;
    model.npar = 3;

    model.E  = E;
    model.lb = lbi;
    model.ub = ubi;
    model.hu = hu;
    model.hl = hl;
    model.xinit = xinit;   
    model.xinitidx = xinitidx;

    model.objective         = @objective;
    model.ineq              = @inequalities;
    model.eq                = @dynamics;


    %% Generate FORCESPRO solver
    solvername = 'FORCESNLPsolver_simple_car';

    % get options
    codeoptions = getOptions(solvername);

    model.computeLagHessian = 1;

    % generate code
    FORCES_NLP(model, codeoptions);

    % run generated code
    problem.x0 = model.x0;
    problem.xinit = model.xinit';
    problem.all_parameters = repmat([1 0 -3]',N,1);

    [output,exitflag,info] = feval(solvername, problem); %#ok<FVAL,ASGLU>
end
