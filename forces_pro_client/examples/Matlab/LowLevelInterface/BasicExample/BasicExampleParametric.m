function [] = BasicExampleParametric()
% Simple MPC - double integrator example solved with FORCESPRO
%              using a convex interior-point algorithm
%              and passing dynamics as real-time parameters
% 
%  min   xN'*P*xN + sum_{i=1}^{N-1} xi'*Q*xi + ui'*R*ui
% xi,ui
%       s.t. x1 = x
%            x_i+1 = A*xi + B*ui  for i = 1...N-1
%            xmin <= xi <= xmax   for i = 1...N
%            umin <= ui <= umax   for i = 1...N
%
% and P is solution of Ricatti eqn. from LQR problem
%
% (c) Embotech AG, Zurich, Switzerland, 2014-2023.

    clc;
    close all;

    %% system
    A = [1.1 1; 0 1];
    B = [1; 0.5];
    [nx,nu] = size(B);

    %% MPC setup
    N = 11;
    Q = eye(nx);
    R = eye(nu);
    % terminal weight obtained from discrete-time Riccati equation
    if exist('dlqr','file')
        [~,P] = dlqr(A,B,Q,R);
    else
        P = [2.02390049, 0.269454848; 0.269454848, 2.652909941];
    end
    umin = -0.5;     umax = 0.5;
    xmin = [-5; -5]; xmax = [5; 5];


    %% FORCESPRO multistage form
    % assume variable ordering z(i) = [u_i; x_i+1] for i=1...N-1

    stages = MultistageProblem(N-1);
    for i = 1:N-1
        % dimension
        stages(i).dims.n = nx+nu; % number of stage variables
        stages(i).dims.r = nx;    % number of equality constraints
        stages(i).dims.l = nx+nu; % number of lower bounds
        stages(i).dims.u = nx+nu; % number of upper bounds
        
        % cost
        if (i == N-1)
            stages(i).cost.H = blkdiag(R,P);
        else
            stages(i).cost.H = blkdiag(R,Q);
        end
        stages(i).cost.f = zeros(nx+nu,1);
        
        % lower bounds
        stages(i).ineq.b.lbidx = 1:(nu+nx); % lower bound acts on these indices
        stages(i).ineq.b.lb = [umin; xmin]; % lower bound for this stage variable
        
        % upper bounds
        stages(i).ineq.b.ubidx = 1:(nu+nx); % upper bound acts on these indices
        stages(i).ineq.b.ub = [umax; xmax]; % upper bound for this stage variable
        
        % equality constraints
        if (i > 1)
            stages(i).eq.c = zeros(nx,1);
        end       
    end
    params(1) = newParam('minusA_times_x0',1,'eq.c'); % RHS of first eq. constr. is a parameter: c1 = -A*x0
    params(2) = newParam('eqC',[],'eq.C'); % Equality constraint matrix C
    params(3) = newParam('eqD',[],'eq.D'); % Equality constraint matrix D


    %% define outputs of the solver
    outputs(1) = newOutput('u0',1,1:nu);

    %% solver settings
    codeoptions = getOptions('myMPC_FORCESPRO');
    codeoptions.BuildSimulinkBlock = 1;

    %% generate code
    generateCode(stages,params,codeoptions,outputs);


	%% prepare (constant) dynamics to be passed as real-time vector
    singleEqC = [zeros(nx,nu), A];
    singleEqD = [B, -eye(nx)];
    eqC = repmat( singleEqC(:),N-1,1 );
    eqD = repmat( singleEqD(:),N-1,1 );
    
    %% simulate
    x1 = [-4; 2];
    kmax = 30;
    X = zeros(nx,kmax+1); X(:,1) = x1;
    U = zeros(nu,kmax);
    for k = 1:kmax
        problem.minusA_times_x0 = -A*X(:,k);
        problem.eqC = eqC;
        problem.eqD = eqD;
        [solverout,exitflag,info] = myMPC_FORCESPRO(problem);
        if( exitflag == 1 )
            U(:,k) = solverout.u0;
        else
            disp(info);
            error('Some problem in solver');
        end
        X(:,k+1) = A*X(:,k) + B*U(:,k);
    end

    %% plot
    figure(1); clf;
    subplot(2,1,1); grid on; title('states'); hold on;
    plot([1 kmax], [xmax xmax]', 'r--'); plot([1 kmax], [xmin xmin]', 'r--');
    ylim(1.1*[min(xmin),max(xmax)]); stairs(1:kmax,X(:,1:kmax)');
    subplot(2,1,2);  grid on; title('input'); hold on;
    plot([1 kmax], [umax umax]', 'r--'); plot([1 kmax], [umin umin]', 'r--');
    ylim(1.1*[min(umin),max(umax)]); stairs(1:kmax,U(:,1:kmax)');
end
