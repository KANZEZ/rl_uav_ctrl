function [] = Preview()
%% HOW TO: Preview Information in MPC Problem using FORCESPRO 
%
% (c) Gian Koenig, Embotech AG, Zurich, Switzerland, 2014-2023.

    clc; 
    close all;

    %% System Model
    A = [ 0.7115   -0.4345; ...
          0.4345    0.8853  ];
    B = [ 1; ...
          1 ];
    Bw = [ 1; ...
           1 ];  

    %% Generate Disturbance
    n = 200;

    road = zeros(n,1);
    for i=1:10
        k = i+15;
        road(k,1) = i*.2;
    end
    for i=1:10
        k = i+25;
        road(k,1) = 2 - i*.2;
    end

    %% Generate Solver: With Preview
    % MPC setup
    [nx, nu] = size(B);
    N = 11;
    Q = 10*eye(nx);
    R = eye(nu);
    if( exist('dlqr','file') )
        [~,P] = dlqr(A,B,Q,R);
    else
        P = 10*Q;
    end
    umin = -1.8;     umax = 1.8;

    % FORCESPRO multistage form
    % assume variable ordering z(i) = [ui; x_i+1] for i=1...N-1

    stages = MultistageProblem(N-1);

    % RHS of first eq. constr. is a parameter: z1=-A*x0 -Bw*Road
    params(1) = newParam('minusA_times_x0_BwPreview',1,'eq.c');
    % Parameter of Preview
    params(2) = newParam('dist2',2,'eq.c');
    params(3) = newParam('dist3',3,'eq.c');
    params(4) = newParam('dist4',4,'eq.c');
    params(5) = newParam('dist5',5,'eq.c');
    params(6) = newParam('dist6',6,'eq.c');
    params(7) = newParam('dist7',7,'eq.c');
    params(8) = newParam('dist8',8,'eq.c');
    params(9) = newParam('dist9',9,'eq.c');
    params(10) = newParam('dist10',10,'eq.c');

    for i = 1:N-1

            % dimension
            stages(i).dims.n = nx+nu; % number of stage variables
            stages(i).dims.r = nx;    % number of equality constraints        
            stages(i).dims.l = nu; % number of lower bounds
            stages(i).dims.u = nu; % number of upper bounds

            % cost
            if( i == N-1 )
                stages(i).cost.H = blkdiag(R,P);
            else
                stages(i).cost.H = blkdiag(R,Q);
            end
            stages(i).cost.f = zeros(nx+nu,1);

            % lower bounds
            stages(i).ineq.b.lbidx = 1; % lower bound acts on these indices
            stages(i).ineq.b.lb = umin; % lower bound for this stage variable

            % upper bounds
            stages(i).ineq.b.ubidx = 1; % upper bound acts on these indices
            stages(i).ineq.b.ub = umax; % upper bound for this stage variable

            % equality constraints
            if( i < N-1 )
                stages(i).eq.C =  [zeros(nx,nu), A];
            end
            stages(i).eq.D = [B, -eye(nx)];

    end 

    % define outputs of the solver
    outputs(1) = newOutput('u0',1,1);

    % solver settings
    codeoptions = getOptions('Preview_Controller');

    % generate code with preview
    generateCode(stages,params,codeoptions,outputs);

    % generate code without preview
    codeoptions = getOptions('NoPreview_Controller');
    for i = 2:N-1, stages(i).eq.c = zeros(nx,1); end
    generateCode(stages,params(1),codeoptions,outputs);

    %% Simulate System with Preview Controller
    x0 = [0; 0];
    kmax = 60;
    X = zeros(nx,kmax+1); X(:,1) = x0;
    U = zeros(nu,kmax);
    for k = 1:kmax
        problem.minusA_times_x0_BwPreview = -A*X(:,k) - Bw*road(k);
        problem.dist2 = -Bw*road(k+1);   problem.dist3 = -Bw*road(k+2);
        problem.dist4 = -Bw*road(k+3);   problem.dist5 = -Bw*road(k+4);
        problem.dist6 = -Bw*road(k+5);   problem.dist7 = -Bw*road(k+6);
        problem.dist8 = -Bw*road(k+7);   problem.dist9 = -Bw*road(k+8);
        problem.dist10 = -Bw*road(k+9);
        [solverout,exitflag,info] = Preview_Controller(problem);
        if( exitflag == 1 )
            U(:,k) = solverout.u0;
        else
            disp(info);
            error('Some problem in solver');
        end
        X(:,k+1) = A*X(:,k) + [B, Bw]*[U(:,k); road(k)];
    end

    x_preview = X;
    u_preview = U;

    %% Simulate System without Preview Controller
    x0 = [0; 0];
    kmax = 60;
    X = zeros(nx,kmax+1); X(:,1) = x0;
    U = zeros(nu,kmax);
    for k = 1:kmax
        problem.minusA_times_x0_BwPreview = -A*X(:,k);
        [solverout,exitflag,info] = NoPreview_Controller(problem);
        if( exitflag == 1 )
            U(:,k) = solverout.u0;
        else
            disp(info);
            error('Some problem in solver');
        end
        X(:,k+1) = A*X(:,k) + [B, Bw]*[U(:,k); road(k)];
    end

    x_nopreview = X;
    u_nopreview = U;

    %% 6. Plot
    kmax = 60; umin = -1.8; umax = 1.8;
    % INPUT PREVIEW MPC SIMULINK INTERFACE
    figure(1); 
    subplot(2,1,1);
    grid on; title('MPC with Preview vs No Preview (input)'); hold on;
    plot([1 kmax], [umax umax]', 'k--'); plot([1 kmax], [umin umin]', 'k--'); hold on;
    stairs(1:kmax, u_preview(:,1:kmax),'g'); hold on;
    stairs(1:kmax, u_nopreview(:,1:kmax),'g:')
    legend('umin','umax','u Preview','u no Preview');
    % STATES PREVIEW MPC SIMULINK INTERFACE
    subplot(2,1,2);
    grid on; title('MPC with Preview vs No Preview (states)'); hold on;
    stairs(1:kmax,x_preview(1,1:kmax)','r'); hold on;
    stairs(1:kmax,x_preview(2,1:kmax)','b'); hold on; 
    stairs(1:kmax,x_nopreview(1,1:kmax)','r:'); hold on;
    stairs(1:kmax,x_nopreview(2,1:kmax)','b:'); hold on;
    legend('x1 Preview', 'x2 Preview','x1 no Preview','x2 no Preview')
end
