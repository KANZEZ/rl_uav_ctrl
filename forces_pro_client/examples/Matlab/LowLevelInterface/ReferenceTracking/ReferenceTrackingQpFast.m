function [] = ReferenceTrackingQpFast()
% Reference Tracking with FORCESPRO's QP_FAST solver
%
% (c) Embotech AG, Zurich, Switzerland, 2013-2023. Email: support@embotech.com

    %% Create and go to temporary directory for solver generation
    [ startDir, tmpDir ] = goToTempDir();

    %% Generate general solver for collecting tuning problems/data
    [stages, params, codeoptions, outputs] = getSolverGenerationData();
    codeoptions.solvemethod = 'PDIP';
    disp('Generating solver for collecting tuning problems/data...')
    generateCode(stages,params,codeoptions,outputs);
    tuningProblems = runSimulation(codeoptions.name, false);

    %% Generate QP_FAST solver and autotune it
    tuningoptions = ForcesAutotuneOptions(tuningProblems);
    ForcesGenerateQpFastSolver(stages,params,codeoptions,tuningoptions,outputs);
    
    %% Run simulation with QP_FAST solver
    runSimulation(codeoptions.name, true);

    %% Clean up
    cd(startDir);
    clear('mex'); %#ok<CLMEX>
    rmdir(tmpDir, 's');

end

function [stages, params, codeoptions, outputs] = getSolverGenerationData()
% Returns all data needed to generate a FORCESPRO solver

    %% System
    [A, B, Q, R, ~] = getMatrices();   
    [nx,nu] = size(B);

    %% MPC setup
    N = 11;
    [umin, umax, xmin, xmax] = getBounds();

    %% FORCESPRO multistage form
    % assume variable ordering z(i) = [ui; x_i+1] for i=1...N-1

    stages = MultistageProblem(N-1);
    for i = 1:N-1

            % dimension
            stages(i).dims.n = nx+nu; % number of stage variables
            stages(i).dims.r = nx;    % number of equality constraints        
            stages(i).dims.l = nx+nu; % number of lower bounds
            stages(i).dims.u = nx+nu; % number of upper bounds

            % cost
            stages(i).cost.H = blkdiag(R,Q);

            % lower bounds
            stages(i).ineq.b.lbidx = 1:(nu+nx); % lower bound acts on these indices
            stages(i).ineq.b.lb = [umin; xmin]; % lower bound for this stage variable

            % upper bounds
            stages(i).ineq.b.ubidx = 1:(nu+nx); % upper bound acts on these indices
            stages(i).ineq.b.ub = [umax; xmax]; % upper bound for this stage variable

            % equality constraints
            if( i < N-1 )
                stages(i).eq.C = [zeros(nx,nu), A];
            end
            if( i>1 )
                stages(i).eq.c = zeros(nx,1);
            end
            stages(i).eq.D = [B, -eye(nx)];

    end
    params(1) = newParam('Reference_Value',1:N-1,'cost.f'); % Reference Value on the states and inputs
    params(2) = newParam('minusA_times_x0',1,'eq.c'); % RHS of first eq. constr. is a parameter   


    %% Define outputs of the solver
    outputs(1) = newOutput('u0',1,1:nu);

    %% Solver settings
    codeoptions = getOptions('FPsolver');
    codeoptions.printlevel = 0;
end

function [ collectedProblems ] = runSimulation(solvername, doPlot)
% Run a simulation

    if nargin < 1
        error('solver name must be passed in order to be able to run solver.');
    end
    if nargin < 2
        doPlot = false;
    end

    %% Simulate
    [A, B, Q, R] = getMatrices();
    [nx,nu] = size(B);
    x1 = [-5; 4];
    kmax = 35;
    X = zeros(nx,kmax+1); X(:,1) = x1;
    X_ref = zeros(nx,kmax+1); X_ref(1,14:end) = 2; X_ref(2,22:end) = -1;
    U = zeros(nu,kmax);
    U_ref = zeros(nu,kmax);
    collectedProblems = cell(kmax,1);
    for k = 1:kmax
        U_ref(:,k) = B\(eye(nx)-A)*X_ref(:,k);
        problem.Reference_Value = [-U_ref(:,k)'*R, -X_ref(:,k)'*Q]';
        problem.minusA_times_x0 = -A*X(:,k);
        problem.warmstart = 1;
        [solverout,exitflag,info] = feval(solvername, problem);
        collectedProblems{k} = problem;
        if( exitflag == 1 )
            U(:,k) = solverout.u0;
        else
            disp(info);
            error('Some problem in solver');
        end
        X(:,k+1) = A*X(:,k) + B*U(:,k);
    end

    if doPlot
        %% Plot
        [umin, umax, xmin, xmax] = getBounds();        
        h = figure(1); clf;
        grid on; h_title = title('States x'); set(h_title,'FontSize',14); hold on;
        stairs(1:kmax,X(1,1:kmax)','g'); hold on; 
        stairs(1:kmax,X(2,1:kmax)','b'); hold on;
        stairs(1:kmax,X_ref(1,1:kmax),'k--'); hold on; 
        stairs(1:kmax,X_ref(2,1:kmax),'k--'); hold on;
        plot([1 kmax], [xmax xmax]', 'r--'); hold on;
        plot([1 kmax], [xmin xmin]', 'r--'); hold on;
        h_xlabel=xlabel('Time step k'); h_ylabel=ylabel('Maginute');
        set(h_xlabel, 'FontSize', 12); set(h_ylabel, 'FontSize', 12);
        h_legend = legend('x_1','x_2'); set(h_legend,'FontSize',14);
        ylim(1.1*[min(xmin),max(xmax)]);
        hline = findobj(gcf, 'type', 'line');
        set(hline,'LineWidth',1.2);
        set(gcf,'PaperUnits','inches','PaperPosition',[0 0 7 4])
        print(h,'-depsc','States_Reference_Tracking');
    
        h = figure(2); clf;
        grid on; h_title = title('Input Signal u'); set(h_title,'FontSize',14); hold on;
        stairs(1:kmax,U(1,1:kmax)','g'); hold on;
        stairs(1:kmax,U(2,1:kmax)','b'); hold on;
        plot([1 kmax], [umax umax]', 'r--'); hold on;
        plot([1 kmax], [umin umin]', 'r--');
        h_xlabel=xlabel('Time step k'); h_ylabel=ylabel('Maginute');
        set(h_xlabel, 'FontSize', 12); set(h_ylabel, 'FontSize', 12);
        ylim(1.1*[min(umin),max(umax)]);
        h_legend = legend('u_1','u_2'); set(h_legend,'FontSize',14);
        hline = findobj(gcf, 'type', 'line');
        set(hline,'LineWidth',1.2);
        set(gcf,'PaperUnits','inches','PaperPosition',[0 0 7 4])
        print(h,'-depsc','Input_Reference_Tracking');
    end

end

function [A, B, Q, R, P] = getMatrices()
% Returns all matrices in the model

    A = [1.1 1; 0 1];
    B = [.8, 0.1;0.3,.8];
    [nx,nu] = size(B);
    Q = eye(nx);
    R = eye(nu);
    P = [ 1.9997, 0.8070;
          0.8070, 2.4904 ];    
end

function [umin, umax, xmin, xmax] = getBounds()
% Returns all bounds in the model

    umin = [-.8;-.8];     
    umax = [1.3;1.3];
    xmin = [-5; -5];      
    xmax = [5; 5];
end

function [ startDir, tmpDir ] = goToTempDir()
% Small helper fcn for entering temporary directory

    startDir = pwd();   
    tmpDir = fullfile(fileparts(mfilename('fullpath')),'TEMP_DIR');
    if isfolder(tmpDir)
        clear('mex'); %#ok<CLMEX>
        rmdir(tmpDir,'s');      
    end
    mkdir(tmpDir); cd(tmpDir);
end
