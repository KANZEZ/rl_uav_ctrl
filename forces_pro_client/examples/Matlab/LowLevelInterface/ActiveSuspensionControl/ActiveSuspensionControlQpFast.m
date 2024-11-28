%% Active Suspension Control using QP FAST
%
% Model Predictive Control Designed with FORCESPRO
%
% (c) Gian Koenig, Embotech AG, Zurich, Switzerland, 2014-2023.

%% System
% output vector (3 outputs)
% - zb_ddot     : heave acceleration 
% - phi_ddot    : pitch acceleration 
% - theta_ddot  : roll acceleration

% REDUCED MODEL : state vector (6 states)
% - zb          : heave displacement 
% - phi         : pitch displacement
% - theta       : roll displacement
% - zb_dot      : heave velocity 
% - phi_dot     : pitch velocity
% - theta_dot   : roll velocity

clear all;
clc;
close all;

load('Active_Suspension_Control_Model.mat')

%% Road Bump

n = 3600;
t = (0:.005:n*.005-.005)';
road1 = zeros(length(t),1);
road2 = zeros(length(t),1);

for i = 1:20
    k = i+160;
    road1(k,1) = i*.005;
end
for i = 1:20
    k = i+180;
    road1(k,1) = .1 - i*.005;
end

%% Simulate the passive model

sim('Active_Suspension_Control_NC')

% System Response: output_nocontrol
Y_NC = output_nocontrol.Data;
t_sim = output_nocontrol.Time;

% System Input: w_ii, w_dot_ii, ii = {fl, fr, rl, rr}
w = [w.signal1.Data, w.signal2.Data,...
    w.signal3.Data,w.signal4.Data...
    w_dot.signal1.Data,w_dot.signal2.Data,...
    w_dot.signal3.Data,w_dot.signal4.Data]';


%% Measurement data for control and for plotting
% Reduce Resolution
N=20;
kmax_pre = 720;
w_pre_temp = [w zeros(8,N)];
w_pre = [zeros(8,kmax_pre) zeros(8,N)];

t_pre = (0:0.025:kmax_pre*0.025-0.025)';
k=1;
for i = 1:kmax_pre
    w_pre(:,i) = w_pre_temp(:,k+1);
    k = 5*i;
end

%% MPC Setup
[stages, codeoptions, parameters, outputs] = getCodegenerationData(Ad,Bdu);
nx = stages(1).dims.r;
nu = stages(1).dims.n - stages(1).dims.r;

codeoptions.server = 'https://forces.embotech.com';

% Generate solver for caching problem data
generateCode(stages,parameters,codeoptions,outputs);
[ cache, ~, Y_pdip, ~, solvetime_pdip ] = runSimulation(codeoptions.name,kmax_pre,nx,nu,Ad,Bdu,Bdw,Cd,Dd,w_pre);

% Generate QP_FAST solver
tuningoptions = ForcesAutotuneOptions(cache);
tuningoptions.setValidationControlOutput("u0",1:4);

ForcesGenerateQpFastSolver(stages,parameters,codeoptions,tuningoptions,outputs);
[ ~, ~, Y_fast, ~, solvetime_fast ] = runSimulation(codeoptions.name,kmax_pre,nx,nu,Ad,Bdu,Bdw,Cd,Dd,w_pre);

% Plot results
plotResults(t_pre,t_sim,w,Y_pdip,Y_fast,solvetime_pdip,solvetime_fast);


function plotResults(t_pre,t_sim,w,Y_pdip,Y_fast,solvetime_pdip,solvetime_fast)
% Plots
    
    % Road Disturbance
    figure(1);
    plot(t_sim, w(2,:)); grid on; 
    title('Speed Bump: Front Right Wheel', 'FontSize', 16); 
    ylim([0, .11]); xlim([0.3, 1.5])
    h_xlabel = xlabel('Time'); h_ylabel = ylabel('Road w_{fr}');
    set(h_xlabel, 'FontSize', 14); set(h_ylabel, 'FontSize', 14);
    set(gcf,'PaperUnits','inches','PaperPosition',[0 0 10 2]);
    
    % Heave Acceleration
    figure(2);
    stairs(t_pre, Y_fast(1,:),'b'); hold on; stairs(t_pre, Y_pdip(1,:),'r'); grid on; 
    title('Heave Acceleration', 'FontSize', 16); 
    ylim([-2, 2]); xlim([.3, 1.5])
    h_xlabel = xlabel('Time'); h_ylabel = ylabel('Heave Acceleration [m/s^2]');
    set(h_xlabel, 'FontSize', 14); set(h_ylabel, 'FontSize', 14);
    h_legend=legend('heave acc QP\_FAST', 'heave acc PDIP','Location','SouthWest'); grid on;
    set(h_legend,'FontSize',14);
    
    % Pitch Acceleration
    figure(3);
    stairs(t_pre, Y_fast(2,:),'b'); hold on; stairs(t_pre, Y_pdip(2,:),'r'); grid on; 
    title('Pitch Acceleration', 'FontSize', 16); 
    ylim([-2, 2]); xlim([.3, 1.5])
    h_xlabel = xlabel('Time'); h_ylabel = ylabel('Pitch Acceleration [m/s^2]');
    set(h_xlabel, 'FontSize', 14); set(h_ylabel, 'FontSize', 14);
    h_legend=legend('pitch acc QP\_FAST', 'pitch acc PDIP','Location','SouthWest'); grid on;
    set(h_legend,'FontSize',14);
    
    % Roll Acceleration
    figure(4); 
    stairs(t_pre, Y_fast(3,:),'b'); hold on; stairs(t_pre, Y_pdip(3,:),'r'); grid on; 
    title('Roll Acceleration', 'FontSize', 16); 
    ylim([-2, 2]); xlim([.3, 1.5])
    h_xlabel = xlabel('Time'); h_ylabel = ylabel('Roll Acceleration [m/s^2]');
    set(h_xlabel, 'FontSize', 14); set(h_ylabel, 'FontSize', 14);
    h_legend=legend('roll acc QP\_FAST', 'roll acc PDIP','Location','SouthWest'); grid on;
    set(h_legend,'FontSize',14);

    % Solvetime comparison
    figure(5); 
    plot(t_pre, solvetime_fast*1000); hold on; plot(t_pre, solvetime_pdip*1000); grid on; 
    title('Solvetime Comparison', 'FontSize', 16); 
    h_xlabel = xlabel('Time'); h_ylabel = ylabel('Solvetime [ms]');
    set(h_xlabel, 'FontSize', 14); set(h_ylabel, 'FontSize', 14);
    h_legend=legend('QP\_FAST', 'PDIP','Location','NorthEast'); grid on;
    set(h_legend,'FontSize',14);
end

function [ cache, X, Y, U, solvetimes ] = runSimulation(solvername,kmax_pre,nx,nu,Ad,Bdu,Bdw,Cd,Dd,w_pre)
% Simulate
    x1 = zeros(6,1);
    X = zeros(nx,kmax_pre+1); X(:,1) = x1;
    Y = zeros(3,kmax_pre);
    U = zeros(nu,kmax_pre);
    cache = cell(kmax_pre,1);
    solvetimes = zeros(kmax_pre,1);
    nSolves = 100;
    for k = 1:kmax_pre
        problem.minusA_times_x0_minusBw_times_w = -Ad*X(:,k)-Bdw*w_pre(:,k);
        for ii = 1:nSolves
            [solverout,exitflag,info] = feval(solvername,problem);
            solvetimes(k) = solvetimes(k) + info.solvetime;
        end
        solvetimes(k) = solvetimes(k) / nSolves;
        if( exitflag == 1 )
            U(:,k) = solverout.u0;
        else
            disp(info);
            error('Some problem in solver');
        end
        cache{k} = problem;
        X(:,k+1) = Ad*X(:,k) + [Bdu, Bdw]*[U(:,k); w_pre(:,k)];
        Y(:,k) = Cd*X(:,k) + Dd*[U(:,k); w_pre(:,k)];
    end
end

function [stages, codeoptions, parameters, outputs] = getCodegenerationData(Ad,Bdu)
% Getter for all codegeneration data

    nx = 6;
    nu = 4; 
    
    N = 20;
    Q = diag([50,50,50,50,50,50]);
    R = eye(nu);
    if( exist('dlqr','file') )
        [~,P] = dlqr(Ad,Bdu,Q,R);
    else
        P = 20*Q;
    end
    umin = -.04;     umax = .04;
    
    % Assume variable ordering zi = [ui; xi+1] for i=1...N-1
    
    % Parameter
    parameters = newParam('minusA_times_x0_minusBw_times_w',1,'eq.c');
    
    stages = MultistageProblem(N);
    for i = 1:N
    
            % dimension
            stages(i).dims.n = nx+nu; % number of stage variables
            stages(i).dims.r = nx;    % number of equality constraints        
            stages(i).dims.l = nu; % number of lower bounds
            stages(i).dims.u = nu; % number of upper bounds
    
            % cost
            if( i == N )
                stages(i).cost.H = blkdiag(R,P);
            else
                stages(i).cost.H = blkdiag(R,Q);
            end
            stages(i).cost.f = zeros(nx+nu,1);
    
            % lower bounds
            stages(i).ineq.b.lbidx = 1:nu; % lower bound acts on these indices
            stages(i).ineq.b.lb = umin*ones(4,1); % lower bound for the input signal
    
            % upper bounds
            stages(i).ineq.b.ubidx = 1:nu; % upper bound acts on these indices
            stages(i).ineq.b.ub = umax*ones(4,1); % upper bound for the input signal
    
            % equality constraints
            if( i < N )
                stages(i).eq.C =  [zeros(nx,nu), Ad];
            end
            if( i>1 )
                stages(i).eq.c = zeros(nx,1);
            end
            stages(i).eq.D = [Bdu, -eye(nx)];
    
    end
    
    % define outputs of the solver
    outputs(1) = newOutput('u0',1,1:nu);    
    
    % solver settings
    codeoptions = getOptions('VEHICLE_MPC_noPreview');
    codeoptions.printlevel = 0;

end
