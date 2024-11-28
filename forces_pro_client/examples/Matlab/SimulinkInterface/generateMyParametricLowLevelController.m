% Simple MPC regulator example for use with FORCESPRO
% 
% General problem form:
%  min   xN'*P*xN + sum_{i=1}^{N-1} xi'*Q*xi + ui'*R*ui
% xi,ui
%       s.t. x1 = x
%            x_i+1 = A*xi + B*ui  for i = 1...N-1
%            xmin <= xi <= xmax   for i = 1...N
%            umin <= ui <= umax   for i = 1...N
%
% (c) Embotech AG, Zurich, Switzerland, 2015-2023.

clc;
close all;

%% SIMULINK INTERFACE SETTINGS
% To use FORCESPRO in a Simulink model, there are two interfaces available:
%     *  A S-function Simulink Block using the non-inlined 
%     FORCESPRO S-function (to use: set interface as 'sfunction')
%     * A MATLAB Code Simulink Block which uses the coder interface of FORCESPRO 
%     and produces an inlined FORCESPRO S-function (to use: set interface as 'coder') 
interface = 'sfunction';
% The FORCESPRO Simulink block is available in two versions, the standard
% and the compact. The compact version combines/stacks inputs and outputs that
% refer to the same problem matrices when the stage dimensions are the same. 
% To use the standard version instead, set compact to false
compact = true;
% set to false to run whole execution without stopping at each step
interactive = true;


%% Prerequisites

% stop any previous executions and delete previous simulink models
target_model = 'myParametricLowLevelController_sim';
close_system(target_model, 0);
target_file = [target_model, '.slx'];
if exist(target_file, 'file')
    delete(target_file);
end

if ~strcmp(interface, 'sfunction') && ~strcmp(interface, 'coder')
    error('Invalid selection for interface. Only ''sfunction'' and ''coder'' options are supported');
end

if ~license('test', 'Simulink')
    error('This example requires Simulink to be installed');
end

if strcmp(interface, 'coder') && ~license('test', 'Real-Time_Workshop')
    error('The coder interface for the FORCESPRO Simulink Block requires Simulink Coder to be installed');
end

%% 1) MPC parameter setup

% system
A = [1.1 1; 0 1];
B = [1; 0.5];

[nx,nu] = size(B);
C = eye(nx);
D = zeros(nx,nu);

% MPC setup
N = 11;
Q = eye(nx);
R = eye(nu);
% terminal weight obtained from discrete-time Riccati equation
if exist('dlqr','file')
    [~,P] = dlqr(A,B,Q,R);
else
    P = [2.02390049, 0.269454848; ...
         0.269454848, 2.652909941];
end
umin = -0.5;     umax = 0.5;
xmin = [-5; -5]; xmax = [5; 5];


%% 2) FORCESPRO multistage form
% assume variable ordering z(i) = [ui; x_i+1] for i=1...N-1

stages = MultistageProblem(N-1);
for i = 1:N-1
    
    % dimension
    stages(i).dims.n = nx+nu; % number of stage variables
    stages(i).dims.r = nx;    % number of equality constraints
    stages(i).dims.l = nx+nu; % number of lower bounds
    stages(i).dims.u = nx+nu; % number of upper bounds
    
    % cost
    if( i == N-1 )
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
    if( i>1 )
        stages(i).eq.c = zeros(nx,1);
    end
    
end
params(1) = newParam('minusA_times_x0',1,'eq.c'); % RHS of first eq. constr. is a parameter: z1=-A*x0
params(2) = newParam('eqC',[],'eq.C'); % Equality constraint matrix C
params(3) = newParam('eqD',[],'eq.D'); % Equality constraint matrix D

% use those quantities to assign system matrices at runtime
singleEqC = [zeros(nx,nu), A];
singleEqD = [B, -eye(nx)];
eqC = repmat( singleEqC(:),N-1,1 );
eqD = repmat( singleEqD(:),N-1,1 );


%% define outputs of the solver
outputs(1) = newOutput('u0',1,1:nu);


%% 3) Generate FORCESPRO solver

% setup options
codeoptions = getOptions('myParametricLowLevelController');
codeoptions.printlevel = 2;
codeoptions.cleanup = 0;
codeoptions.BuildSimulinkBlock = 1;

% generate code
generateCode(stages,params,codeoptions,outputs);


%% 4) Model Initialization (needed for Simulink only)

% initial conditions
x_init = [-4; 2];


%% 5) Open Simulink Model

disp('### Open Simulink Model.');
start_of_step(interactive);

if compact
    forces_simulink_lib = [codeoptions.name, 'compact_lib'];
else
    forces_simulink_lib = [codeoptions.name, '_lib'];
end

% open a new simulink model (copied from the template)
copyfile('myParametricLowLevelController_sim_template.slx', target_file);
open_system(target_model);

%% 6) Add FORCESPRO Simulink block to Simulink model
if strcmp(interface, 'sfunction')
    fullpath = fullfile(codeoptions.name, 'interface', forces_simulink_lib);
    open_system(fullpath);
    
    disp('### Add FORCESPRO Simulink block to Simulink model.');
    disp(['Open FORCESPRO Simulink library at ', fullpath]);
    disp('Copy the Simulink block to the Simulink model.');
    start_of_step(interactive);
    
    % for the sfunction case open the library in
    % myFirstController/interface/myFirstController_compactlib.mdl and manually 
    % copy the Simulink block to the Simulink model
    add_block([forces_simulink_lib, '/', codeoptions.name], [target_model, '/', codeoptions.name]);
    close_system(forces_simulink_lib, 0);
else
    disp('### Change Simulink model Solver to Fixed Step.');
    start_of_step(interactive);
    
    % the coder interface only supports Fixed-Step Simulations
    set_param(target_model, 'Solver', 'FixedStep');
    
    addpath(fullfile(codeoptions.name, 'interface'));
    
    disp('### Add FORCESPRO Simulink block to Simulink model.');
    disp('Run myParametricLowLevelController_createCoderMatlabFunction in myParametricLowLevelController/interface.');
    start_of_step(interactive);
    
    % for the coder case run myFirstController_createCoderMatlabFunction in
    % myFirstController/interface to create a FORCESPRO Simulink block to
    % the Simulink model:
    %     * The first parameter is the target Simulink model
    %     * The second parameter is the name of the created Simulink model
    %     * The third parameter selects whether to use the standard or the
    %       compact version
    %     * The fourth parameter must be set to true if the Simulink model
    %       already exists (by default the script always tries to create a
    %       new Simulink model)
    myParametricLowLevelController_createCoderMatlabFunction(target_model, codeoptions.name, compact, true);
    rmpath(fullfile(codeoptions.name, 'interface'));
end

%% 7) Connect inputs and outputs of FORCESPRO Simulink block (manually)
disp('### Connect inputs and outputs of FORCESPRO Simulink block.');
start_of_step(interactive);

% this position is pre-determined so that the ports will automatically
% connect to inputs and outputs
set_param([target_model, '/', codeoptions.name], 'Position', [290   130   670   290]);

%% 8) Run Simulink model simulation and check results
disp('### Run Simulink model simulation.');
start_of_step(interactive);

sim(target_model);
disp('Done. Check simulation results.');

function start_of_step(interactive)
% Defines the start of a step in the example.
% If interactive is set to true, the example will
% pause and wait for the user to press a key before
% continuing. If set to false, execution will not stop.
    if nargin < 1
        interactive = true;
    end
    press_key_msg = 'Press key to continue...';
    if interactive
        commandwindow;
        input(press_key_msg);
    end
end
