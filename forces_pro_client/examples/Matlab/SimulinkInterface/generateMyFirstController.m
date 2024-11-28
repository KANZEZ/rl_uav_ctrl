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
% Actual problem form: 
%  min   sum_{i=1}^{N} xi'*Q*xi + ui'*R*ui
% xi,ui
%       s.t. x1 = [0 6]'
%            x_i+1 = A*xi + B*ui  for i = 1...N-1
%            0 <= xi(2)           for i = 1...N
%            -5 <= ui <= 5        for i = 1...N
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
target_model = 'myFirstController_sim';
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
A = [0.7115 -0.4345; 0.4345 0.8853];
B = [0.2173; 0.0573];    
[nx,nu] = size(B);
C = eye(nx);
D = zeros(nx,1);

% horizon length
N = 10;

% weights in the objective
R = eye(nu);
Q = 10*eye(nx);    

% bounds
umin = -5;     umax = 5;
xmin = [-inf, 0]; xmax = [+inf, +inf];


%% 2) FORCESPRO multistage form
% assume variable ordering z(i) = [ui; xi] for i=1...N

% dimensions
model.N     = N;        % horizon length
model.nvar  = nx+nu;    % number of variables
model.neq   = nx;       % number of equality constraints

% objective 
model.objective = @(z) z(1)*R*z(1) + [z(2);z(3)]'*Q*[z(2);z(3)];

% equalities
model.eq = @(z) [ A(1,:)*[z(2);z(3)] + B(1)*z(1);
              A(2,:)*[z(2);z(3)] + B(2)*z(1)];

model.E = [zeros(2,1), eye(2)];

% inequalities
model.lb = [umin, xmin];
model.ub = [umax, xmax];

% initial state
model.xinitidx = 2:3;


%% 3) Generate FORCESPRO solver

% setup options
codeoptions = getOptions('myFirstController');
codeoptions.printlevel = 2;
codeoptions.cleanup = 0;
codeoptions.BuildSimulinkBlock = 1;

% generate code
output1 = newOutput('u0', 1, 1);
FORCES_NLP(model, codeoptions, output1);


%% 4) Model Initialization (needed for Simulink only)

% initial conditions
x_init = [0; 6];

% initial guess
x0 = zeros(model.N*model.nvar,1);

%% 5) Open Simulink Model

disp('### Open Simulink Model.');
start_of_step(interactive);

if compact
    forces_simulink_lib = [codeoptions.name, 'compact_lib'];
else
    forces_simulink_lib = [codeoptions.name, '_lib'];
end

% open a new simulink model (copied from the template)
copyfile('myFirstController_sim_template.slx', target_file);
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
    disp('Run myFirstController_createCoderMatlabFunction in myFirstController/interface.');
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
    myFirstController_createCoderMatlabFunction(target_model, codeoptions.name, compact, true);
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
