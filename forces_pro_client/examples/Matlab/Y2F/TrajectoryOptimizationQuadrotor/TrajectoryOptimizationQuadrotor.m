function [] = TrajectoryOptimizationQuadrotor()
% Trajectory Optimization for Quadrotor Flight with FORCESPRO using
% Y2F and Yalmip to formulate
% 
% Written by Aldo Zgraggen, inspired by the formulation in
% S. Liu et al., "Planning Dynamically Feasible Trajectories for 
% Quadrotors Using Safe Flight Corridors in 3-D Complex Environments," 
% IEEE Robotics and Automation Letters, vol. 2, no. 3, pp. 1688-1695, 
% July 2017.
%
% See the FORCESPRO documentation for more details.
%
% (c) Embotech AG, Zurich, Switzerland, 2020-2023.


    %% Clear from brown fields
    clc;
    close all;

    %% Parameters
    nStates = 4;  % [-]   Number of states
                  %       Flat outputs [x position; y position; z position; yaw angle]
    n = 8;        % [-]   Order of piece-wise polynomial used as basis function                   
    nSample = 5;  % [-]   Number of intermediate samples (where constraints are checked)

    withVisualization = true;  % [-]   Bool if MPT Toolbox for visualization is installed
    bbConstr = false;          % [-]   true:   bounding-box constraints (separable in coordinates) (n=7,8,9)
                               %       false:  Polyhedron along path (non-separable polytopic constraints) (n=8)

    if withVisualization && (~exist('mpt_init','file') || ~exist('cddmex','file'))
        warning('Visualization requires the MPT Toolbox and the ccd library to be installed.')
        withVisualization = false;
    end


    %% WayPoints and time needed for segment
    % Simple case with 3 segments
    p0 = [0;0;0;0];
    p1 = [1;1;1;0];
    p2 = [3;1;1;pi];
    p3 = [4;2;2;pi];

    pathSegments = [p0,p1,p2,p3];
    N = size(pathSegments,2) - 1;

    % deltaT being the same for all segments
    deltaT = 1*ones(N,1);


    %% Create artificial Polyhedrons around path segments
    Polyh = cell(N,1);
    Ap = cell(N,1);
    bp = cell(N,1);
    dist = 0.2;
    constrIdx = [1,2,3,4,5,6];    % which constraints defined subsequently to be actually applied

    if bbConstr
        % Create artificial boxes around path segments (separable constraints -> upper and lower bounds)
        % Constraint Indices:
        %   1: max X
        %   2: max Y
        %   3: max Z
        %   4: min X
        %   5: min Y
        %   6: min Z
        for s=1:N
            p_s  = pathSegments(1:3,s);     % Segment start point   (idx=s)
            p_s1 = pathSegments(1:3,s+1);   % Segment end point     (idx=s+1)
            % max points
            pMat = [p_s,p_s1];
            [maxS,idxMaxS] = max(pMat,[],2);
            [minS,idxMinS] = min(pMat,[],2);

            % normal vectors
            ns_xp = [1;0;0];
            ns_xn = -ns_xp;
            ns_yp = [0;1;0];
            ns_yn = -ns_yp;
            ns_zp = [0;0;1];
            ns_zn = -ns_zp;

            % Construct extreme points in bounding box planes
            maxSX = pMat(:,idxMaxS(1)) + dist*ns_xp;
            maxSY = pMat(:,idxMaxS(2)) + dist*ns_yp;
            maxSZ = pMat(:,idxMaxS(3)) + dist*ns_zp;
            minSX = pMat(:,idxMinS(1)) + dist*ns_xn;
            minSY = pMat(:,idxMinS(2)) + dist*ns_yn;
            minSZ = pMat(:,idxMinS(3)) + dist*ns_zn;

            % Construct plane offset from normal vector and point on plane
            bXp = -ns_xp'*maxSX;
            bYp = -ns_yp'*maxSY;
            bZp = -ns_zp'*maxSZ;
            bXn = -ns_xn'*minSX;
            bYn = -ns_yn'*minSY;
            bZn = -ns_zn'*minSZ;

            % Define halfspace inequality matrix and offset
            A = [ns_xp';ns_yp';ns_zp';ns_xn';ns_yn';ns_zn'];
            b = [-bXp;-bYp;-bZp;-bXn;-bYn;-bZn];

            Ap{s,1} = A;
            bp{s,1} = b;

            if withVisualization
                % Plane Equation: A*x + b == 0
                % Polyhedron (from MPT) P = { x | A*x <= b}
                Polyh{s,1} = Polyhedron('A',A,'b',b);

                assert(~Polyh{s,1}.isEmptySet,'Polyhedron has empty interior')
                assert(Polyh{s,1}.isBounded,'Polyhedron is unbounded')
            end
        end
    else
        % The created polyhedron are square tubes around the path segment and thus create a non-separable problem
        % Plane Equation: A*x + b == 0
        % Polyhedron (from MPT) P = { x | A*x <= b}
        % Constraint Indices:
        %   1: Start Plane
        %   2: End Plane
        %   3: Upper Plane
        %   4: Lower Plane
        %   5: Left Plane
        %   6: Right Plane
        for s=1:N
            p_s  = pathSegments(1:3,s);     % Segment start point   (idx=s)
            p_s1 = pathSegments(1:3,s+1);   % Segment end point     (idx=s+1)

            rP = p_s1 - p_s;        % Vector pointing from p_s to p_{s+1}
            eP = rP/norm(rP);       % Unit vector pointing from p_s to p_{s+1}

            % Start plane perpendicular to path segment starting before start point at pI
            pI = p_s - dist*eP;     % Start offset point
            nI = -eP;               % unit normal vector pointin outside
            bI = -nI'*pI;
            % End plane perpendicular to path segment ending after end point at pE
            pE = p_s1 + dist*eP;    % End offset point
            nE = eP;
            bE = -nE'*pE;
            % Horizontal and vertical unit vectors for normal of side planes
            rHxy = -eP(2)/eP(1);    % x/y ratio of horizontal unit vector perpendicular to eP
            nH  = [-sqrt(rHxy^2/(rHxy^2+1));sqrt(1/(rHxy^2+1));0]; % Horizontal unit vector perpendicular to eP
            nU  = cross(eP,nH);     % Vertical unit vector perpendicular to eP and nH
            % Upper Plane
            pUs = p_s + dist*nU;    % Upper plane point above segment start point
            bU = -nU'*pUs;
            % Lower Plane
            pDs = p_s - dist*nU;    % Lower plane (D:Down) point below segment start point
            bD = nU'*pDs;
            % Left Plane
            pLs = p_s + dist*nH;
            bL = -nH'*pLs;
            % Right Plane
            pRs = p_s - dist*nH;
            bR = nH'*pRs;

            % Constraint Halfspace
            A = [nI';nE';nU';-nU';nH';-nH'];
            b = [-bI;-bE;-bU;-bD;-bL;-bR];

            Ap{s,1} = A;
            bp{s,1} = b;

            if withVisualization
                % Plane Equation: A*x + b == 0
                % Polyhedron (from MPT) P = { x | A*x <= b}
                Polyh{s,1} = Polyhedron('A',A,'b',b);

                assert(~Polyh{s,1}.isEmptySet,'Polyhedron has empty interior')
                assert(Polyh{s,1}.isBounded,'Polyhedron is unbounded')
            end
        end
    end


    %% YALMIP Variables
    Z = sdpvar((n+1)*nStates,N,'full');   % Trajectory parameters: z0,z1,...,z{N-1} (columns of Z for N stages/segm.)
                                          % z_i = [c_0^x,c_1^x,...,c_n^x, ...       (n-th order polynomials -> n+1 coeff.
                                          %        c_0^y,c_1^y,...,c_n^y, ...
                                          %        c_0^z,c_1^z,...,c_n^z, ...
                                          %        c_0^phi,c_1^phi,...,c_n^phi]
                                          % where [x,y,z,phi] are the flat outputs (# of flat outputs == nStates)
    T = sdpvar(nStates,N+1,'full');       % Trajcetory positions used as parameters

    % Data summary
    fprintf(1,'QP Problem has %d decision variables\n',(n+1)*nStates*N)
    fprintf(1,'\t# States:\t\t\t\t%d\n',nStates);
    fprintf(1,'\t# Path Segments:\t\t%d\n',N);
    fprintf(1,'\t# basis function order:\t%d\n',n);


    %% Construct QP
    % Cost function: sum of the integrals over the piece-wise path segments i with relative time allocation dTi
    % \sum_{i=1}^{N}{\int_{0}^{dTi}{|d^4/dt^4 Phi_i(t)|^2}dt
    %
    % n-th order polynomial basis function:
    % Phi_i(t)          := c0+c1*t+c2*t^2+c3*t^3+c4*t^4+c5*t^5+c6*t^6+ ... +cn*t^n
    % d^4/dt^4 Phi_i(t)  = 24*c4+120*c5*t+360*c6*t^2+ ... +n!/(n-4)!*t^(n-4)
    %
    % Phi4                  := d^4/dt^4 Phi_i(t)
    % p                     := [c0 c1 c2 c3 c4 c5 c6 ... cn]' polynomial coefficients of size [(n+1)x1]
    % (d^4/dt^4 Phi_i(t))^2 = 24*c4*Phi4+120*c5*t*Phi4+360*c6*t^2*Phi4+ ... +n!/(n-4)!*t^(n-4)*Phi4
    % 
    % Quadratic Cost Matrix Qi of a single polynomial to represent cost function in the form p'*Qi*p
    % where Qi is symmetric:
    % Qi  = [ zeros(4,4)       zeros(4,(n+1)-4); 
    %         zeros((n+1)-4,4) Qni]
    %
    % Qni = [ 24^2        24*120dt     24*360*dt^2  ... 24*n!/(n-4)!*dt^(n-4); 
    %         24*120t     (120*dt)^2   120*360*dt^3 ... 120*n!/(n-4)!*dt^(n-3); 
    %         24*360*dt^2 120*360*dt^3 (360dt^2)^2  ... 360*n!/(n-4)!*dt^(n-2); 
    %         ...         ...          ...          ... ...;
    %         24*n!/(n-4)!*dt^(n-4) ...]

    assert(n >= 4,'The Polynomial must be at least of order 4');

    % Loop over all path segments 
    constr = [];
    cost = 0;

    for s=1:N
        %% Cost and Constraint Matrices per segment s / Quadratic Cost 
        % The squared norm of a polynomial is equal the squared sum of the polynomial coefficients

        % tauQ is equal the factors for the polynomial coefficients for the 4th derivative
        % tauQ = [0 0 0 0 24 120 360 ... n!/(n-4)!]'*DeltaT
        tauQ = zeros(n+1,1);
        for i=4:n
            tauQ(i,1) = factorial(i)/factorial(i-4)*sqrt(2)/(2*(i-4)+1);
        end

        % Cost according to Richter et al
        Q4 = zeros(n+1);
        for i=1:n+1
            for j=1:n+1
                if (i >= 4) && (j >= 4)
                    % Calculate entry
                    pF = 1;
                    for l=0:3
                        pF = pF*(i-l)*(j-l);
                    end
                    Q4(i,j) = 2*pF*deltaT(s)^(i+j-2*4+1)/(i+j-2*4+1);
                end              
            end
        end
        Qni = Q4;

        %% Segment Start Point Constraint Matrix As and End Point Constraint Matrix Af
        As  = zeros(5,n+1);
        Af  = zeros(5,n+1);
        for j=0:4       % up to 4-th order shall be stationary
            % Start state constraint (note that this is the same for all segments)
            As(j+1,j+1) = factorial(j);

            % Final state constraint: d^(j)/dt^(j) Phi_i(dT_f) == 0  => Af*zf == df
            % d^(j)/dt^(j) Phi_i(dT_f^i)
            for k=j:n   % up to n-th order coefficient
                Af(j+1,k+1) = factorial(k)/factorial(k-j)  * deltaT(s)^(k-j+1);
            end
        end

        %% Inter-Segment Smoothness Constraint Matrix Ai and Ai_1
        % Smoothness up to snap (4-th order)
        % interstage constraint for intersection i 
        %   d^(j)/dt^(j) Phi_i(dT_i) == d^(j)/dt^(j) Phi_{i+1}(0)  => Ai*zi == Ai1*zi1  where j=0...4
        Ai      = zeros(4+1,n+1);   % for segment i at (relative) time dT_f^i
        Ai_1    = zeros(4+1,n+1);   % for segment i+1 at (relative) time 0
        for j=0:4       % up to 4-th order shall be smooth (snap)
            % d^(j)/dt^(j) Phi_i(dT_i)
            for k=j:n   % up to n-th order coefficient
                Ai(j+1,k+1) = factorial(k)/factorial(k-j) * deltaT(s)^(k-j+1); 
            end

            % d^(j)/dt^(j) Phi_{i+1}(0)     (note that this is the same for all segments)
            Ai_1(j+1,j+1) = factorial(j); 
        end

        %% Cost function (direct)
        % Block diagonalize for number of flat outputs n
        Qcell = cell(1,nStates);
        for i=1:nStates
            Qcell{1,i} = Qni;
        end
        Q = blkdiag(Qcell{:});

        % Adding cost for segment i
        cost = cost + Z(:,s)'*Q*Z(:,s);

        %% Setting the position and inter-segment constraints
        %% Initial Start Point Constraint
        % Assuming initial stationary point (up to 4-th order)
        if (s == 1) % Initial stationary starting point at p0
            for i=1:nStates
                idxS = (i-1)*(n+1) + 1;
                idxE = i*(n+1);
                % Apply position constraint for starting point of segment (t=0) up to 4-th order stationarity
                constrName = ['Initial Point Constraint ',num2str(i)];
                constr = [constr, (As*Z(idxS:idxE,s) == [T(i,s);zeros(4,1)] ):constrName]; %#ok<BDSCI,BDSCA,AGROW>
            end
        end

        %% Inter-Segment Smoothness Constraint
        %  start point of segment and smoothness to next segment up to 4-th order
        if (s >= 1) && (s < N) % intermediate path points with fixed position and smooth higher order derivatives
            % Apply interstage  smoothness constraint
            for i=1:nStates     % number of flat outputs (states)
                % Apply smoothness constraint at waypoint
                idxS = (i-1)*(n+1) + 1;
                idxE = i*(n+1);
                constr = [constr, (Ai*Z(idxS:idxE,s) == Ai_1*Z(idxS:idxE,s+1)):['Smoothness Constraint ',num2str(s)]]; %#ok<BDSCA,AGROW>
            end
        end

        %% End Segment Constraint
        if (s == N) % Final stationary ending point at pN
            % Values for final state constraint
            for i=1:nStates
                idxS = (i-1)*(n+1) + 1;
                idxE = i*(n+1);

                % Apply position constraint for end point of segment (t=dT) up to 4-th order
                constrName = ['Final EndPoint Constraint ',num2str(i)];
                constr = [constr, (Af*Z(idxS:idxE,s) == [T(i,s+1);zeros(4,1)]):constrName]; %#ok<BDSCI,BDSCA,AGROW>        
            end
        end

        %% Polyhedron constraints
        % WARNING:  The paper samples the time of each segment to check the polyhedron constraint => Thus there is
        %           no quarantee that the constraint is satisfied in continuous time. There might always be some
        %           edge cases.
        % From Polyhedrons:  P = { x | A*x <= b} where x = [pos_x; pox_y; pos_z] (Positional vector of trajectory)
        % x_i(t) = Phi_i(t) := c0+c1*t+c2*t^2+c3*t^3+c4*t^4+c5*t^5+c6*t^6+ ... +cn*t^n | c_i = [4x1]
        %        = tau(t)'*c_x          tau(t) = [1 t t^2 t^3 ... t^n]'*ones(1,3)
        %                               c_x    = [c0^{xyz} c1^{xyz} c2^{xyz} ... cn^{xyz}]'
        % => P = { x | A*tau(t)'*c_x and t \in [0,deltaT(i)]}
        if ~isempty(constrIdx)
            for t=linspace(0,deltaT(s),nSample) % number of samples in t to check constraint
                % construct tau(t) vector
                tau = zeros(n+1,1);
                for i=0:n
                    tau(i+1,1) = t^i;
                end
                TAU = blkdiag(tau',tau',tau');

                %  A   * z  <  b    | z = [tau',0,0;0,tau',0;0,0,tau']*[C_x;C_y;C_z]
                % [6x3]*[3x1] [6x1]       [3 x 3*(n+1)]                [3*(n+1)x1]
                idxS = 1;
                idxE = 3*(n+1); % First three states
                Aps = Ap{s};
                bps = bp{s};
                constrName = sprintf('PolyConstr Seg %d on state [x,y,z] at 0<=%4.2f<=%4.2f',s,t,deltaT(s));
                constr = [constr, (Aps(constrIdx,:)*TAU*Z(idxS:idxE,s) <= bps(constrIdx)):constrName]; %#ok<AGROW>
            end
        end

    end


    %% Generate Solver
    codeoptions = getOptions('TrajOptQuadrotor'); % solverName
    codeoptions.optlevel = 3;
    codeoptions.timing = 1;
    codeoptions.BuildSimulinkBlock = 0;

    controller = optimizerFORCES(constr, cost, codeoptions, T, Z, {'wayPoints'}, {'z_opt'});


    %% Solve
    [out_opt, ~, ~] = TrajOptQuadrotor({pathSegments});
    z_opt = out_opt;

    % Show constraint violation
    assign(T,pathSegments)
    assign(Z,z_opt);
    check(constr);


    %% Show Results
    % Extract data

    % Construct polynomials of piece-wise trajectory
    % position, velocity, acceleration, jerk, and snap
    nSamples    = 100;
    pos         = zeros(N*nSamples,nStates+1); % nStates + time: [t_vec;x_vec;y_vec;z_vec;phi_vec]
    vel         = zeros(N*nSamples,nStates+1);
    acc         = zeros(N*nSamples,nStates+1);
    jerk        = zeros(N*nSamples,nStates+1);
    snap        = zeros(N*nSamples,nStates+1);

    coefficients = zeros(N,nStates,(n+1));
    t0 = 0;

    for i=1:N
        % linear time vector for i-th piece
        ti = linspace(0,deltaT(i),nSamples);
        pos((i-1)*nSamples+1:i*nSamples,1) = ti + t0;
        vel((i-1)*nSamples+1:i*nSamples,1) = ti + t0;
        acc((i-1)*nSamples+1:i*nSamples,1) = ti + t0;
        jerk((i-1)*nSamples+1:i*nSamples,1) = ti + t0;
        snap((i-1)*nSamples+1:i*nSamples,1) = ti + t0;

        %% construct n-th order polynomial from solution
        % z_i = [c_0^x,c_1^x,...,c_n^x, ...       (n-th order polynomials -> n+1 coeff.
        %        c_0^y,c_1^y,...,c_n^y, ...
        %        c_0^z,c_1^z,...,c_n^z, ...
        %        c_0^phi,c_1^phi,...,c_n^phi]
        z_i  = z_opt((i-1)*(n+1)*nStates+1:i*(n+1)*nStates); % coefficients of this path piece i

        % Separate the coefficients for the different states
        for j=1:nStates
            % Extract poly. coeffs for state j in segment i
            coeffs              = z_i((j-1)*(n+1)+1:j*(n+1));

            % Create polynomial coefficients for derivatives of polynomial
            cD      = polyder(flip(coeffs));
            cDD     = polyder(cD);
            cDDD    = polyder(cDD);
            cDDDD   = polyder(cDDD);

            % Store coefficients of trajectory polynomial
            coefficients(i,j,:) = coeffs;   

            % Evaluate polyinomial at times ti (polyval expects c_n first and c_0 last -> flip)
            pos((i-1)*nSamples+1:i*nSamples,j+1) = polyval(flip(coeffs),ti);
            vel((i-1)*nSamples+1:i*nSamples,j+1) = polyval(cD,ti);
            acc((i-1)*nSamples+1:i*nSamples,j+1) = polyval(cDD,ti);
            jerk((i-1)*nSamples+1:i*nSamples,j+1) = polyval(cDDD,ti);
            snap((i-1)*nSamples+1:i*nSamples,j+1) = polyval(cDDDD,ti);
        end

        % next time point shall start at deltaT(i)
        t0 = t0+ deltaT(i);
    end

    % Plot Path point
    plotOffset = 2;
    minX = min(pathSegments(1,:))-plotOffset;
    minY = min(pathSegments(2,:))-plotOffset;
    minZ = min(pathSegments(3,:))-plotOffset;
    maxX = max(pathSegments(1,:))+plotOffset;
    maxY = max(pathSegments(2,:))+plotOffset;
    maxZ = max(pathSegments(3,:))+plotOffset;

    figure('Name','Path')
    plot3(pathSegments(1,:),pathSegments(2,:),pathSegments(3,:),'-o','Color','b','MarkerSize',10,'MarkerFaceColor','#D9FFFF','LineWidth',2)
    hold on, grid on, 
    axis([minX, maxX, minY, maxY, minZ, maxZ]);
    title('3D Path')
    xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');

    % axis equal
    plot3(pos(:,2),pos(:,3),pos(:,4),'Color','g','LineWidth',5); % i-th piece of solution
    % Projections on walls
    plot3(pos(:,2),pos(:,3),minZ*ones(size(pos(:,4))),'Color','k','LineWidth',2); % i-th piece on ground (XY)
    plot3(pos(:,2),maxY*ones(size(pos(:,3))),pos(:,4),'Color','k','LineWidth',2); % i-th piece on ZX
    plot3(maxX*ones(size(pos(:,2))),pos(:,3),pos(:,4),'Color','k','LineWidth',2); % i-th piece on ZY

    if withVisualization
        % Plot Polyhedron constraints
        for s=1:N
            % 3D Polyhedron
            plot(Polyh{s,1},'color','blue','alpha',0.1);

            % Projection on Coord-System planes and plotted as patches
            pP12 = Polyh{s,1}.projection([1,2]); pP12.computeHRep(); V12 = [pP12.V,minZ*ones(size(pP12.V(:,1)))];
            pP13 = Polyh{s,1}.projection([1,3]); pP13.computeHRep(); V13 = [pP13.V(:,1),maxY*ones(size(pP13.V(:,1))),pP13.V(:,2)];
            pP23 = Polyh{s,1}.projection([2,3]); pP12.computeHRep(); V23 = [maxX*ones(size(pP23.V(:,1))),pP23.V];
            pP12_3d = Polyhedron(V12);
            pP13_3d = Polyhedron(V13);
            pP23_3d = Polyhedron(V23);
            plot(pP12_3d,'color','blue','alpha',0.05);
            plot(pP13_3d,'color','blue','alpha',0.05);
            plot(pP23_3d,'color','blue','alpha',0.05);
        end
    end

    figure('Name','1D Signals')
    derivNames = {'Position ','Velocity ','Acceleration ','Jerk ','Snap '};
    stateNames = {'X','Y','Z','Phi'};
    unitNames  = {'Pos [m]','Pos [m]','Pos [m]','Angle [rad]';...
                  'Vel [m/s]','Vel [m/s]','Vel [m/s]','Ang. Vel [rad/s]';...
                  'Acc [m/s^2]','Acc [m/s^2]','Acc [m/s^2]','Ang. Acc [rad/s^2]';...
                  '[m/s^3]','[m/s^3]','[m/s^3]','[rad/s^3]';...
                  '[m/s^3]','[m/s^3]','[m/s^3]','[rad/s^4]'};
    for i=1:nStates
        subplot(5,nStates,i)
        plot(pos(:,1),pos(:,i+1))
        title([derivNames{1},stateNames{i}])
        xlabel('Time [s]');
        ylabel(unitNames{1,i});
        subplot(5,nStates,nStates + i)
        plot(vel(:,1),vel(:,i+1))
        title([derivNames{2},stateNames{i}])
        xlabel('Time [s]');
        ylabel(unitNames{2,i});
        subplot(5,nStates,2*nStates + i)
        plot(acc(:,1),acc(:,i+1))
        title([derivNames{3},stateNames{i}])
        xlabel('Time [s]');
        ylabel(unitNames{3,i});
        subplot(5,nStates,3*nStates + i)
        plot(jerk(:,1),jerk(:,i+1))
        title([derivNames{4},stateNames{i}])
        xlabel('Time [s]');
        ylabel(unitNames{4,i});
        subplot(5,nStates,4*nStates + i)
        plot(snap(:,1),snap(:,i+1))
        title([derivNames{5},stateNames{i}])
        xlabel('Time [s]');
        ylabel(unitNames{5,i});
    end
end
