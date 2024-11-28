function [ xnext ] = dynamics( X )

    x = X(1:4);
    u = X(5:6);
    
    % implements a RK4 integrator for the dynamics.
    h = 0.1; % step size
    xnext = RK4( x,u,@my_continuous_dynamics,h );
    
end


function [ xdot ] = my_continuous_dynamics( X,U )

    % Dynamics of system
    m = 1;
    I = 1;
    
    %x = X(1);
    %y = X(2);
    v = X(3);
    theta = X(4);
    F = U(1);
    s = U(2);
    
    xdot =  [v*cos(theta);
            v*sin(theta);
            F/m;
            s/I];

end
