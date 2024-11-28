function [Gx, Gmv, Gdmv] = RocketPlannerCostGradientFcn(stage,x,u,dmv) %#ok<INUSD>
% Rocket planner cost gradient function.

% Copyright 2020-2023 The MathWorks, Inc.

    Gx = zeros(6,1);
    Gmv = ones(2,1);
    Gdmv = zeros(2,1);
end
