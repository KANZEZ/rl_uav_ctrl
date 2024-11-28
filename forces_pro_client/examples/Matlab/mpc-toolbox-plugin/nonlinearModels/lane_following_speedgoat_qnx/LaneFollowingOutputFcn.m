function [ y ] = LaneFollowingOutputFcn(x,u) %#ok<INUSD>
% Copyright 2020-2023 The MathWorks, Inc. and Embotech AG, Zurich, Switzerland

    y = [x(3);...
         x(5);...
         x(6)+x(7)];

end
