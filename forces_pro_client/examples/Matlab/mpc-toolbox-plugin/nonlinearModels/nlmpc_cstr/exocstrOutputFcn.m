function [ y ] = exocstrOutputFcn(x,u) %#ok<INUSD>
% Output is the concentration of A in the exit stream plus the unmeasured
% disturbance
%
% The states of the CSTR model are:
%
%   x(1) = T        Reactor temperature [K]
%   x(2) = CA       Concentration of A in reactor tank [kgmol/m^3]
%   x(3) = Dist     State of unmeasured output disturbance

    y = x(2);

end
