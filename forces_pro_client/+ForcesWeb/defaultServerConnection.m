function serverConnection = defaultServerConnection()
% Check which connection to the server is the default
%
%    SERVERCONNECTION = defaultServerConnection() will return:
%        RestAPI:     for connections to the RestAPI interface
%        WSDL:        for connections to the WSDL interface
%        WSDL_legacy: for connections to the WSDL interface using
%                       legacy MATLAB functionality
%
%    By default the function is set to return RestAPI
%
% See also ForcesWeb ServerConnections
%
%
% This file is part of the FORCESPRO client software for Matlab.
% (c) embotech AG, 2013-2023, Zurich, Switzerland. All rights reserved.
    % use WSDL legacy for MATLAB R2014b or earlier
    if verLessThan('matlab', '8.5')
        serverConnection = ForcesWeb.ServerConnections.WSDL_legacy;
    else
        serverConnection = ForcesWeb.ServerConnections.RestAPI;
    end
end
