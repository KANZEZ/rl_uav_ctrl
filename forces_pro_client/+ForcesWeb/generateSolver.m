% Creates a solver generation request on the FORCESPRO server
%
% STATUSFILE = generateSolver(SERVER, USERID, MATFILE_BYTESTREAM_AS_STRING, DATABASE) 
% creates a solver generation request on the specified server. SERVER is 
% a url for a RestAPI connection and a WSDL object for a WSDL 
% connection. The solver is generated for the USERID user in the specified
% DATABASE for the problem specified in MATFILE_BYTESTREAM_AS_STRING. 
% The request returns to STATUSFILE a link to the status file for the 
% solver generation
%
% STATUSFILE = generateSolver(_, SERVERCONNECTION) will use the 
% server communication specified in SERVERCONNECTION.
%
% See also ForcesWeb initializeWSDL ServerConnections ServerResponse
%
%
% This file is part of the FORCESPRO client software for Matlab.
% (c) embotech AG, 2013-2023, Zurich, Switzerland. All rights reserved.
