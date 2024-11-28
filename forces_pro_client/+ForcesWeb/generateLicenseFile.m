% Creates a license file generation request on the FORCESPRO server
%
% LICENSEFILE = generateLicenseFile(SERVER, USERID, LICENSE_FILE_NAME, DATABASE) 
% creates a license file generation request on the specified server. 
% SERVER is a url  for a RestAPI connection and a WSDL object for 
% a WSDL connection. The license file with name LICENSE_FILE_NAME will
% be geenrated for the USERID user in the specified DATABASE. 
% The request returns to LICENSEFILE a link to the license file for the 
% solver generation
%
% LICENSEFILE = generateLicenseFile(_, SERVERCONNECTION) will use the 
% server communication specified in SERVERCONNECTION.
%
% See also ForcesWeb initializeWSDL ServerConnections ServerResponse
%
%
% This file is part of the FORCESPRO client software for Matlab.
% (c) embotech AG, 2013-2023, Zurich, Switzerland. All rights reserved.
