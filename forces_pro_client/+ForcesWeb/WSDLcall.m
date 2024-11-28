% Performs a WSDL call to the specified FORCESPRO server
%
% RESULT = WSDLcall(SERVER, SERVER_CALL) performs the provided SERVER_CALL
% to the SERVER and returns the RESULT. SERVER is a WSDL object or a url. 
% SERVER_CALL is a lamda function with a single parameter which will be
% the WSDL object.
%
% RESULT = getServerVersion(_, LEGACYVERSION) will use the 
% legacy WSDL communication if LEGACYVERSION is specified.
%
% See also ForcesWeb initializeWSDL finalizeWSDL ServerConnections
%
%
% This file is part of the FORCESPRO client software for Matlab.
% (c) embotech AG, 2013-2023, Zurich, Switzerland. All rights reserved.
