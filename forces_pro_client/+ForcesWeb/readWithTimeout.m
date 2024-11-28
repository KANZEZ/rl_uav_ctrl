% Read content from the given url with specified timeout
%
% [DATA, TIMEDOUT] = readWithTimeout(URL, TIMEOUT) reads the content 
% from the given URL and returns it to the character array DATA. 
% If the operation did not finish before the default timeout 
% the operation returns true in TIMEDOUT
%
% [DATA, TIMEDOUT] = readWithTimeout(URL, TIMEOUT) reads the content 
% from the given URL and returns it to the character array DATA. 
% If the operation did not finish before the set TIMEOUT (in sec) 
% the operation returns true in TIMEDOUT
%
% DATA = readWithTimeout(_, SERVERCERTIFICATE) will use the selected
% SERVERCERTIFICATE to authenticate the server
%
% See also ForcesWeb read download write notFoundException ServerCertificates
%
%
% This file is part of the FORCESPRO client software for Matlab.
% (c) embotech AG, 2013-2023, Zurich, Switzerland. All rights reserved.
