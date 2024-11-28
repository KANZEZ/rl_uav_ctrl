% Download a file from a web service in the given url
%
% OUTFILENAME = download(FILENAME, URL) downloads the content from the 
% web service in the given URL and saves it to the given FILENAME. The 
% name of the saved file is returned to OUTFILENAME
%
% OUTFILENAME = download(_, SERVERCERTIFICATE) will use the selected
% SERVERCERTIFICATE to authenticate the server
%
% See also ForcesWeb read readWithTimeout write notFoundException ServerCertificates
%
%
% This file is part of the FORCESPRO client software for Matlab.
% (c) embotech AG, 2013-2023, Zurich, Switzerland. All rights reserved.
