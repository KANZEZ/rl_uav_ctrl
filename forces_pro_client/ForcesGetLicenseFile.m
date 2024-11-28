% Receive license file from the FORCESPRO server to validate the 
% generated FORCESPRO solvers
%
%    FORCESGETLICENSEFILE() receives a license file from the default
%    FORCESPRO server and saves it to FORCESPRO.license
%
%    FORCESGETLICENSEFILE(LICENSE_FILE_NAME) receives a license file 
%    from the default FORCESPRO server and saves it to the selected 
%    filename
%
%    FORCESGETLICENSEFILE(LICENSE_FILE_NAME, FORCESURL) receives a 
%    license file from the selected FORCESPRO server and saves it 
%    to the selected filename
%
%    FORCESGETLICENSEFILE(LICENSE_FILE_NAME, FORCESURL, SERVERCONNECTION) 
%    receives a license file from the selected FORCESPRO server with the 
%    specified SERVERCONNECTION and saves it to the selected filename
%
%
%    FORCESGETLICENSEFILE(LICENSE_FILE_NAME, FORCESURL, SERVERCONNECTION, USERID) 
%    receives a license file for the selected userid from the selected 
%    FORCESPRO server with the specified SERVERCONNECTION and saves it to 
%    the selected filename. 
%
%    FORCESGETLICENSEFILE(LICENSE_FILE_NAME, FORCESURL, SERVERCONNECTION, USERID, DATABASE)  
%    receives a license file for the selected userid from the selected 
%    FORCESPRO server with the specified SERVERCONNECTION and saves it to 
%    the selected filename. 
%    The user performing the request will be checked in the selected 
%    database. Available values for database are 'default', 'portal', 
%    'old'. If not sure, use 'default'.
%
% This file is part of the FORCESPRO client software for Matlab.
% (c) embotech AG, 2013-2023, Zurich, Switzerland. All rights reserved.
