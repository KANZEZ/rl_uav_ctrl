function [codegen, storage, secondary_storage] = SSLCertificateFiles()
%    Get paths to certificate files for FORCESPRO servers authentication
%
%    This file can be edited to return the paths for the certificate files for the
%        codegen
%        storage
%        secondary_storage (optional) 
%    servers used for FORCESPRO
%
% See also ForcesWeb ServerCertificates getCertificatePath
%
% This file is part of the FORCESPRO client software for Matlab.
% (c) embotech AG, 2013-2023, Zurich, Switzerland. All rights reserved.
    % path to certificates file that contains client certificate to authenticate for codegen server. To use default set to empty
    codegen = '';
    % path to certificates file that contains client certificate to authenticate for storage server. To use default set to empty
    storage = '';
    % path to certificates file that contains client certificate to authenticate for secondary storage server. To use default set to empty
    secondary_storage = '';
end
