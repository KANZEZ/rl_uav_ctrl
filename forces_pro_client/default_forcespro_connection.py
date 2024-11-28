"""
This file is part of the FORCESPRO client software for Python.
(c) embotech AG, 2013-2023, Zurich, Switzerland. All rights reserved.
"""

import forcespro
from forcespro.serverConnection import server_connections

def default_server_connection():
    """
    Check which connection to the server is the default

    Available connections:
        RestAPI:     for connections to the RestAPI interface
        WSDL:        for connections to the WSDL interface

    By default the function is set to return RestAPI

    (c) embotech AG, 2013-2023, Zurich, Switzerland. All rights reserved.
    """
    return server_connections.RestAPI

if __name__ == '__main__':
    print(str(default_server_connection()))
    import sys
    sys.exit(0)
