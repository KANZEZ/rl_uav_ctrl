"""
This file is part of the FORCESPRO client software for Python.
(c) embotech AG, 2013-2023, Zurich, Switzerland. All rights reserved.
"""

import sys
import forcespro  # required to adjust PYTHONPATH
import forcespro.updateClient

if __name__ == "__main__":
    forcespro.updateClient.main(sys.argv)
    sys.exit(0)
