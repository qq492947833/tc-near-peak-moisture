"""Conditionally import numba based on the TCPYPI_DISABLE_NUMBA environment variable.

Set the TCPYPI_DISABLE_NUMBA environment variable to 1 to disable numba.

This is useful for coverage testing.
"""

import os
from functools import wraps

import numpy as np


def noop_njit(*args, **kwargs):
    """No-op decorator to replace @njit when numba is disabled.

    Note that this decorator isn't entirely a noop, because numba does some casting
    of NumPy types to Python types. Since NumPy v2, NumPy types like np.float64 are
    printed explicitly, so the doctests would fail without casting scalars back
    to Python types.

    See <https://github.com/numba/numba/issues/704> for more details about Numba's
    behavior.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, np.floating):
                result = float(result)
            elif isinstance(result, tuple):
                # Cast components to float
                result = tuple(
                    float(x) if isinstance(x, np.floating) else x for x in result
                )
            return result

        return wrapper

    # Handle both @njit and @njit() syntax
    if len(args) == 1 and callable(args[0]):
        return decorator(args[0])
    return decorator


if os.getenv("TCPYPI_DISABLE_NUMBA") == "1":
    njit = noop_njit
else:
    import numba as nb

    njit = nb.njit
