def MWT_levels(n_time, long_period, short_period, dt=1.0):
    """
    按原包手册把标称周期换成(j0, j1, j2)，不自动取整。

    n_time必须为2次幂；long_period、short_period和dt使用相同时间单位。
    P_j=2*n_time*dt/2**j。要求周期恰好对应整数层级，且三个窗均非空。
    例如 MWT_levels(512, 64, 8, dt=1) 返回(4, 7, 9)。
    若想使用非二进制周期，应明确选择相邻尺度并做敏感性检验。
    """
    import numpy as np

    if isinstance(n_time, (bool, np.bool_)) or not isinstance(n_time, (int, np.integer)):
        raise TypeError("n_time必须为整数。")
    n_time = int(n_time)
    if n_time < 4 or n_time & (n_time - 1):
        raise ValueError("n_time必须为至少4的2次幂。")
    periods = np.array([long_period, short_period, dt], dtype=float)
    if not np.all(np.isfinite(periods)) or np.any(periods <= 0):
        raise ValueError("周期及dt必须是有限正数，并使用相同时间单位。")
    if long_period <= short_period:
        raise ValueError("long_period必须大于short_period。")
    levels = np.log2(2.0 * n_time * dt / periods[:2])
    rounded = np.rint(levels)
    if not np.allclose(levels, rounded, atol=1e-10, rtol=0):
        raise ValueError("指定周期不对应整数尺度层级；请明确选择相邻二进制周期。")
    j0, j1 = (int(x) for x in rounded)
    j2 = n_time.bit_length() - 1
    if not 0 <= j0 < j1 < j2:
        raise ValueError("指定周期不能形成三个非空尺度窗；需要0 <= j0 < j1 < j2。")
    return j0, j1, j2
