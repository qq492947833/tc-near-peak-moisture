def MWT_Interaction_Transfer(coefficients, product_coefficients, basic_state, target_window=1):
    """
    只用已保存的MWT系数，计算一组interaction对目标窗的KE和APE传递。

    这是完整的独立def：仅依赖NumPy，所有小工具均在函数内。它不调用
    MWT、不重算重构或乘积，不修改输入，不打印，不读写文件。

    Parameters
    ----------
    coefficients : dict或支持按键读取的缓存
        至少有u、v、t三个键，每项形状(3,N,p,lat,lon)。必须是原始风场
        及温度异常的MWT系数，normalization='physical'。t不是绝对温度。
    product_coefficients : dict或支持按键读取的缓存
        必须含uu、uv、vv、omega_u、omega_v、u_t、v_t、omega_t，各形状
        (3,N,p,lat,lon)。先由MWT_Interaction_Products构造指定目标窗和组
        的8种原场乘积，再由调用者显式调用一次MWT生成这些系数缓存。
        本函数只取其中target_window对应的系数，不再次变换。
    basic_state : dict或支持按键读取的缓存
        所需键全部为NumPy兼容数组：
        lon、lat：一维经纬格点中心，单位度；lon可跨日期变更线。
        pressure：一维气压，单位Pa，不在本函数中猜测或换算hPa。
        temperature_mean：一维参考温度Tbar(p)，单位K。
        c：一维正APE系数g/[Tbar*(g/cp-L)]。
        dc_dp：一维c对pressure(Pa)的导数。
        data_shape：整数一维数组[N, p的长度, lat的长度, lon的长度]。
        各空间坐标须严格单调、至少2点；不支持极点或重复0/360端点。
        使用MWT_Prepare返回的同一basic_state，避免参考态或网格混用。
    target_window : int, default 1
        目标窗，只能为0、1、2；必须与构造该乘积组时选用的目标窗相同。

    Returns
    -------
    Gamma_K, Gamma_A : ndarray
        各为(N,p,lat,lon)，单位m^2/s^3。正号表示该interaction项对目标
        窗供能；来源标签由调用者保存的interaction_group确定。
        四组分别为另一窗a->s、另一窗b->s、a+b->s和s->s，其中a<b。
        不能把混合组或同窗组直接当成一个指定供体窗的传递。

    Notes
    -----
    所有基础和乘积缓存必须使用同一段输入、时间/空间网格、j0、j1、
    boundary以及normalization='physical'。无元数据的数值数组不能
    自动揭示它是系数还是重构，也不能验证变换设置；调用者负责保持
    一致，并记录target_window与interaction_group。np.savez保存的数组
    字典可直接作为缓存使用；本函数不引入额外bundle或自定义类型。

    使用Liang (2016)压力坐标球面保守式(46)、(62)，与主能量诊断一致。
    KE保留水平基矢几何项，不加入高度坐标的omega/a项。APE只扣
    0.5*(dc/dp)*T_hat*(omega*T)_hat；R/(p*Tbar)修正属于表观源，
    不属于canonical transfer，因此这里不扣该项。
    差分为不等距内点中心、端点单边；经度不自动周期闭合。有限差分
    乘积法则与MWT有限基表均存在离散误差，不人工强制逐时两两反对称
    或机器精度零和。这里没有时间差分，因此无需dt。
    半径a=6.371e6 m，与原包一致；其余热力常数已经包含在缓存c中。

    示例：
        Gamma_K, Gamma_A = MWT_Interaction_Transfer(
            coefficients, product_coefficients, basic_state, target_window=1)
    调用前需确认product_coefficients确为目标窗1的某一组，而不是目标
    窗0/2的组，也不是8种完整乘积系数；后者会得到总传递而非分组传递。

    参考：Liang (2016), J. Atmos. Sci. 73，式(46)、(54)、(62)，§7a。
    """
    import numpy as np

    if isinstance(target_window, (bool, np.bool_)) or not isinstance(target_window, (int, np.integer)):
        raise TypeError("target_window必须为整数。")
    if not 0 <= target_window <= 2:
        raise ValueError("target_window只能为0、1或2。")
    target_window = int(target_window)

    def get_item(cache, key, label):
        try:
            return cache[key]
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError(label + "缺少所需键：" + key) from exc

    def real_array(value, label):
        if np.iscomplexobj(value):
            raise TypeError(label + "必须为实数。")
        if np.ma.isMaskedArray(value):
            value = np.ma.asarray(value, dtype=float).filled(np.nan)
        array = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError(label + "含NaN、Inf或缺测值。")
        return array

    dimensions = np.asarray(get_item(basic_state, "data_shape", "basic_state"))
    if (dimensions.shape != (4,) or not np.issubdtype(dimensions.dtype, np.integer)
            or np.any(dimensions <= 0)):
        raise ValueError("basic_state['data_shape']必须为四个正整数[N,np,nlat,nlon]。")
    shape = tuple(int(length) for length in dimensions)
    if any(length < 2 for length in shape[1:]):
        raise ValueError("压力、纬度和经度维度各至少需要2个格点。")

    def coordinate(key, length):
        value = real_array(get_item(basic_state, key, "basic_state"), key)
        if value.shape != (length,):
            raise ValueError(key + "必须为与data_shape一致的一维坐标。")
        return value

    lon = coordinate("lon", shape[-1])
    lat = coordinate("lat", shape[-2])
    pressure = coordinate("pressure", shape[-3])
    longitude = np.unwrap(np.deg2rad(lon))
    latitude = np.deg2rad(lat)
    for name, value in (("展开后的经度", longitude), ("纬度", latitude), ("气压", pressure)):
        difference = np.diff(value)
        if not (np.all(difference > 0) or np.all(difference < 0)):
            raise ValueError(name + "必须严格单调且无重复格点。")
    if np.ptp(longitude) >= 2 * np.pi - 1e-10:
        raise ValueError("经度范围不能含重复0/360度端点或绕行超过一周。")
    if np.any(np.abs(lat) >= 90):
        raise ValueError("球面经纬公式要求纬度严格位于(-90,90)度。")
    if np.any(pressure <= 0):
        raise ValueError("pressure必须为正气压，单位Pa。")

    def profile(key):
        value = real_array(get_item(basic_state, key, "basic_state"), key)
        if value.shape != (shape[-3],):
            raise ValueError(key + "必须为与pressure长度一致的一维剖面。")
        return value

    tbar, c, dc_dp = (profile(key) for key in ("temperature_mean", "c", "dc_dp"))
    if np.any(tbar <= 0):
        raise ValueError("temperature_mean必须为正的绝对温度K。")
    if np.any(c <= 0):
        raise ValueError("APE系数c必须为正，对应稳定基本态。")
    c, dc_dp = c[:, None, None], dc_dp[:, None, None]

    def read_coefficient(cache, key, label):
        array = real_array(get_item(cache, key, label), label + "[" + key + "]")
        if array.shape != (3,) + shape:
            raise ValueError(label + "[" + key + "]必须具有与data_shape一致的(3,N,p,lat,lon)形状。")
        return array[target_window]

    uc, vc, tc = (read_coefficient(coefficients, key, "coefficients") for key in ("u", "v", "t"))
    cosine = np.cos(latitude)[:, None]
    metric = np.tan(latitude)[:, None] / 6.371e6

    def derivative(array, values, axis):
        return np.gradient(array, values, axis=axis, edge_order=2 if len(values) >= 3 else 1)

    def dx(array):
        return derivative(array, longitude, -1) / (6.371e6 * cosine)

    def dy_div(array):
        return derivative(array * cosine, latitude, -2) / (6.371e6 * cosine)

    def dp(array):
        return derivative(array, pressure, -3)

    def product(key):
        return read_coefficient(product_coefficients, key, "product_coefficients")

    qx, qy, qp = (np.zeros_like(uc) for _ in range(3))
    adv_k = np.zeros_like(uc)
    f = product("uu")
    qx += 0.5 * uc * f
    adv_k -= uc * dx(f) + vc * metric * f
    f = product("uv")
    qx += 0.5 * vc * f
    qy += 0.5 * uc * f
    adv_k -= vc * dx(f) + uc * dy_div(f) - uc * metric * f
    f = product("vv")
    qy += 0.5 * vc * f
    adv_k -= vc * dy_div(f)
    f = product("omega_u")
    qp += 0.5 * uc * f
    adv_k -= uc * dp(f)
    f = product("omega_v")
    qp += 0.5 * vc * f
    adv_k -= vc * dp(f)
    div_qk = dx(qx) + dy_div(qy) + dp(qp)
    gamma_k = adv_k + div_qk

    f = product("u_t")
    qx = 0.5 * c * tc * f
    adv_a = -c * tc * dx(f)
    f = product("v_t")
    qy = 0.5 * c * tc * f
    adv_a -= c * tc * dy_div(f)
    f = product("omega_t")
    qp = 0.5 * c * tc * f
    adv_a -= c * tc * dp(f)
    div_qa = dx(qx) + dy_div(qy) + dp(qp)
    source_gradient = 0.5 * dc_dp * tc * f
    gamma_a = adv_a + div_qa - source_gradient
    return gamma_k, gamma_a
