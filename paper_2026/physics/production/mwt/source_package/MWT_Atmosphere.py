def MWT_Atmosphere(coefficients, basic_state, dt=3600.0):
    """
    第三步：仅用已保存的MWT系数和基本态，返回大气能量诊断result。

    整个def可独立复制到新脚本或Notebook，只依赖NumPy。不导入/调用
    其他自定义代码，不调用MWT，不需要原始风温位势场或重构场，
    不读写文件、不打印、不修改输入。输入可来自前一步已保存的NPZ。

    coefficients：dict，各数组形状为(3,time,pressure,lat,lon)，必须是
    normalization='physical'的变换系数，且所有变量/乘积使用完全相同
    的时间网格、j0、j1、boundary、axis=0。它不是重构场字典。
    必要键：u,v,omega,t,uu,uv,vv,omega_u,omega_v,u_t,v_t,omega_t。
    若有phi系数，额外计算pressure_work及residual_K。乘积系数必须
    来自对相应原场乘积的MWT，不能用各变量系数相乘替代。

    basic_state：上一步返回的同名字典。必要键lon、lat为一维角度，
    pressure统一为Pa；temperature_mean、c、dc_dp均为一维压力剖面；
    data_shape为原场(time,pressure,lat,lon)四维形状。函数直接使用
    已计算的c/dc_dp/temperature_mean，不重新构造温度或位势异常。
    dt：等间隔时间步长，单位秒，默认3600；6小时资料须给21600。

    返回result：dict，各能量/收支数组形状均为(3,time,pressure,lat,lon)。
      KE、APE：比能量，m^2/s^2；KE仅含水平风。
      Gamma_K、Gamma_A：正则传递，正号为该项对目标窗供能。
      transport_K、transport_A：-div(Q)，正号为通量辐合供能。
      conversion_A_to_K：-R/p*T_hat*omega_hat，正号为APE转为KE。
      apparent_source_A：参考态相关两项之和，见Liang(2016)式(65)。
      dKE_dt、dAPE_dt：按dt求得的时间倾向。
      residual_A：dAPE_dt-transport_A-Gamma_A+conversion_A_to_K
                  -apparent_source_A。
    有phi时另返回pressure_work=-div(v_hat*Phi_hat)，及
      residual_K=dKE_dt-transport_K-Gamma_K-conversion_A_to_K-pressure_work。
    所有收支率单位m^2/s^3；残差包含未解析净源及数值闭合误差，不能
    直接称为潜热加热或摩擦。区域总量另用面积和正的层厚dp/g积分。

    可独立运行的用法：在新环境仅定义本函数，读取coefficients和
    basic_state两个字典，然后result=MWT_Atmosphere(coefficients,
    basic_state,dt=21600)。不需要定义MWT或MWT_Prepare。

    保持已验证版本的压力坐标球面保守式(46)、(62)、(69)-(70)，
    APE仅从Gamma扣dc/dp修正，R/(p*Tbar)项进入apparent_source_A。
    KE不加入高度坐标omega/a项。空间差分支持不等距、端点单边；
    经度微分不自动周期闭合。离散差分及MWT基表存在数值误差，
    不强制能量或传递零和闭合。随台风移动的区域积分另需移动边界项。
    常数沿用原版R=287.058、a=6.371e6。参考Liang(2016)，
    doi:10.1175/JAS-D-16-0131.1。
    """
    import numpy as np

    def real_array(value, name):
        if np.iscomplexobj(value):
            raise TypeError(name + "必须为实数。")
        if np.ma.isMaskedArray(value):
            value = np.ma.asarray(value, dtype=float).filled(np.nan)
        array = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError(name + "包含NaN、Inf或缺测值。")
        return array

    def coordinate(value, name, expected):
        array = real_array(value, name)
        if array.ndim != 1 or len(array) != expected or len(array) < 2:
            raise ValueError(name + "必须是匹配空间维、至少2点的一维坐标。")
        return array

    def monotonic(array, name):
        difference = np.diff(array)
        if not (np.all(difference > 0) or np.all(difference < 0)):
            raise ValueError(name + "必须严格单调且无重复格点。")

    def derivative(array, coordinates, axis):
        return np.gradient(array, coordinates, axis=axis,
                           edge_order=2 if len(coordinates) >= 3 else 1)

    if not isinstance(coefficients, dict) or not isinstance(basic_state, dict):
        raise TypeError("coefficients和basic_state必须是读取保存结果后恢复的字典。")
    required = ("u", "v", "omega", "t", "uu", "uv", "vv", "omega_u", "omega_v",
                "u_t", "v_t", "omega_t")
    missing = [key for key in required if key not in coefficients]
    if missing:
        raise ValueError("coefficients缺少必要变量：" + ", ".join(missing))
    needed_state = ("lon", "lat", "pressure", "temperature_mean", "c", "dc_dp", "data_shape")
    missing = [key for key in needed_state if key not in basic_state]
    if missing:
        raise ValueError("basic_state缺少：" + ", ".join(missing))
    dt_array = np.asarray(dt)
    if dt_array.ndim != 0 or np.iscomplexobj(dt_array):
        raise ValueError("dt必须为正的秒数标量。")
    dt = float(dt_array)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt必须为正的秒数。")

    values = {key: real_array(coefficients[key], key) for key in required}
    coefficient_shape = values["u"].shape
    if len(coefficient_shape) != 5 or coefficient_shape[0] != 3:
        raise ValueError("系数必须具有(3,time,pressure,lat,lon)五维形状。")
    if any(value.shape != coefficient_shape for value in values.values()):
        raise ValueError("所有变量及非线性乘积系数必须具有相同形状。")
    shape = coefficient_shape[1:]
    if shape[0] < 4 or shape[0] & (shape[0] - 1):
        raise ValueError("系数的时间长度必须是至少4的2次幂。")
    declared_shape = real_array(basic_state["data_shape"], "data_shape")
    if declared_shape.shape != (4,) or not np.array_equal(declared_shape, shape):
        raise ValueError("basic_state的data_shape与变换系数不一致。")
    lon = coordinate(basic_state["lon"], "lon", shape[-1])
    lat = coordinate(basic_state["lat"], "lat", shape[-2])
    p = coordinate(basic_state["pressure"], "pressure", shape[-3])
    lon_r, lat_r = np.unwrap(np.deg2rad(lon)), np.deg2rad(lat)
    monotonic(lon_r, "展开后的经度")
    monotonic(lat_r, "纬度")
    monotonic(p, "气压")
    if np.ptp(lon_r) >= 2*np.pi-1e-10 or np.any(np.abs(lat) >= 90):
        raise ValueError("不接受重复经度端点、超一周经度或极点。")
    if np.any(p <= 0):
        raise ValueError("basic_state中的pressure必须为正、单位Pa。")

    profiles = {}
    for key in ("temperature_mean", "c", "dc_dp"):
        profiles[key] = real_array(basic_state[key], key)
        if profiles[key].shape != (len(p),):
            raise ValueError(key + "必须是与pressure等长的一维剖面。")
    if np.any(profiles["temperature_mean"] <= 0) or np.any(profiles["c"] <= 0):
        raise ValueError("temperature_mean和c必须为正。")
    gas, radius = 287.058, 6.371e6
    c = profiles["c"][:, None, None]
    dc_dp = profiles["dc_dp"][:, None, None]
    tbar_3d = profiles["temperature_mean"][:, None, None]
    pressure_3d = p[:, None, None]
    cosine = np.cos(lat_r)[:, None]
    metric = np.tan(lat_r)[:, None] / radius

    def dx(array):
        return derivative(array, lon_r, -1) / (radius * cosine)

    def dy_div(array):
        return derivative(array * cosine, lat_r, -2) / (radius * cosine)

    def dp(array):
        return derivative(array, p, -3)

    uc, vc, wc, tc = (values[name] for name in ("u", "v", "omega", "t"))
    qx, qy, qp = (np.zeros_like(uc) for _ in range(3))
    adv_k = np.zeros_like(uc)
    f = values["uu"]
    qx += 0.5 * uc * f
    adv_k -= uc * dx(f) + vc * metric * f
    f = values["uv"]
    qx += 0.5 * vc * f
    qy += 0.5 * uc * f
    adv_k -= vc * dx(f) + uc * dy_div(f) - uc * metric * f
    f = values["vv"]
    qy += 0.5 * vc * f
    adv_k -= vc * dy_div(f)
    f = values["omega_u"]
    qp += 0.5 * uc * f
    adv_k -= uc * dp(f)
    f = values["omega_v"]
    qp += 0.5 * vc * f
    adv_k -= vc * dp(f)
    div_qk = dx(qx) + dy_div(qy) + dp(qp)
    gamma_k = adv_k + div_qk

    f = values["u_t"]
    qx = 0.5 * c * tc * f
    adv_a = -c * tc * dx(f)
    f = values["v_t"]
    qy = 0.5 * c * tc * f
    adv_a -= c * tc * dy_div(f)
    f = values["omega_t"]
    qp = 0.5 * c * tc * f
    adv_a -= c * tc * dp(f)
    div_qa = dx(qx) + dy_div(qy) + dp(qp)
    source_gradient = 0.5 * dc_dp * tc * f
    source_thermo = gas / (pressure_3d * tbar_3d) * tc * f
    gamma_a = adv_a + div_qa - source_gradient
    transport_k, transport_a = -div_qk, -div_qa
    apparent_a = source_gradient + source_thermo

    ke = 0.5 * (uc * uc + vc * vc)
    ape = 0.5 * c * tc * tc
    conversion = -gas / pressure_3d * tc * wc
    time = np.arange(shape[0], dtype=float) * dt
    dke = derivative(ke, time, 1)
    dape = derivative(ape, time, 1)
    result = {"KE": ke, "APE": ape,
              "Gamma_K": gamma_k, "Gamma_A": gamma_a,
              "transport_K": transport_k, "transport_A": transport_a,
              "conversion_A_to_K": conversion, "apparent_source_A": apparent_a,
              "dKE_dt": dke, "dAPE_dt": dape,
              "residual_A": dape - transport_a - gamma_a + conversion - apparent_a}
    if "phi" in coefficients:
        phic = real_array(coefficients["phi"], "phi")
        if phic.shape != coefficient_shape:
            raise ValueError("phi系数必须与其他系数具有相同形状。")
        pressure_work = -(dx(uc * phic) + dy_div(vc * phic) + dp(wc * phic))
        result["pressure_work"] = pressure_work
        result["residual_K"] = dke - transport_k - gamma_k - conversion - pressure_work
    return result
