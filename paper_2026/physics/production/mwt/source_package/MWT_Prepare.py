def MWT_Prepare(u, v, omega, temperature, lon, lat, pressure,
                geopotential=None, temperature_mean=None, lapse_rate=None,
                pressure_unit="Pa", geopotential_is_anomaly=False):
    """
    第一步：准备MWT所需原场、非线性乘积和基本态，返回data, basic_state。

    整个def可独立复制使用，只依赖NumPy；不调用MWT或其他自定义函数，
    不读写文件、不打印、不修改输入。内部小工具均嵌套在本函数中。

    输入
    ----
    u/v/omega/temperature：相同(time, pressure, lat, lon)四维实数数组，
    无缺测。u/v为m/s；omega=dp/dt为Pa/s，上升为负；temperature为
    绝对温度K。时间等间隔，长度须为至少4的2次幂，供后续MWT使用。
    lon/lat：一维格点中心坐标，单位度；pressure：一维正气压。
    pressure_unit='Pa'或'hPa'，但omega始终为Pa/s。空间维均至少2点，
    坐标可严格升序或降序，经度允许跨日期线。不接受极点或重复经度端点。

    temperature_mean：Tbar(p)，K，标量或一维压力剖面；默认取温度的
    时间及水平面积平均。lapse_rate：同一基本态的L=-dTbar/dz，K/m；
    默认按静力关系L=g*p/(R*Tbar)*dTbar/dp计算。仅接受稳定参考态。

    geopotential：可选的全位势，m^2/s^2。默认从同一Tbar积分出Phibar，
    用最大气压层的面积/时间均值定标，再取Phi=Phi*-Phibar。若已提供
    相对同一基本态的位势异常，设geopotential_is_anomaly=True。

    返回
    ----
    data：dict，所有值为(time, pressure, lat, lon)数组。键为
      u, v, omega, t, uu, uv, vv, omega_u, omega_v, u_t, v_t, omega_t；
      若输入位势，另有phi。t=T*-Tbar；八个乘积均由原场形成，
      不是变换系数之间的乘积。
    basic_state：dict，各值为NumPy数组，包含
      lon, lat：度；pressure：统一为Pa；
      temperature_mean, c, dc_dp：一维压力剖面；
      data_shape：原场四维形状，整数数组。
      c=g/[Tbar*(g/cp-L)]；后续能量诊断直接使用这些已算好的基本态。

    接续方式
    --------
    data, basic_state = MWT_Prepare(...)
    coefficients, reconstruction = MWT(data, j0=..., j1=...)
    result = MWT_Atmosphere(coefficients, basic_state, dt=...)
    上述MWT和MWT_Atmosphere由用户分步调用，本函数不会调用它们。
    data和basic_state均可用np.savez(..., **变量名)保存，随后用
    np.load(..., allow_pickle=False)读取并恢复为同名字典。

    应在固定地理网格上准备/变换/诊断，再按台风位置取样；地下压力层
    和其他无效数据须事先处理。常数沿用已验证版本：R=287.058，
    cp=1005，g=9.8。科学公式保持Liang(2016)压力坐标干空气框架。
    """
    import numpy as np

    def real_array(value, name):
        if np.iscomplexobj(value):
            raise TypeError(name + "必须为实数。")
        if np.ma.isMaskedArray(value):
            value = np.ma.asarray(value, dtype=float).filled(np.nan)
        array = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError(name + "包含NaN、Inf或缺测值；不自动插补。")
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

    def profile(value, name, length):
        array = real_array(value, name)
        if array.ndim == 0:
            return np.full(length, float(array))
        if array.shape != (length,):
            raise ValueError(name + "必须是标量或与pressure等长的一维剖面。")
        return array

    def cell_edges(centers):
        edges = np.empty(len(centers) + 1)
        edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
        edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
        edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
        return edges

    def derivative(array, coordinates, axis):
        return np.gradient(array, coordinates, axis=axis,
                           edge_order=2 if len(coordinates) >= 3 else 1)

    gas, cp, gravity = 287.058, 1005.0, 9.8
    fields = {}
    for name, value in (("u", u), ("v", v), ("omega", omega), ("temperature", temperature)):
        fields[name] = real_array(value, name)
    shape = fields["u"].shape
    if len(shape) != 4 or any(value.shape != shape for value in fields.values()):
        raise ValueError("四个变量必须具有相同的(time, pressure, lat, lon)四维形状。")
    if shape[0] < 4 or shape[0] & (shape[0] - 1):
        raise ValueError("时间长度必须是至少4的2次幂。")
    if np.any(fields["temperature"] <= 0):
        raise ValueError("temperature必须为正的绝对温度K；请勿输入温度距平。")
    lon = coordinate(lon, "lon", shape[-1])
    lat = coordinate(lat, "lat", shape[-2])
    p = coordinate(pressure, "pressure", shape[-3])
    lon_r = np.unwrap(np.deg2rad(lon))
    lat_r = np.deg2rad(lat)
    monotonic(lon_r, "展开后的经度")
    monotonic(lat_r, "纬度")
    monotonic(p, "气压")
    if np.ptp(lon_r) >= 2 * np.pi - 1e-10:
        raise ValueError("经度范围不能含重复的0/360度端点或绕行超过一周。")
    if np.any(np.abs(lat) >= 90):
        raise ValueError("纬度必须严格位于(-90,90)度。")
    if pressure_unit == "hPa":
        p = p * 100.0
    elif pressure_unit != "Pa":
        raise ValueError("pressure_unit只能是'Pa'或'hPa'。")
    if np.any(p <= 0):
        raise ValueError("气压必须为正。")
    if not isinstance(geopotential_is_anomaly, (bool, np.bool_)):
        raise TypeError("geopotential_is_anomaly必须为布尔值。")

    lon_edges = cell_edges(lon_r)
    lat_edges = np.clip(cell_edges(lat_r), -np.pi / 2, np.pi / 2)
    area_weights = (np.abs(np.diff(np.sin(lat_edges)))[:, None]
                    * np.abs(np.diff(lon_edges))[None, :])

    def horizontal_time_mean(array):
        return np.sum(array.mean(axis=0) * area_weights, axis=(-2, -1)) / area_weights.sum()

    if temperature_mean is None:
        tbar = horizontal_time_mean(fields["temperature"])
    else:
        tbar = profile(temperature_mean, "temperature_mean", len(p))
    if np.any(tbar <= 0):
        raise ValueError("temperature_mean必须为正的绝对温度K。")
    if lapse_rate is None:
        lapse = gravity * p / (gas * tbar) * derivative(tbar, p, 0)
    else:
        lapse = profile(lapse_rate, "lapse_rate", len(p))
    stability = gravity / cp - lapse
    if np.any(stability <= 0):
        raise ValueError("APE要求静力稳定参考态：每层均须g/cp-lapse_rate>0。")
    c = (gravity / (tbar * stability))[:, None, None]
    dc_dp = derivative(c, p, 0)
    fields["t"] = fields.pop("temperature") - tbar[:, None, None]

    data = dict(fields)
    pairs = {"uu": ("u", "u"), "uv": ("u", "v"), "vv": ("v", "v"),
             "omega_u": ("omega", "u"), "omega_v": ("omega", "v"),
             "u_t": ("u", "t"), "v_t": ("v", "t"), "omega_t": ("omega", "t")}
    for name, (left, right) in pairs.items():
        data[name] = fields[left] * fields[right]

    if geopotential is not None:
        phi = real_array(geopotential, "geopotential")
        if phi.shape != shape:
            raise ValueError("geopotential必须与风温变量具有相同四维形状。")
        if not geopotential_is_anomaly:
            phi_reference = np.empty(len(p), dtype=float)
            phi_reference[0] = 0.0
            phi_reference[1:] = -gas * np.cumsum(0.5 * (tbar[:-1] + tbar[1:]) * np.diff(np.log(p)))
            anchor = int(np.argmax(p))
            phi_reference += horizontal_time_mean(phi)[anchor] - phi_reference[anchor]
            phi = phi - phi_reference[:, None, None]
        data["phi"] = phi

    basic_state = {"lon": lon.copy(), "lat": lat.copy(), "pressure": p.copy(),
                   "temperature_mean": tbar.copy(), "c": c[:, 0, 0].copy(),
                   "dc_dp": dc_dp[:, 0, 0].copy(),
                   "data_shape": np.asarray(shape, dtype=np.int64)}
    return data, basic_state
