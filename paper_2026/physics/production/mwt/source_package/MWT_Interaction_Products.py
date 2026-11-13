def MWT_Interaction_Products(reconstruction, target_window=1, interaction_group=0):
    """
    用已有尺度窗重构，构造一个目标窗、一组interaction的8种原场乘积。

    这是完整的独立def：仅依赖NumPy，不调用MWT或其他自定义函数，
    不重新重构、不修改输入、不打印、不读写文件。

    Parameters
    ----------
    reconstruction : dict或支持按键读取的缓存
        至少包含u、v、omega、t，全部为(3,N,p,lat,lon)形状的MWT重构场。
        第0维依次为低频、中间、高频窗。t必须为相对同一Tbar(p)的温度
        异常重构，omega为dp/dt，单位Pa/s。必须传重构，不能传变换系数；
        两者shape相同，本函数无法从数值自动辨别。
    target_window : int, default 1
        目标窗s，只能为0、1、2。
    interaction_group : int, default 0
        其余两窗记作a<b。对任一原场乘积p*q，四组定义为：
        0: p[a]*q[a] + p[a]*q[s] + p[s]*q[a]，标签a->s；
        1: p[b]*q[b] + p[b]*q[s] + p[s]*q[b]，标签b->s；
        2: p[a]*q[b] + p[b]*q[a]，标签a+b->s；
        3: p[s]*q[s]，标签s->s。
        平方项的交叉乘积因此自然出现2倍，不应额外删去重复项。

    Returns
    -------
    data : dict
        固定8个键：uu、uv、vv、omega_u、omega_v、u_t、v_t、omega_t。
        每项形状(N,p,lat,lon)。它们仍是原场乘积，尚未做MWT。
        一次仅生成一组的8项，避免同时暂存12组共96项原场。

    Notes
    -----
    随后由调用者显式执行MWT(data, ..., return_reconstruction=False)，
    再将该乘积系数缓存传给MWT_Interaction_Transfer。所有缓存必须
    使用同一输入时间网格、j0、j1和boundary；供能量计算的MWT系数
    必须采用normalization='physical'。本函数不存储、不推断这些设置；
    调用者需在保存缓存时记住target_window和interaction_group。
    可用np.savez保存返回字典的8个数组，不需要自定义对象或类型。

    示例（示例中的MWT及Transfer需由调用者分别粘贴或定义）：
        group_data = MWT_Interaction_Products(reconstruction, 1, 0)
        group_coefficients = MWT(group_data, j0=4, j1=7,
                                 boundary='symmetric', normalization='physical',
                                 return_reconstruction=False)
        Gamma_K, Gamma_A = MWT_Interaction_Transfer(
            coefficients, group_coefficients, basic_state, target_window=1)
    该示例要求reconstruction与coefficients原本也使用j0=4、j1=7、
    symmetric边界及同一段数据；不能仅在第二次MWT时更换参数。

    参考：Liang (2016), J. Atmos. Sci. 73, §7a，pp.4457-4458。
    """
    import numpy as np

    for name, value, maximum in (("target_window", target_window, 2),
                                 ("interaction_group", interaction_group, 3)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(name + "必须为整数。")
        if not 0 <= value <= maximum:
            raise ValueError(name + "超出允许范围。")
    target_window, interaction_group = int(target_window), int(interaction_group)

    def read_reconstruction(name):
        try:
            value = reconstruction[name]
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError("reconstruction缺少所需键：" + name) from exc
        if np.iscomplexobj(value):
            raise TypeError("重构场" + name + "必须为实数。")
        if np.ma.isMaskedArray(value):
            value = np.ma.asarray(value, dtype=float).filled(np.nan)
        array = np.asarray(value, dtype=float)
        if array.ndim != 5 or array.shape[0] != 3 or any(length == 0 for length in array.shape):
            raise ValueError("重构场" + name + "必须具有(3,N,p,lat,lon)非空形状。")
        if not np.all(np.isfinite(array)):
            raise ValueError("重构场" + name + "含NaN、Inf或缺测值。")
        return array

    fields = {name: read_reconstruction(name) for name in ("u", "v", "omega", "t")}
    shape = fields["u"].shape
    if any(value.shape != shape for value in fields.values()):
        raise ValueError("u、v、omega、t四个重构场必须具有相同形状。")
    first, second = [index for index in range(3) if index != target_window]

    def grouped_product(left, right):
        left, right = fields[left], fields[right]
        if interaction_group < 2:
            source = (first, second)[interaction_group]
            return (left[source] * right[source]
                    + left[source] * right[target_window]
                    + left[target_window] * right[source])
        if interaction_group == 2:
            return left[first] * right[second] + left[second] * right[first]
        return left[target_window] * right[target_window]

    pairs = {"uu": ("u", "u"), "uv": ("u", "v"), "vv": ("v", "v"),
             "omega_u": ("omega", "u"), "omega_v": ("omega", "v"),
             "u_t": ("u", "t"), "v_t": ("v", "t"), "omega_t": ("omega", "t")}
    return {key: grouped_product(left, right) for key, (left, right) in pairs.items()}
