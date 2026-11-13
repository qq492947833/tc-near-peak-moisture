def MWT(data, j0=3, j1=5, axis=0, boundary="symmetric",
        normalization="physical", return_reconstruction=True,
        block_size=256, nan_policy="raise", reconstruction_keys=None):
    """
    多尺度窗口变换：把变量分为低频、中间尺度、高频三个窗口。

    这是所附 mseva_atmos 中 proc_pt1.f / Jp.f / Decompose.f / Syn.f 的
    NumPy 移植，保留原三次样条基表、半步积分、有效半宽12，以及2025年
    Syn.f中的对称延拓修正。没有用FFT或另一套小波滤波器替代原算法。

    Parameters
    ----------
    data : array_like or dict[str, array_like]
        实数数组；axis指定时间维，其余维可以是站点或空间网格。时间必须
        等间隔，长度N必须为2的整数次幂。函数不补零、不插值、不截断。
        字典输入必须非空且键均为字符串；值是已准备好的原场或乘积场。
        各值分别应用相同的尺度、axis、边界和归一化选项，不自动构造乘积，
        不把不同变量stack成大数组。返回字典保留输入键的顺序。
    j0, j1 : int, default 3, 5
        两个尺度层级，要求0 <= j0 < j1 < log2(N)。数字越大，尺度越短。
        原包教程的标称周期为 P_j=2*N*dt/2**j；不是理想傅里叶硬截止。
        例：512个逐日样本、64天与8天边界，取j0=4、j1=7。
        也可使用另附的独立 MWT_levels 函数换算；本函数不调用它。
    axis : int, default 0
        输入的时间轴，支持负索引。
    boundary : "symmetric" or "periodic", default "symmetric"
        同时接受原Fortran编号2或1。原核心不支持编号0或3。
        对称延拓仍有端点影响；应使用事件前后充分长的连续时间段。
    normalization : "physical" or "original", default "physical"
        physical返回sqrt(N)*原变换系数，能量直接用0.5*coeff**2。
        original返回原Jp积分归一化的系数，能量必须乘N。
        两种设置不影响重构场。
    return_reconstruction : bool, default True
        True返回(coefficients, reconstruction)；False只返回coefficients。
        字典输入返回对应的字典；False时禁止指定reconstruction_keys。
    block_size : int, default 256
        同时处理的空间序列数；只限制中间工作数组，最终输出仍占内存。
    nan_policy : "raise" or "propagate", default "raise"
        raise拒绝NaN；propagate将含NaN的整条时间序列返回为NaN，不填补。
        Inf始终报错。缺测不能直接当作0降水、0风速等物理观测。
    reconstruction_keys : tuple[str, ...], list[str], or None, default None
        仅对字典输入有效。None保留全部重构；tuple/list只保留指定键的
        重构，系数仍返回所有键。空tuple/list返回空重构字典。键必须存在、
        不重复且均为字符串。重构字典按输入键顺序排列。
        未选中的变量不分配完整重构返回数组，仍需计算变换所需的分块
        重构中间量。block_size不能减少最终系数字典的存储量。

    Returns
    -------
    coefficients : ndarray or dict[str, ndarray]
        数组输入时shape=(3,)+data.shape；第0维依次为低频、中间尺度、
        高频窗口。字典输入按原键返回相同规则的系数数组。
        能量诊断使用此变换系数。
    reconstruction : ndarray or dict[str, ndarray], optional
        数组输入时与coefficients相同shape；字典只保留所选键。每个返回
        重构的三个窗口之和等于对应输入。绘制尺度分量、构造
        interaction products时使用它，不要把它当作变换系数计算能量。

    Notes
    -----
    只依赖NumPy及Python标准库operator；调用不读写磁盘、不打印。
    输入数组及输入字典不修改。所有辅助函数和原基表都嵌入此def，
    将此def完整复制到空白命名空间即可调用；无需其他本地函数、
    编译器或外部.dat文件。

    理论的能量正交性和正则传递零和涉及对采样位置n的边缘化，不是每个
    原始时次逐点的平方恒等式。原程序有限积分、基函数截断及端点处理
    会引入离散误差，最高频尤其明显；重构闭合不等于能量闭合。
    不应为了使误差为0而重新归一化各窗口或人工平衡传递项。

    Examples
    --------
    >>> coefficients, reconstruction = MWT(u, j0=4, j1=7, axis=0)
    >>> u_background, u_middle, u_fast = reconstruction
    >>> u_middle_coeff = coefficients[1]

    References
    ----------
    Liang & Anderson (2007), Multiscale Window Transform.
    Liang (2016), J. Atmos. Sci. 73, 4439-4468,
    doi:10.1175/JAS-D-16-0131.1.
    Original algorithm and basis tables: X. San Liang, supplied mseva_atmos.
    Port prepared 2026-10-22. This attribution is not a new upstream license.
    """

    def _mwt_phi(x, basis):
        """与原phi1_shape相同的表格线性插值；有效参数范围由调用者控制。"""
        import numpy as np

        phi, _, step = basis
        xx = np.asarray(x, dtype=np.float64) / step
        k = np.floor(xx).astype(np.int64)
        return phi[k + 450] + (phi[k + 451] - phi[k + 450]) * (xx - k)

    def _mwt_analysis(z, boundary, basis):
        """Jp.f: 最细层半步积分，并保留原代码特殊的系数延拓。"""
        import numpy as np

        n = z.shape[0]
        out = np.zeros_like(z)
        positions = np.arange(n)
        for shift in range(-12, 13):
            left = positions + shift
            right = left + 1
            if boundary == 1:
                left, right = left % n, right % n
            else:
                left, right = left % (2*n), right % (2*n)
                left = np.where(left < n, left, 2*n - 1 - left)
                right = np.where(right < n, right, 2*n - 1 - right)
            out += _mwt_phi(shift, basis) * z[left]
            if shift < 12:
                out += _mwt_phi(shift + 0.5, basis) * (0.5*z[left] + 0.5*z[right])
        out /= (2.0*np.sqrt(n))
        if boundary == 1:
            return np.concatenate((out, out), axis=0)
        # 原Jp.f明确指定a(N)=a(N-1), a(N+1)=a(N-1), ... a(2N-1)=a(1)。
        # 此处不能替换成通用np.pad(..., mode="symmetric")。
        return np.concatenate((out, out[-1:], out[:0:-1]), axis=0)

    def _mwt_lowpass(z, level, boundary, basis):
        """Decomp_ajj + 2025 Syn: 输入(time,series)，返回指定层级的重构。"""
        import numpy as np

        n = z.shape[0]
        _, h0, _ = basis
        a = _mwt_analysis(z, boundary, basis)
        jmax = n.bit_length() - 1
        for j in range(jmax - 1, level - 1, -1):
            target = np.arange(2 * 2**j)
            coarse = np.zeros((target.size, z.shape[1]), dtype=np.float64)
            for shift in range(-12, 13):
                coarse += h0[shift + 15] * a[(2*target + shift) % a.shape[0]]
            a = coarse
        x = np.arange(n) * (2.0**level / n)
        center = np.floor(x).astype(np.int64)
        out = np.zeros_like(z)
        for shift in range(-12, 13):
            k = center + shift
            arg = x - k
            valid = (arg >= -12) & (arg <= 12)
            if np.any(valid):
                weight = np.zeros(n)
                weight[valid] = _mwt_phi(arg[valid], basis) * 2.0**(level/2.0)
                out += weight[:, None] * a[k % a.shape[0]]
        return out

    def _mwt_basis():
        """原src/bases/1D/phi.dat及h0.dat；保持原始截断系数，不重新归一化。"""
        import numpy as np

        phi = np.fromstring("""
            -7.6946487e-06 -6.9052794e-06 -5.9830047e-06 -4.9424895e-06 -3.7981958e-06 -2.5646464e-06
            -1.2563523e-06 1.1218393e-07 1.5264415e-06 2.9719234e-06 4.4341052e-06 5.8984924e-06
            7.3505586e-06 8.7758112e-06 1.0159722e-05 1.1487800e-05 1.2745515e-05 1.3918379e-05
            1.4991858e-05 1.5951469e-05 1.6782671e-05 1.7470989e-05 1.8001874e-05 1.8360861e-05
            1.8533387e-05 1.8505006e-05 1.8261125e-05 1.7787349e-05 1.7068999e-05 1.6091845e-05
            1.4840873e-05 1.3310986e-05 1.1523957e-05 9.5082086e-06 7.2917502e-06 4.9027247e-06
            2.3692400e-06 -2.8060229e-07 -3.0186738e-06 -5.8168844e-06 -8.6470987e-06 -1.1481231e-05
            -1.4291143e-05 -1.7048752e-05 -1.9725916e-05 -2.2294557e-05 -2.4726526e-05 -2.6993753e-05
            -2.9068082e-05 -3.0921450e-05 -3.2525693e-05 -3.3852758e-05 -3.4874466e-05 -3.5562787e-05
            -3.5889511e-05 -3.5826648e-05 -3.5345929e-05 -3.4419459e-05 -3.3018808e-05 -3.1116397e-05
            -2.8683143e-05 -2.5708978e-05 -2.2236024e-05 -1.8319519e-05 -1.4013874e-05 -9.3737787e-06
            -4.4538399e-06 6.9133602e-07 6.0071014e-06 1.1438873e-05 1.6931988e-05 2.2431874e-05
            2.7883861e-05 3.3233382e-05 3.8425759e-05 4.3406436e-05 4.8120725e-05 5.2514079e-05
            5.6531797e-05 6.0119350e-05 6.3222018e-05 6.5785293e-05 6.7754426e-05 6.9074950e-05
            6.9692060e-05 6.9551366e-05 6.8597952e-05 6.6777602e-05 6.4035095e-05 6.0316827e-05
            5.5566303e-05 4.9763593e-05 4.2990460e-05 3.5354631e-05 2.6962157e-05 1.7919681e-05
            8.3336443e-06 -1.6894769e-06 -1.2043157e-05 -2.2620969e-05 -3.3316355e-05 -4.4022911e-05
            -5.4634062e-05 -6.5043419e-05 -7.5144392e-05 -8.4830606e-05 -9.3995456e-05 -1.0253259e-04
            -1.1033537e-04 -1.1729748e-04 -1.2331224e-04 -1.2827339e-04 -1.3207419e-04 -1.3460844e-04
            -1.3576931e-04 -1.3545075e-04 -1.3354572e-04 -1.2994848e-04 -1.2455142e-04 -1.1724999e-04
            -1.0793407e-04 -9.6564126e-05 -8.3299516e-05 -6.8351174e-05 -5.1926644e-05 -3.4234714e-05
            -1.5483710e-05 4.1179181e-06 2.4361671e-05 4.5039201e-05 6.5941943e-05 8.6861595e-05
            1.0758956e-04 1.2791756e-04 1.4763696e-04 1.6653954e-04 1.8441661e-04 2.0105999e-04
            2.1626097e-04 2.2981138e-04 2.4150247e-04 2.5112616e-04 2.5847357e-04 2.6333679e-04
            2.6550672e-04 2.6477573e-04 2.6093433e-04 2.5377553e-04 2.4308869e-04 2.2866911e-04
            2.1030129e-04 1.8790655e-04 1.6179706e-04 1.3238785e-04 1.0008708e-04 6.5305549e-05
            2.8452983e-05 -1.0060520e-05 -4.9824870e-05 -9.0430177e-05 -1.3146622e-04 -1.7252320e-04
            -2.1319082e-04 -2.5305936e-04 -2.9171844e-04 -3.2875842e-04 -3.6376885e-04 -3.9634014e-04
            -4.2606179e-04 -4.5252431e-04 -4.7531704e-04 -4.9403068e-04 -5.0825436e-04 -5.1757903e-04
            -5.2159348e-04 -5.1988917e-04 -5.1205410e-04 -4.9768101e-04 -4.7635565e-04 -4.4767525e-04
            -4.1121595e-04 -3.6682073e-04 -3.1510478e-04 -2.5688978e-04 -1.9298332e-04 -1.2419856e-04
            -5.1346286e-05 2.4761767e-05 1.0331407e-04 1.8349932e-04 2.6450572e-04 3.4552214e-04
            4.2573664e-04 5.0433824e-04 5.8051486e-04 6.5345564e-04 7.2234837e-04 7.8638235e-04
            8.4474519e-04 8.9662639e-04 9.4121331e-04 9.7769578e-04 1.0052607e-03 1.0230986e-03
            1.0303954e-03 1.0263428e-03 1.0101253e-03 9.8093679e-04 9.3795766e-04 8.8039046e-04
            8.0739651e-04 7.1866105e-04 6.1540624e-04 4.9927142e-04 3.7186703e-04 2.3481528e-04
            8.9733128e-05 -6.1760131e-05 -2.1804602e-04 -3.7750615e-04 -5.3852144e-04 -6.9947392e-04
            -8.5874420e-04 -1.0147146e-03 -1.1657654e-03 -1.3102792e-03 -1.4466362e-03 -1.5732190e-03
            -1.6884075e-03 -1.7905850e-03 -1.8781307e-03 -1.9494284e-03 -2.0028566e-03 -2.0368003e-03
            -2.0496364e-03 -2.0397519e-03 -2.0055209e-03 -1.9453350e-03 -1.8575599e-03 -1.7406044e-03
            -1.5927961e-03 -1.4135008e-03 -1.2051723e-03 -9.7111534e-04 -7.1457473e-04 -4.3882036e-04
            -1.4711050e-04 1.5729099e-04 4.7112270e-04 7.9112264e-04 1.1140281e-03 1.4365781e-03
            1.7555093e-03 2.0675610e-03 2.3694695e-03 2.6579747e-03 2.9298123e-03 3.1817227e-03
            3.4104411e-03 3.6127086e-03 3.7852593e-03 3.9248358e-03 4.0281705e-03 4.0920079e-03
            4.1130778e-03 4.0881287e-03 4.0138842e-03 3.8871026e-03 3.7044906e-03 3.4628410e-03
            3.1587830e-03 2.7910316e-03 2.3645878e-03 1.8862104e-03 1.3625331e-03 8.0024366e-04
            2.0600344e-04 -4.1351263e-04 -1.0516363e-03 -1.7016966e-03 -2.3570228e-03 -3.0109458e-03
            -3.6567933e-03 -4.2878974e-03 -4.8975848e-03 -5.4791887e-03 -6.0260346e-03 -6.5314569e-03
            -6.9887798e-03 -7.3913393e-03 -7.7324576e-03 -8.0054733e-03 -8.2037052e-03 -8.3204961e-03
            -8.3491594e-03 -8.2830456e-03 -8.1154563e-03 -7.8397612e-03 -7.4492277e-03 -6.9372961e-03
            -6.2970736e-03 -5.5259379e-03 -4.6343149e-03 -3.6363343e-03 -2.5458590e-03 -1.3768690e-03
            -1.4328615e-04 1.1409365e-03 2.4618627e-03 3.8055482e-03 5.1580509e-03 6.5054305e-03
            7.8337420e-03 9.1290477e-03 1.0377400e-02 1.1564864e-02 1.2677489e-02 1.3701343e-02
            1.4622473e-02 1.5426950e-02 1.6100816e-02 1.6630147e-02 1.7000981e-02 1.7199401e-02
            1.7211432e-02 1.7023176e-02 1.6620631e-02 1.5989940e-02 1.5117030e-02 1.3988192e-02
            1.2589012e-02 1.0914105e-02 8.9860432e-03 6.8354567e-03 4.4923862e-03 1.9871367e-03
            -6.5012185e-04 -3.3891439e-03 -6.1997275e-03 -9.0516465e-03 -1.1914686e-02 -1.4758629e-02
            -1.7553253e-02 -2.0268348e-02 -2.2873686e-02 -2.5339062e-02 -2.7634242e-02 -2.9729027e-02
            -3.1593177e-02 -3.3196500e-02 -3.4508748e-02 -3.5499739e-02 -3.6139212e-02 -3.6397002e-02
            -3.6242823e-02 -3.5646548e-02 -3.4577835e-02 -3.3006644e-02 -3.0902477e-02 -2.8235625e-02
            -2.4974818e-02 -2.1108978e-02 -1.6690449e-02 -1.1790115e-02 -6.4774923e-03 -8.2272242e-04
            5.1043789e-03 1.1233809e-02 1.7495676e-02 2.3820023e-02 3.0136929e-02 3.6376458e-02
            4.2468673e-02 4.8343651e-02 5.3931442e-02 5.9162137e-02 6.3965772e-02 6.8272451e-02
            7.2012197e-02 7.5115131e-02 7.7511253e-02 7.9130712e-02 7.9903474e-02 7.9759731e-02
            7.8629391e-02 7.6442732e-02 7.3129528e-02 6.8620275e-02 6.2844355e-02 5.5733085e-02
            4.7213922e-02 3.7264388e-02 2.6021437e-02 1.3669312e-02 3.8873209e-04 -1.3637934e-02
            -2.8229205e-02 -4.3203069e-02 -5.8377848e-02 -7.3571650e-02 -8.8602719e-02 -1.0328922e-01
            -1.1744935e-01 -1.3090131e-01 -1.4346327e-01 -1.5495346e-01 -1.6519002e-01 -1.7399120e-01
            -1.8117513e-01 -1.8656008e-01 -1.8996413e-01 -1.9120562e-01 -1.9010255e-01 -1.8647336e-01
            -1.8013593e-01 -1.7090885e-01 -1.5860976e-01 -1.4305768e-01 -1.2406946e-01 -1.0146576e-01
            -7.5059559e-02 -4.4763875e-02 -1.0814753e-02 2.6454522e-02 6.6718014e-02 1.0964627e-01
            1.5491177e-01 2.0218579e-01 2.5114044e-01 3.0144722e-01 3.5277807e-01 4.0480462e-01
            4.5719871e-01 5.0963205e-01 5.6177642e-01 6.1330359e-01 6.6388527e-01 7.1319329e-01
            7.6089932e-01 8.0667521e-01 8.5019262e-01 8.9112344e-01 9.2913923e-01 9.6391202e-01
            9.9511321e-01 1.0224150e+00 1.0454886e+00 1.0640066e+00 1.0776393e+00 1.0860613e+00
            1.0889380e+00 1.0860613e+00 1.0776393e+00 1.0640066e+00 1.0454886e+00 1.0224150e+00
            9.9511321e-01 9.6391202e-01 9.2913923e-01 8.9112344e-01 8.5019262e-01 8.0667521e-01
            7.6089932e-01 7.1319329e-01 6.6388527e-01 6.1330359e-01 5.6177642e-01 5.0963205e-01
            4.5719871e-01 4.0480462e-01 3.5277807e-01 3.0144722e-01 2.5114044e-01 2.0218579e-01
            1.5491177e-01 1.0964627e-01 6.6718014e-02 2.6454522e-02 -1.0814753e-02 -4.4763875e-02
            -7.5059559e-02 -1.0146576e-01 -1.2406946e-01 -1.4305768e-01 -1.5860976e-01 -1.7090885e-01
            -1.8013593e-01 -1.8647336e-01 -1.9010255e-01 -1.9120562e-01 -1.8996413e-01 -1.8656008e-01
            -1.8117513e-01 -1.7399120e-01 -1.6519002e-01 -1.5495346e-01 -1.4346327e-01 -1.3090131e-01
            -1.1744935e-01 -1.0328922e-01 -8.8602719e-02 -7.3571650e-02 -5.8377848e-02 -4.3203069e-02
            -2.8229205e-02 -1.3637934e-02 3.8873209e-04 1.3669312e-02 2.6021437e-02 3.7264388e-02
            4.7213922e-02 5.5733085e-02 6.2844355e-02 6.8620275e-02 7.3129528e-02 7.6442732e-02
            7.8629391e-02 7.9759731e-02 7.9903474e-02 7.9130712e-02 7.7511253e-02 7.5115131e-02
            7.2012197e-02 6.8272451e-02 6.3965772e-02 5.9162137e-02 5.3931442e-02 4.8343651e-02
            4.2468673e-02 3.6376458e-02 3.0136929e-02 2.3820023e-02 1.7495676e-02 1.1233809e-02
            5.1043789e-03 -8.2272242e-04 -6.4774923e-03 -1.1790115e-02 -1.6690449e-02 -2.1108978e-02
            -2.4974818e-02 -2.8235625e-02 -3.0902477e-02 -3.3006644e-02 -3.4577835e-02 -3.5646548e-02
            -3.6242823e-02 -3.6397002e-02 -3.6139212e-02 -3.5499739e-02 -3.4508748e-02 -3.3196500e-02
            -3.1593177e-02 -2.9729027e-02 -2.7634242e-02 -2.5339062e-02 -2.2873686e-02 -2.0268348e-02
            -1.7553253e-02 -1.4758629e-02 -1.1914686e-02 -9.0516465e-03 -6.1997275e-03 -3.3891439e-03
            -6.5012185e-04 1.9871367e-03 4.4923862e-03 6.8354567e-03 8.9860432e-03 1.0914105e-02
            1.2589012e-02 1.3988192e-02 1.5117030e-02 1.5989940e-02 1.6620631e-02 1.7023176e-02
            1.7211432e-02 1.7199401e-02 1.7000981e-02 1.6630147e-02 1.6100816e-02 1.5426950e-02
            1.4622473e-02 1.3701343e-02 1.2677489e-02 1.1564864e-02 1.0377400e-02 9.1290477e-03
            7.8337420e-03 6.5054305e-03 5.1580509e-03 3.8055482e-03 2.4618627e-03 1.1409365e-03
            -1.4328615e-04 -1.3768690e-03 -2.5458590e-03 -3.6363343e-03 -4.6343149e-03 -5.5259379e-03
            -6.2970736e-03 -6.9372961e-03 -7.4492277e-03 -7.8397612e-03 -8.1154563e-03 -8.2830456e-03
            -8.3491594e-03 -8.3204961e-03 -8.2037052e-03 -8.0054733e-03 -7.7324576e-03 -7.3913393e-03
            -6.9887798e-03 -6.5314569e-03 -6.0260346e-03 -5.4791887e-03 -4.8975848e-03 -4.2878974e-03
            -3.6567933e-03 -3.0109458e-03 -2.3570228e-03 -1.7016966e-03 -1.0516363e-03 -4.1351263e-04
            2.0600344e-04 8.0024366e-04 1.3625331e-03 1.8862104e-03 2.3645878e-03 2.7910316e-03
            3.1587830e-03 3.4628410e-03 3.7044906e-03 3.8871026e-03 4.0138842e-03 4.0881287e-03
            4.1130778e-03 4.0920079e-03 4.0281705e-03 3.9248358e-03 3.7852593e-03 3.6127086e-03
            3.4104411e-03 3.1817227e-03 2.9298123e-03 2.6579747e-03 2.3694695e-03 2.0675610e-03
            1.7555093e-03 1.4365781e-03 1.1140281e-03 7.9112264e-04 4.7112270e-04 1.5729099e-04
            -1.4711050e-04 -4.3882036e-04 -7.1457473e-04 -9.7111534e-04 -1.2051723e-03 -1.4135008e-03
            -1.5927961e-03 -1.7406044e-03 -1.8575599e-03 -1.9453350e-03 -2.0055209e-03 -2.0397519e-03
            -2.0496364e-03 -2.0368003e-03 -2.0028566e-03 -1.9494284e-03 -1.8781307e-03 -1.7905850e-03
            -1.6884075e-03 -1.5732190e-03 -1.4466362e-03 -1.3102792e-03 -1.1657654e-03 -1.0147146e-03
            -8.5874420e-04 -6.9947392e-04 -5.3852144e-04 -3.7750615e-04 -2.1804602e-04 -6.1760131e-05
            8.9733128e-05 2.3481528e-04 3.7186703e-04 4.9927142e-04 6.1540624e-04 7.1866105e-04
            8.0739651e-04 8.8039046e-04 9.3795766e-04 9.8093679e-04 1.0101253e-03 1.0263428e-03
            1.0303954e-03 1.0230986e-03 1.0052607e-03 9.7769578e-04 9.4121331e-04 8.9662639e-04
            8.4474519e-04 7.8638235e-04 7.2234837e-04 6.5345564e-04 5.8051486e-04 5.0433824e-04
            4.2573664e-04 3.4552214e-04 2.6450572e-04 1.8349932e-04 1.0331407e-04 2.4761768e-05
            -5.1346286e-05 -1.2419856e-04 -1.9298332e-04 -2.5688978e-04 -3.1510478e-04 -3.6682073e-04
            -4.1121595e-04 -4.4767525e-04 -4.7635565e-04 -4.9768100e-04 -5.1205410e-04 -5.1988917e-04
            -5.2159348e-04 -5.1757903e-04 -5.0825436e-04 -4.9403068e-04 -4.7531704e-04 -4.5252431e-04
            -4.2606179e-04 -3.9634014e-04 -3.6376885e-04 -3.2875842e-04 -2.9171844e-04 -2.5305936e-04
            -2.1319082e-04 -1.7252320e-04 -1.3146622e-04 -9.0430177e-05 -4.9824870e-05 -1.0060520e-05
            2.8452983e-05 6.5305549e-05 1.0008708e-04 1.3238785e-04 1.6179706e-04 1.8790655e-04
            2.1030129e-04 2.2866911e-04 2.4308869e-04 2.5377553e-04 2.6093433e-04 2.6477573e-04
            2.6550672e-04 2.6333679e-04 2.5847357e-04 2.5112616e-04 2.4150247e-04 2.2981138e-04
            2.1626097e-04 2.0105999e-04 1.8441661e-04 1.6653954e-04 1.4763696e-04 1.2791756e-04
            1.0758956e-04 8.6861595e-05 6.5941943e-05 4.5039201e-05 2.4361672e-05 4.1179182e-06
            -1.5483710e-05 -3.4234714e-05 -5.1926643e-05 -6.8351174e-05 -8.3299516e-05 -9.6564125e-05
            -1.0793407e-04 -1.1724999e-04 -1.2455142e-04 -1.2994848e-04 -1.3354572e-04 -1.3545075e-04
            -1.3576931e-04 -1.3460844e-04 -1.3207419e-04 -1.2827339e-04 -1.2331224e-04 -1.1729748e-04
            -1.1033537e-04 -1.0253259e-04 -9.3995456e-05 -8.4830606e-05 -7.5144391e-05 -6.5043419e-05
            -5.4634062e-05 -4.4022911e-05 -3.3316355e-05 -2.2620969e-05 -1.2043157e-05 -1.6894767e-06
            8.3336445e-06 1.7919681e-05 2.6962157e-05 3.5354631e-05 4.2990460e-05 4.9763593e-05
            5.5566303e-05 6.0316828e-05 6.4035095e-05 6.6777602e-05 6.8597952e-05 6.9551366e-05
            6.9692060e-05 6.9074950e-05 6.7754427e-05 6.5785293e-05 6.3222018e-05 6.0119351e-05
            5.6531798e-05 5.2514079e-05 4.8120725e-05 4.3406436e-05 3.8425759e-05 3.3233382e-05
            2.7883861e-05 2.2431875e-05 1.6931988e-05 1.1438873e-05 6.0071016e-06 6.9133616e-07
            -4.4538397e-06 -9.3737786e-06 -1.4013874e-05 -1.8319519e-05 -2.2236024e-05 -2.5708978e-05
            -2.8683143e-05 -3.1116397e-05 -3.3018808e-05 -3.4419459e-05 -3.5345929e-05 -3.5826647e-05
            -3.5889511e-05 -3.5562787e-05 -3.4874466e-05 -3.3852758e-05 -3.2525693e-05 -3.0921450e-05
            -2.9068082e-05 -2.6993753e-05 -2.4726526e-05 -2.2294556e-05 -1.9725916e-05 -1.7048752e-05
            -1.4291143e-05 -1.1481231e-05 -8.6470985e-06 -5.8168843e-06 -3.0186737e-06 -2.8060216e-07
            2.3692401e-06 4.9027248e-06 7.2917504e-06 9.5082087e-06 1.1523957e-05 1.3310986e-05
            1.4840873e-05 1.6091845e-05 1.7068999e-05 1.7787349e-05 1.8261126e-05 1.8505006e-05
            1.8533387e-05 1.8360861e-05 1.8001874e-05 1.7470989e-05 1.6782672e-05 1.5951469e-05
            1.4991858e-05 1.3918379e-05 1.2745515e-05 1.1487800e-05 1.0159722e-05 8.7758113e-06
            7.3505587e-06 5.8984925e-06 4.4341053e-06 2.9719235e-06 1.5264417e-06 1.1218406e-07
            -1.2563522e-06 -2.5646463e-06 -3.7981957e-06 -4.9424894e-06 -5.9830046e-06 -6.9052792e-06
            -7.6946486e-06
        """, sep=" ")
        h0 = np.fromstring("""
            -9.1124926e-04 -1.1111255e-03 1.8563725e-03 2.2020410e-03 -3.8417067e-03 -4.3844042e-03
            8.1391208e-03 8.7439499e-03 -1.7891666e-02 -1.7283937e-02 4.1948920e-02 3.2264476e-02
            -1.0990692e-01 -5.0471585e-02 4.3378231e-01 7.6628806e-01 4.3378231e-01 -5.0471585e-02
            -1.0990692e-01 3.2264476e-02 4.1948920e-02 -1.7283937e-02 -1.7891666e-02 8.7439499e-03
            8.1391208e-03 -4.3844042e-03 -3.8417067e-03 2.2020410e-03 1.8563725e-03 -1.1111255e-03
            -9.1124926e-04
        """, sep=" ")
        # phi1_shape使用px(2)-px(1)，t.dat的两个原始值如下。
        return phi, h0, 0.066666667 - 0.033333333

    def _run_array(data, j0=3, j1=5, axis=0, boundary="symmetric",
            normalization="physical", return_reconstruction=True,
            block_size=256, nan_policy="raise"):
        import operator
        import numpy as np

        if np.iscomplexobj(data):
            raise TypeError("MWT输入必须为实数；不接受复数。")
        if np.ma.isMaskedArray(data):
            data = np.ma.asarray(data, dtype=np.float64).filled(np.nan)
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 0:
            raise ValueError("data至少需要一个时间维。")
        try:
            axis = operator.index(axis)
        except TypeError as exc:
            raise TypeError("axis必须是整数。") from exc
        if not -data.ndim <= axis < data.ndim:
            raise ValueError("axis超出输入数组维数。")
        axis %= data.ndim
        n = data.shape[axis]
        if n < 4 or n & (n - 1):
            raise ValueError("时间长度N必须是至少4的2次幂；请提供连续等间隔数据。")
        jmax = n.bit_length() - 1
        for name, value in (("j0", j0), ("j1", j1), ("block_size", block_size)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(name + "必须是整数。")
        j0, j1, block_size = int(j0), int(j1), int(block_size)
        if not 0 <= j0 < j1 < jmax:
            raise ValueError("需要0 <= j0 < j1 < log2(N)，保证三个尺度窗均非空。")
        if block_size <= 0:
            raise ValueError("block_size必须为正整数。")
        if boundary in ("symmetric", 2) and not isinstance(boundary, bool):
            boundary = 2
        elif boundary in ("periodic", 1) and not isinstance(boundary, bool):
            boundary = 1
        else:
            raise ValueError("boundary只支持'symmetric'/2或'periodic'/1。")
        if normalization not in ("physical", "original"):
            raise ValueError("normalization只支持'physical'或'original'。")
        if nan_policy not in ("raise", "propagate"):
            raise ValueError("nan_policy只支持'raise'或'propagate'。")
        if np.isinf(data).any():
            raise ValueError("输入含Inf，请先核对无效值和数据单位。")
        if nan_policy == "raise" and np.isnan(data).any():
            raise ValueError("输入含NaN；请先处理缺测，或用nan_policy='propagate'保留缺测列。")

        moved = np.moveaxis(data, axis, 0)
        flat = moved.reshape(n, -1)
        coefficients = np.full((3, n, flat.shape[1]), np.nan, dtype=np.float64)
        reconstruction = np.full_like(coefficients, np.nan) if return_reconstruction else None
        phi, h0, basis_step = _mwt_basis()
        basis = phi, h0, basis_step
        factor = np.sqrt(n) if normalization == "physical" else 1.0
        for start in range(0, flat.shape[1], block_size):
            stop = min(start + block_size, flat.shape[1])
            block = flat[:, start:stop]
            good = ~np.isnan(block).any(axis=0)
            if not np.any(good):
                continue
            z = block[:, good]
            mean = z.mean(axis=0, keepdims=True)
            low = _mwt_lowpass(z - mean, j0, boundary, basis)
            residual = (z - mean) - low
            middle = _mwt_lowpass(residual, j1, boundary, basis)
            high = residual - middle
            low = low + mean
            destinations = np.flatnonzero(good) + start
            for window, part in enumerate((low, middle, high)):
                transformed = _mwt_analysis(part, boundary, basis)[:n] * factor
                coefficients[window][:, destinations] = transformed
                if return_reconstruction:
                    reconstruction[window][:, destinations] = part

        shape = (3,) + moved.shape
        coefficients = np.moveaxis(coefficients.reshape(shape), 1, axis + 1)
        if not return_reconstruction:
            return coefficients
        reconstruction = np.moveaxis(reconstruction.reshape(shape), 1, axis + 1)
        return coefficients, reconstruction

    if not isinstance(data, dict):
        if reconstruction_keys is not None:
            raise ValueError("reconstruction_keys仅支持字典输入；数组输入请保持None。")
        return _run_array(data, j0, j1, axis, boundary, normalization,
                          return_reconstruction, block_size, nan_policy)

    if not data:
        raise ValueError("data字典不能为空。")
    for key in data:
        if not isinstance(key, str):
            raise TypeError("data字典的键必须全部是字符串。")

    selected = None
    if reconstruction_keys is not None:
        if not return_reconstruction:
            raise ValueError("return_reconstruction=False时reconstruction_keys必须为None。")
        if not isinstance(reconstruction_keys, (tuple, list)):
            raise TypeError("reconstruction_keys必须是tuple、list或None。")
        if any(not isinstance(key, str) for key in reconstruction_keys):
            raise TypeError("reconstruction_keys中的键必须全部是字符串。")
        selected = set(reconstruction_keys)
        if len(selected) != len(reconstruction_keys):
            raise ValueError("reconstruction_keys不能包含重复键。")
        unknown = [key for key in reconstruction_keys if key not in data]
        if unknown:
            raise ValueError("reconstruction_keys包含未知键：" + repr(unknown))

    coefficient_map = {}
    reconstruction_map = {} if return_reconstruction else None
    for key, value in data.items():
        keep_reconstruction = return_reconstruction and (selected is None or key in selected)
        try:
            transformed = _run_array(value, j0, j1, axis, boundary, normalization,
                                     keep_reconstruction, block_size, nan_policy)
        except (TypeError, ValueError) as exc:
            raise type(exc)("data[" + repr(key) + "]: " + str(exc)) from exc
        if keep_reconstruction:
            coefficient_map[key], reconstruction_map[key] = transformed
        else:
            coefficient_map[key] = transformed

    if return_reconstruction:
        return coefficient_map, reconstruction_map
    return coefficient_map
