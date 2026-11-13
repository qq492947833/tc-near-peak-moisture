"""Display only: full precision data remain unchanged. Decimal ROUND_HALF_UP."""
from decimal import Decimal, ROUND_HALF_UP, ROUND_CEILING, localcontext
import math
_SUP=str.maketrans("0123456789-+","⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺")
def exponent_text(exponent):
    return "×10"+str(int(exponent)).translate(_SUP)
def _decimal(value):
    if isinstance(value,Decimal): return value
    return Decimal(str(value))
def _plain(value,decimals):
    with localcontext() as ctx:
        ctx.prec=80
        q=value.quantize(Decimal(1).scaleb(-int(decimals)),rounding=ROUND_HALF_UP)
    if q==0:return "0"
    return format(q,"f").rstrip("0").rstrip(".") if "." in format(q,"f") else format(q,"f")
def format_number(value,decimals=3,sci_decimals=3):
    """Text number; zero/NA preserved, tiny or rounded-zero nonzero -> ×10 superscript."""
    if value is None:return "NA"
    try:d=_decimal(value)
    except Exception:return str(value)
    if not d.is_finite():return str(value) if isinstance(value,str) else ("NA" if d.is_nan() else ("−∞" if d<0 else "∞"))
    if d==0:return "0"
    decimals=max(0,min(4,int(decimals)))
    plain=_plain(d,decimals)
    if abs(d)<Decimal("0.001") or plain=="0":
        exp=d.copy_abs().adjusted()
        coefficient=d.scaleb(-exp)
        shown=_plain(coefficient,max(0,min(4,int(sci_decimals))))
        if abs(Decimal(shown))>=10:
            exp+=1
            shown=_plain(d.scaleb(-exp),sci_decimals)
        return shown.replace("-","−")+" "+exponent_text(exp)
    return plain.replace("-","−")
def format_group(values,decimals=3,sci_decimals=3):
    """Return (display_strings, explicit shared multiplier); no data mutation."""
    ds=[_decimal(v) for v in values]
    nz=[abs(d) for d in ds if d.is_finite() and d!=0]
    if nz and min(nz)<Decimal("0.001") and max(nz)/min(nz)<Decimal("1000"):
        exp=max(nz).adjusted()
        result=[_plain(d.scaleb(-exp),sci_decimals).replace("-","−") if d.is_finite() else str(d) for d in ds]
        if all(d==0 or s!="0" for d,s in zip(ds,result)):return result,exponent_text(exp)
    return [format_number(v,decimals,sci_decimals) for v in values],""
def nice_symmetric_limit(value):
    """One decimal in mantissa, rounded upward; display colour limit only."""
    d=abs(_decimal(value))
    if d==0:return 0.01
    step=Decimal(1).scaleb(d.adjusted()-1)
    return float((d/step).to_integral_value(rounding=ROUND_CEILING)*step)
def format_axis(ax,axis="y",decimals=3,force_scientific=False):
    """Numeric ticks only, with an explicit common multiplier in the axis unit."""
    from matplotlib.ticker import Formatter,ScalarFormatter
    obj=ax.yaxis if axis=="y" else ax.xaxis
    if not isinstance(obj.get_major_formatter(),ScalarFormatter):return None
    lim=ax.get_ylim() if axis=="y" else ax.get_xlim()
    vals=[abs(float(v)) for v in obj.get_majorticklocs() if min(lim)<=v<=max(lim) and math.isfinite(float(v)) and float(v)!=0]
    exp=0
    if vals and (force_scientific or (min(vals)<.001 and max(vals)<.1)):
        exp=math.floor(math.log10(max(vals)))
    class DisplayFormatter(Formatter):
        def __call__(self,x,pos=None):
            return format_number(_decimal(x).scaleb(-exp),decimals)
    obj.set_major_formatter(DisplayFormatter())
    if exp:
        label=ax.get_ylabel() if axis=="y" else ax.get_xlabel()
        marker=" ("+exponent_text(exp)+")"
        if marker not in label:
            if axis=="y":ax.set_ylabel(label+marker)
            else:ax.set_xlabel(label+marker)
    return {"axis":axis,"exponent":exp,"decimals":decimals}
def apply_numeric_format(fig,decimals=3):
    """Apply coherent half-up tick formatting; fixed category/angle/date labels retained."""
    result=[]
    for ax in fig.axes:
        for axis in ("x","y"):
            r=format_axis(ax,axis,decimals)
            if r is not None:result.append({"panel":ax.get_gid() or ax.get_label(),**r})
    return result
