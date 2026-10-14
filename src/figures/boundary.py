from .common import *
D=DATA/'boundary'
COL = {'D_to_E': '#1679A5', 'E_to_F': '#BE782A'}
LAB = {'D_to_E': 'D → E: −24 to 0 h', 'E_to_F': 'E → F: −24 to 0 h'}

def style():
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'DejaVu Sans'], 'font.size': 8, 'axes.titlesize': 8.3, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'legend.frameon': False, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.65, 'xtick.major.width': 0.65, 'ytick.major.width': 0.65, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white', 'savefig.dpi': 601})

def read(name):
    return pd.read_csv(D / name, keep_default_na=False, na_values=['', 'nan', 'NaN'])

def build(new):
    style()
    src = (D / 'boundary_geometry.py').read_text(encoding='utf-8-sig')
    forest = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == 'forest']
    assert len(forest) == 1
    env = dict(np=np, pd=pd, plt=plt, COL=COL, LAB=LAB, ledger=[], bg=read('fig4_background.csv'), tr=read('fig4_transport.csv'), pidf=read('fig4_PI.csv'))
    exec(compile(ast.Module(body=forest, type_ignores=[]), 'boundary_forest', 'exec'), env)
    code = src[src.index('fig,axs=plt.subplots'):src.index('require_matplotlib_panel_alignment(fig,')]
    code = code.replace("'Along-track moisture change'", "'Boundary-neighbourhood moisture change'").replace("['Water vapour','Temperature','Sampling change','Total']", "['Water vapour','Temperature','Sampling/support','Total']")
    code = code.replace("'Approach to the intensity bound'", "'PI headroom H = PI − V'").replace("['PI change','−Wind change','Headroom']", "['ΔPI','−ΔV','ΔH']")
    code = code.replace("'Endpoint changes: −24 to 0 h; intervals: paired cyclone bootstrap'", "'a–e: −24 to 0 h; f: lagged 850 hPa rate; intervals: 95% paired cyclone bootstrap'")
    if new:
        rh = read('fig4_M3_fourway.csv')
        assert len(rh) == 20 and rh.scope.eq('primary25').all()
        assert rh.units.eq('percentage points').all()
        assert rh[rh.variable.eq('support_change')][['estimate', 'ci_low', 'ci_high']].eq(0).all().all()
        rh = rh.rename(columns={'variable': 'component', 'ci_low': 'lo', 'ci_high': 'hi'}).assign(region='200_800km', window='-24_0')
        code = code.replace("['q_contribution','T_contribution','sampling_support_remainder','native_RH_change']", "['time_q_contribution','time_T_contribution','domain_symmetric','fixed_support_endpoint_change']")
        code = code.replace("['Water vapour','Temperature','Sampling/support','Total']", "['Water vapour (time)','Temperature (time)','Geographic','Total']")
    else:
        rh = read('fig4_RH.csv')
    env['rh'] = rh
    exec(code, env)
    fig = env['fig']
    axes = env['axs'].ravel()
    for i, ax in enumerate(axes):
        ax.set_gid(chr(97 + i))
    for text in fig.findobj(Text):
        text.set_fontfamily('DejaVu Sans')
    return (fig, axes, pd.DataFrame(env['ledger']))
def render():
 fig,axes,rows=build(True)
 export(fig,'Fig2')

