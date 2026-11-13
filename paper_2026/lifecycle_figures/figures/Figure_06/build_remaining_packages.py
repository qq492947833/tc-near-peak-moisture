"""Package existing, saved MWT/rise display data and source plot statements."""
from pathlib import Path
import sys,ast,json,shutil,hashlib,importlib.util
N=Path(__file__).resolve().parents[2]
sp=importlib.util.spec_from_file_location('builder',N/'figures/Figure_04/build_assigned_packages.py');b=importlib.util.module_from_spec(sp);sp.loader.exec_module(b)
A=b.U.parent; D=b.R/'mwt_addendum_20261025'; M=A.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def trees(p):return ast.parse(p.read_text(encoding='utf-8-sig'))
def func(p,n):return next(x for x in trees(p).body if isinstance(x,ast.FunctionDef) and x.name==n)
def cut_before_draw(fn):
    return fn.body[:next(i for i,x in enumerate(fn.body) if isinstance(x,ast.Expr) and ast.unparse(x.value)=='fig.canvas.draw()')]
def make(fid,code,sources,origin,replacements=None):
    pkg=N/'figures'/fid;(pkg/'inputs').mkdir(parents=True,exist_ok=True);ii={}
    for key,p in sources.items():
        dest=pkg/'inputs'/p.name;shutil.copy2(p,dest);ii[key]={'relative_path':'inputs/'+p.name,'sha256':sha(dest),'original_path':str(p),'bytes':dest.stat().st_size}
    (pkg/'render_source.py').write_text(b.HEADER+'\n'+code,encoding='utf-8')
    run=b.RUN.replace('m.rd=m.source=m.read=m.external=load;','m.load=load;m.rd=m.source=m.read=m.external=load;')
    (pkg/'run.py').write_text(run,encoding='utf-8');(pkg/'inputs.json').write_text(json.dumps(ii,indent=2),encoding='utf-8')
    (pkg/'config.json').write_text(json.dumps({'figure_id':fid,'function':'build','arguments':[],'text_replacements':replacements or {},'original_source':str(origin),'original_source_sha256':sha(origin)},ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copy2(origin,pkg/'original_plot_source.txt');print('PACKAGED',fid,len(ii),flush=True)
    shutil.copy2(N/'environment/qa_tools/audit_panel_alignment.py',pkg/'audit_panel_alignment.py')
def mwt():
    script=A/'scripts/mwt_departure_figures.py';ipt=A/'scripts/mwt_departure_figure_inputs.py';tree=trees(script);it=trees(ipt)
    constants='\n'.join(ast.unparse(x) for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['STYLE','LABELS','UNITS']for t in x.targets))
    constants+='\n'+ '\n'.join(ast.unparse(x) for x in it.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name)and t.id=='PRIMARY'for t in x.targets))
    helpers='\n'.join(ast.unparse(func(ipt,k))for k in ['primary_mask','display_intervals'])
    setup=func(script,'setup');setup.body=setup.body[:3]+[ast.parse('return plt, None').body[0]]
    common=constants+'\n'+helpers+'\n'+ast.unparse(setup)+'\n'+ast.unparse(func(script,'decorate'))+'\n'+ast.unparse(func(script,'interval'))+'\nimport types\ninputs=types.SimpleNamespace(PRIMARY=PRIMARY,primary_mask=primary_mask,display_intervals=display_intervals)\n'
    files=ast.literal_eval(next(x.value for x in it.body if isinstance(x,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='FILES'for t in x.targets)))
    files.update(ast.literal_eval(next(x.value for x in it.body if isinstance(x,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='THEORY_FILES'for t in x.targets))))
    state=json.loads((M/'reports/scientific_review_watcher_status.json').read_text());tab=Path(state['table_review']['path']);theory=Path(state['theory_review']['path'])
    for fid,draw,keys in [('Figure_08','draw_core',['time','primary']),('Figure_S16','draw_support',['coverage','reference_size','hydro_size']),('Figure_S17','draw_levels',['pre']),('Figure_S18','draw_sensitivity',['sensitivity','motion'])]:
        code=common+'\n'+ast.unparse(func(script,draw))+ '\ndef build():\n    frames={k:load(k) for k in '+repr(keys)+'}\n    for k,d in frames.items():\n        if "level" in d:d["level"]=d["level"].astype(str)\n        table("'+fid+'", k, d, "Frozen precomputed display source; selected below by original plotting statements")\n    fig,layout='+draw+'(frames)\n    for a,i in zip(layout["axes"],layout["panel_ids"]):a.set_gid(i)\n    export(fig,"'+fid+'",**layout)\n'
        sources={k:(theory if k in ['reference_size','hydro_size']else tab)/files[k] for k in keys};make(fid,code,sources,script)
def rise():
    script=A/'scripts/plot_rise_environment_history.py';fn=func(script,'plot');fn.body=cut_before_draw(fn)+ast.parse('table("rise","a_f",pd.DataFrame(records),"Saved per-SID pointwise histories")\nexport(fig,"rise")').body
    code=ast.unparse(fn)+'\ndef build():\n    plot(load("time"))\n'
    make('Figure_S19',code,{'time':A/'figures/RISE_ENVIRONMENT_HISTORY_source.csv'},script,{'PI headroom':'PI − V','Headroom (m s$^{-1}$)':'PI − V (m s$^{-1}$)'})
    script=A/'scripts/plot_rise_moisture_process.py';fn=func(script,'main');body=cut_before_draw(fn);start=next(i for i,x in enumerate(body)if 'mpl.rcParams.update' in ast.unparse(x))
    init=ast.parse('wide=load("states")\nstates=wide.melt(id_vars=["sid","radial"],value_vars=["matched_rise","matched_Resolution"],var_name="estimand",value_name="value")\nstates["metric"]="q700"\nsummary=load("integrals")').body
    fn.name='build';fn.body=init+body[start:]+ast.parse('table("moisture","a_b",pd.DataFrame(scatter_records),"Actual same-SID matched-wind states")\ntable("moisture","c_d",pd.DataFrame(records),"Saved actual-ascent diagnosed integrals and percentile intervals")\nexport(fig,"moisture")').body
    make('Figure_S20',ast.unparse(fn),{'states':A/'figures/RISE_MOISTURE_PROCESS_states_source.csv','integrals':A/'figures/RISE_MOISTURE_PROCESS_integrals_source.csv'},script)
def rise9():
    origin=D/'figures/extend_figure9.py';af=D/'figures/sources/executed_frozen_af_body.py'
    text=origin.read_text(encoding='utf-8');t=ast.parse(text)
    first=next(i for i,x in enumerate(t.body)if isinstance(x,ast.Assign)and any(isinstance(a,ast.Name)and a.id=='OLD_HEIGHT'for a in x.targets))
    last=next(i for i,x in enumerate(t.body[first:],first)if isinstance(x,ast.Assign)and any(isinstance(a,ast.Name)and a.id=='after'for a in x.targets))
    segment=t.body[first:last]
    start='curves=load("curves")\npanels=[curves[curves.panel.eq(chr(97+i))].copy().sort_values("relative_hour") for i in range(6)]\ng=load("g").sort_values("window")\nassert g.window.tolist()==[0,1,2] and set(g.n_sid)=={108} and set(g.n_events)=={115}\nfig,axes=build_frozen(panels,None,None,None)\naxes=list(axes.flat)\npos_before=[list(a.get_position().bounds)for a in axes]\n'
    fn=ast.parse('def build():\n    pass').body[0];fn.body=ast.parse(start).body+segment+ast.parse('for i,d in enumerate(panels):table("rise",chr(97+i),d,"Saved equal-SID pointwise histories and percentile intervals")\ntable("rise","g",g,"Matched ascent-minus-Resolution APE; original 115 events / 108 SID; three saved windows")\nexport(fig,"rise")').body
    code=af.read_text(encoding='utf-8')+'\n'+ast.unparse(fn)
    make('Figure_09',code,{'curves':b.U/'reports/master_figure_regeneration/MAIN_RISE_COMPARISON_source.csv','g':D/'figures/sources/panel_g_source_rows.csv'},origin)
    shutil.copy2(af,N/'figures/Figure_09/original_af_source.txt')
if __name__=='__main__':mwt();rise();rise9()
