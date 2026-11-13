"""Bounded release verification; no model fit, new bootstrap, or figure render."""
from pathlib import Path
import argparse, ast, csv, hashlib, json, os, re, subprocess, sys, time

ROOT = Path(__file__).resolve().parent

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024**2),b''):h.update(chunk)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--static-only',action='store_true')
    args=parser.parse_args()
    output=args.output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    work=output.parent/(output.stem+'_work');work.mkdir(exist_ok=False)
    checks=[]
    def check(name,condition,detail=None):
        checks.append({'check':name,'pass':bool(condition),'detail':detail})
        if not condition:raise AssertionError(name)
    files=[p for p in ROOT.rglob('*.py') if '.git' not in p.parts and '.venv' not in p.parts and work not in p.parents]
    for p in files:ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p.relative_to(ROOT)))
    check('Python source syntax',True,len(files))
    notebooks=list((ROOT/'notebooks').glob('*.ipynb'));cells=0
    for path in notebooks:
        obj=json.loads(path.read_text(encoding='utf-8-sig'))
        for cell in obj['cells']:
            if cell['cell_type']!='code':continue
            source=''.join(cell['source'])
            if re.search(r'^\s*[!%]',source,re.M):
                from IPython.core.inputtransformer2 import TransformerManager
                source=TransformerManager().transform_cell(source)
            compile(source,str(path), 'exec');cells+=1
    check('Notebook code syntax',True,{'notebooks':len(notebooks),'code_cells':cells})
    config=json.loads((ROOT/'config/raw_inputs.json').read_text())['ibtracs']
    check('Current years and USA_WIND',config['genesis_years']==[1982,2023] and config['wind_field']=='USA_WIND')
    check('E progress without rate gate default',config['lifecycle_parameters']['escalation_progress']==.25 and 'escalation_rate_threshold' not in config['lifecycle_parameters'])
    check('Current frozen lifecycle input paths',all(config[k].startswith('paper_2026/') for k in ['frozen_sid_table','frozen_record_table']))
    with (ROOT/'figure_map.csv').open(encoding='utf-8-sig',newline='')as f:figures=list(csv.DictReader(f))
    check('Current 50-figure catalogue',len(figures)==50)
    if not args.static_only:
        data=json.loads((ROOT/'release_data_manifest.json').read_text())
        count=0
        for asset in data['assets']:
            for record in asset['members']:
                p=ROOT/record['path'];check('input '+record['path'],p.is_file() and p.stat().st_size==record['bytes'] and sha256(p)==record['sha256']);count+=1
        check('All release input identities',True,count)
        commands=[
            ('lifecycle', [str(ROOT/'paper_2026/lifecycle_figures/validate_saved.py'),'--output',str(work/'lifecycle.json')]),
            ('newanalyses',[str(ROOT/'paper_2026/newanalyses/validate_saved.py'),'--output',str(work/'newanalyses.json')]),
            ('direct_mwt',[str(ROOT/'paper_2026/physics/direct_mwt/verify.py'),'--output',str(work/'direct_mwt')]),
            ('physics',[str(ROOT/'paper_2026/physics/verify_saved_results.py'),'--output',str(work/'physics')]),
            ('recovery_help',[str(ROOT/'paper_2026/physics/recovery/fit_policies_v6.py'),'--help']),
            ('latitude_help',[str(ROOT/'paper_2026/newanalyses/spatial/agents/latitude_risk/v2_fold_supported/fit_latitude_risk.py'),'--help']),
            ('raw_lifecycle_help',['-m','src.raw_workflow.lifecycle','--help'])
        ]
        temp=work/'temporary';temp.mkdir();mpl=work/'mpl';mpl.mkdir()
        env=dict(os.environ,TEMP=str(temp),TMP=str(temp),TMPDIR=str(temp),MPLCONFIGDIR=str(mpl),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        for name,cmd in commands:
            t=time.monotonic()
            p=subprocess.run([sys.executable,'-B','-X','utf8',*cmd],cwd=ROOT,env=env,capture_output=True,timeout=300)
            (work/(name+'.stdout.txt')).write_bytes(p.stdout);(work/(name+'.stderr.txt')).write_bytes(p.stderr)
            check(name,p.returncode==0,{'seconds':round(time.monotonic()-t,3),'returncode':p.returncode})
    result={'status':'PASS','scope':'Syntax, input integrity and representative deterministic/local checks. No full archive campaign, refit, new resampling or figure rendering.','python':sys.version,'checks_passed':len(checks),'checks':checks}
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'PASS','checks':len(checks),'output':str(output)}))

if __name__=='__main__':main()
