"""Bounded release verification. No fitting, resampling, ERA5 reads or rendering."""
from pathlib import Path
import argparse,ast,hashlib,json,re,subprocess,sys
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    checks=[]
    def check(name,condition,**details):
        checks.append(dict(check=name,passed=bool(condition),**details))
        if not condition:raise AssertionError((name,details))
    master=pd.read_csv(ROOT/'lifecycle/input/main_sid.csv',keep_default_na=False,na_values=[''])
    check('unique_master',len(master)==2340 and master.sid.is_unique)
    records=pd.read_csv(ROOT/'lifecycle/input/analysis_records_current.csv.gz',keep_default_na=False,na_values=[''])
    check('current_record_support',len(records)==63769 and set(records.sid)==set(master.sid),records=len(records))
    events=pd.read_csv(ROOT/'lifecycle/input/all_return_events.csv',keep_default_na=False,na_values=[''])
    check('return_events',len(events)==133 and events.sid.nunique()==123,events=len(events),SID=events.sid.nunique())
    run=subprocess.run([sys.executable,'-B',str(ROOT/'verify_frozen_lifecycle.py')],capture_output=True,text=True)
    check('three_actual_tracks_current_core',run.returncode==0,result=run.stdout.strip(),stderr=run.stderr[-500:])
    map_=pd.read_csv(ROOT/'figure_map.csv',keep_default_na=False)
    check('50_current_figures',len(map_)==50 and map_.figure.nunique()==50)
    for row in map_.to_dict('records'):
        check('figure_entry', (ROOT/row['entry']).is_file(),figure=row['figure'],entry=row['entry'])
        if row['final_layout_entry']:check('layout_entry',(ROOT/row['final_layout_entry']).is_file(),figure=row['figure'])
    for file in ROOT.rglob('*.py'):
        ast.parse(file.read_text(encoding='utf-8-sig'))
        check('python_syntax',True,file=file.relative_to(ROOT).as_posix())
        check('portable_code',not re.search(r'[CHI]:[/\\]|Users[/\\]TBYC',file.read_text(encoding='utf-8-sig')),file=file.relative_to(ROOT).as_posix())
    for file in ROOT.glob('figures/*/inputs.json'):
        for key,item in json.loads(file.read_text(encoding='utf-8-sig')).items():
            p=file.parent/item['relative_path']
            check('frozen_figure_input',p.is_file() and sha(p)==item['sha256'],file=p.relative_to(ROOT).as_posix())
    for file in ROOT.glob('framework/source_data/de_threshold/*/complete.json'):
        for name,h in json.loads(file.read_text())['outputs'].items():
            check('threshold_input_hash',sha(file.parent/name)==h,file=(file.parent/name).relative_to(ROOT).as_posix())
    p=ROOT/'framework/reference/primary/primary_trends.csv';table=pd.read_csv(p)
    check('saved_seven_test_BH',len(table)==7 and np.allclose(multipletests(table.p_value,method='fdr_bh')[1],table.q_value,rtol=0,atol=1e-14))
    # Drivers load exact source functions through AST. Verify every declared name.
    deps=ROOT/'framework/dependencies'
    expected={deps/'v62/stages/analysis_b.py':{'make_pair','fit_pair','summarize_raw','bh_adjust'},deps/'v62/framework/run_framework_v62.py':{'bools','fold','ridge','bootstrap_indices','a2','design_matrix','a3'},deps/'temporal/temporal_analysis.py':{'fit_trend'}}
    for file,names in expected.items():
        found={n.name for n in ast.parse(file.read_text()).body if isinstance(n,ast.FunctionDef)}
        check('exact_function_dependencies',names<=found,file=file.relative_to(ROOT).as_posix())
    report=dict(status='PASS',scope='saved inputs, exact small-track lifecycle regression, source syntax, entry/input existence, seven-test saved BH; no full scientific campaign or figure rendering',checks=checks,figures=50)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='PASS',checks=len(checks),figures=50)))

if __name__=='__main__':main()
