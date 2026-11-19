"""Export the accepted nine frozen MWT integrals in manuscript S26 order."""
from pathlib import Path
import csv, hashlib, argparse, shutil
def export(source, statistics, output):
    expected={'png': '7feca4f645ec60bece2c0455121bc502a74fbbb379ee9096764aba34e47c5adb', 'tiff': 'c0eb3fab144c19c882fcdb4687c307f95499362f6ddc446dabe3fb9e9add6f11'}
    output.mkdir(parents=True,exist_ok=True)
    if hashlib.sha256(statistics.read_bytes()).hexdigest()!='316f04cdddecc270020d2d8dcf9d226006b5e26b28c2dfb429f68fcfe4a0e364': raise ValueError('unexpected source data')
    for ext,digest in expected.items():
        path=source/('MWT_vertical_three_panel.'+ext)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest: raise ValueError('unexpected input: '+str(path))
        shutil.copyfile(path,output/('Figure_S26.'+ext))
    cols='panel row metric comparison window level region variant unit n_sid n_events n_estimable estimate ci_lower ci_upper'.split()
    with statistics.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)!=9: raise ValueError('expected nine integral estimates')
    with (output/'Figure_S26_source.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-dir',type=Path,required=True);p.add_argument('--statistics',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();export(a.source_dir,a.statistics,a.out)
