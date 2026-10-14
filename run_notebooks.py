"""Run each notebook in a fresh kernel using this Python environment."""
from pathlib import Path
import argparse,json,os,sys,tempfile,time
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
def run(root, names=None, output=None):
    root=Path(root).absolute()
    output=Path(output or root/'outputs/notebook_runs');output.mkdir(parents=True,exist_ok=True)
    notebooks=sorted((root/'notebooks').glob('*.ipynb'))
    if names:notebooks=[p for p in notebooks if p.name in names]
    assert notebooks
    records=[]
    with tempfile.TemporaryDirectory(prefix='tc_kernel_') as temp:
        kernels=Path(temp)/'kernels';folder=kernels/'tc';folder.mkdir(parents=True)
        (folder/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','--HistoryManager.enabled=False','-f','{connection_file}'],'display_name':'TC reproduction','language':'python'}))
        os.environ['JUPYTER_RUNTIME_DIR']=str(Path(temp)/'runtime')
        os.environ['IPYTHONDIR']=str(Path(temp)/'ipython')
        for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
        for path in notebooks:
            started=time.monotonic()
            km=KernelManager(kernel_name='tc',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
            nb=nbformat.read(path,as_version=4)
            client=NotebookClient(nb,km=km,timeout=1800,resources={'metadata':{'path':str(root/'notebooks')}})
            try:
                client.execute()
            finally:
                if client.kc is not None:client.kc.stop_channels()
                if km.has_kernel:km.shutdown_kernel(now=True)
            nbformat.write(nb,output/path.name)
            records.append({'notebook':path.name,'code_cells':sum(c.cell_type=='code' for c in nb.cells),'seconds':round(time.monotonic()-started,2),'completed':True})
            (output/'execution.json').write_text(json.dumps(records,indent=2))
            print(path.name,'completed',flush=True)
    return records
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=str(Path(__file__).absolute().parent));p.add_argument('--output');p.add_argument('--notebook',action='append')
    a=p.parse_args();run(a.root,a.notebook,a.output)



