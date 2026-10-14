"""Rebuild NPJ main and supplemental figures from packaged source tables."""
import importlib,json,argparse
from . import common
MODULES=['environment_exchange', 'thematic', 'boundary', 'branches', 'supp_sf_a', 'supp_sf_b', 'supp_sf_d', 'supp_sf_e', 'supp_sf_m', 'supp_sf_f', 'supp_sf_l', 'supp_sf_n', 'framework', 'climate', 'spatial', 'physics', 'same_sample', 'recovery', 'geography', 'synthesis']
def render_all(figures=None):
    common.SELECTED=None if figures is None else set(figures)
    common.MANIFEST.clear()
    for name in MODULES:
        module=importlib.import_module('.'+name,__package__)
        module.render()
    common.OUTPUT.mkdir(parents=True,exist_ok=True)
    (common.OUTPUT/'render_manifest.json').write_text(json.dumps(common.MANIFEST,ensure_ascii=False,indent=2),encoding='utf-8')
    return common.MANIFEST
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--figures',nargs='*');args=parser.parse_args();render_all(args.figures)
