"""Portable locations for the frozen 2026 analysis extension."""
from pathlib import Path
import os
PACKAGE=Path(__file__).resolve().parent
ADDENDUM=Path(os.environ.get('FLTCCF_ADDENDUM_DIR',str(PACKAGE/'inputs/addendum'))).expanduser().resolve()
UNIFIED=Path(os.environ.get('FLTCCF_UNIFIED_DIR',str(ADDENDUM/'unified_main_20261023_1642'))).expanduser().resolve()
def bootstrap(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for key in ['TEMP','TMP','TMPDIR','MPLCONFIGDIR','XDG_CACHE_HOME']:
        target=output/key;target.mkdir(parents=True,exist_ok=True);os.environ[key]=str(target)
    return {'vendor': ''}
