"""Portable file bindings; scientific constants remain in their original modules."""
from pathlib import Path
import json
import os

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = Path(os.environ.get('FLTCCF_VERTICAL_CONFIG', PACKAGE_ROOT / 'config.example.json')).resolve()
CONFIG = json.loads(CONFIG_FILE.read_text(encoding='utf-8'))


def configured_path(name):
    value = Path(os.path.expandvars(os.path.expanduser(CONFIG[name])))
    return (value if value.is_absolute() else CONFIG_FILE.parent / value).resolve()


OUTPUT_ROOT = configured_path('output_root')
MWT_ROOT = configured_path('mwt_root')
TEMPLATE_ROOT = configured_path('template_root')
BACKGROUND_ROOT = configured_path('background_root')
RISE_INPUT_ROOT = configured_path('rise_input_root')
ERA5_ROOT = configured_path('era5_root')


def era5_path(value=None, level=None):
    """Resolve the original inventory or an added pressure-velocity input."""
    if value is None:
        return ERA5_ROOT / CONFIG['omega_filename_pattern'].format(level=int(level))
    value = str(value).replace('${RAW_ERA5}', str(ERA5_ROOT))
    path = Path(value)
    return path if path.is_absolute() else ERA5_ROOT / path


def initialize_output(root=OUTPUT_ROOT):
    """Create only the configured run's output and process-local cache folders."""
    root = Path(root).resolve()
    if root == PACKAGE_ROOT or any(root == p or root.is_relative_to(p) for p in [TEMPLATE_ROOT, BACKGROUND_ROOT, RISE_INPUT_ROOT, MWT_ROOT]):
        raise ValueError('Output must be separate from code, input templates, and accepted MWT files')
    for name in ['tmp', 'inputs_manifest', 'diagnostics', 'statistics', 'outputs', 'audit', 'checkpoints']:
        (root / name).mkdir(parents=True, exist_ok=True)
    for name in ['TEMP', 'TMP', 'TMPDIR', 'MPLCONFIGDIR', 'XDG_CACHE_HOME']:
        os.environ[name] = str(root / 'tmp')
    return root
