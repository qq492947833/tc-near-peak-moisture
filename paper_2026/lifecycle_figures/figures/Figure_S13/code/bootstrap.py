"""Task-process-only library initialization using native Windows OS identity.

This host's Python platform.machine/system can hang in Windows WMI. Use the
actual native architecture and verified sys.platform instead. No installed
library, global environment, OS setting or scientific parameter is changed.
"""
from pathlib import Path
import ctypes,os,platform,sys

def apply(output_dir=None):
    root=Path(__file__).resolve().parents[1]
    out=Path(output_dir).resolve() if output_dir else root/'temp'
    # Output location is selected explicitly by the researcher.
    for name in ('temp','cache','cache/matplotlib'):(out/name).mkdir(parents=True,exist_ok=True)
    os.environ.update(TEMP=str(out/'temp'),TMP=str(out/'temp'),TMPDIR=str(out/'temp'),
        MPLCONFIGDIR=str(out/'cache/matplotlib'),XDG_CACHE_HOME=str(out/'cache'),
        PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    sys.dont_write_bytecode=True
    vendor=str(root/'environment/vendor')
    if vendor not in sys.path:sys.path.insert(0,vendor)
    native={'sys_platform':sys.platform,'python':sys.version,'vendor':vendor}
    if sys.platform=='win32':
        class SYSTEM_INFO(ctypes.Structure):
            _fields_=[('architecture',ctypes.c_ushort),('reserved',ctypes.c_ushort),
              ('page_size',ctypes.c_ulong),('minimum_address',ctypes.c_void_p),
              ('maximum_address',ctypes.c_void_p),('active_mask',ctypes.c_size_t),
              ('processor_count',ctypes.c_ulong),('processor_type',ctypes.c_ulong),
              ('allocation_granularity',ctypes.c_ulong),('processor_level',ctypes.c_ushort),
              ('processor_revision',ctypes.c_ushort)]
        info=SYSTEM_INFO();ctypes.windll.kernel32.GetNativeSystemInfo(ctypes.byref(info))
        machine={0:'x86',9:'AMD64',12:'ARM64'}.get(info.architecture)
        if machine is None:raise RuntimeError('Unsupported native architecture, no guessed identity')
        platform.system=lambda:'Windows'
        platform.machine=lambda:machine
        native.update(native_architecture_id=info.architecture,native_machine=machine,
          windows_version=list(sys.getwindowsversion()),metadata_policy='process-only platform.system/machine use verified native values; no WMI')
    return native
