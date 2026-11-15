"""Record local input metadata for direct-omega extraction and cache readback."""
import json
import h5py
import numpy as np
from configuration import OUTPUT_ROOT as ROOT, MWT_ROOT, era5_path, initialize_output


def serializable(value):
    if isinstance(value, bytes):return value.decode('utf-8',errors='replace')
    if isinstance(value,np.ndarray):return [serializable(x) for x in value]
    if isinstance(value,np.generic):return serializable(value.item())
    return value


def preflight_inputs():
    initialize_output()
    inventory=json.loads((MWT_ROOT/'reports/data_era5_inventory.json').read_text(encoding='utf-8'))
    lookup={(x['field'],x['level_hpa']):x for x in inventory}
    paths={}
    for level in [400,500,600,700,850,925]:
        for field in ['u','v','t','z','omega']:
            path=era5_path(level=level) if field=='omega' and level not in [500,700,850] else era5_path(lookup[(field,level)]['path'])
            paths[path]=(field,level)
    rows=[]
    for path,(field,level) in paths.items():
        st=path.stat()
        with h5py.File(path,'r') as data:
            variable='w' if field=='omega' else field
            array=data[variable]
            units=serializable(array.attrs.get('units',''))
            if field=='omega' and units not in ['Pa s**-1','Pa s-1','Pa s^-1']:raise ValueError('Unexpected pressure-velocity units: '+str(units))
            variables={name:{'shape':list(v.shape),'dtype':str(v.dtype),'attributes':{k:serializable(x) for k,x in v.attrs.items() if k not in ['DIMENSION_LIST','REFERENCE_LIST']}} for name,v in data.items() if isinstance(v,h5py.Dataset)}
            coords={}
            for name in ['time','latitude','longitude']:
                v=data[name]
                coords[name]={'n':len(v),'first':serializable(v[:3]),'last':serializable(v[-3:])}
            rows.append({'path':str(path),'field':field,'level_hpa':level,'size':st.st_size,'mtime_ns':st.st_mtime_ns,'variables':variables,'coordinates':coords})
    output=ROOT/'inputs_manifest/NC_METADATA.json'
    output.write_text(json.dumps(rows,indent=2),encoding='utf-8')
    return output


if __name__=='__main__':print(preflight_inputs())
