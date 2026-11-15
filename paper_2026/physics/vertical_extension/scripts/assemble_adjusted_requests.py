"""Collect verified saved requests for the six background-adjusted targets."""
from pathlib import Path
import hashlib
import json
import pandas as pd
from configuration import OUTPUT_ROOT, initialize_output
from raw_statistics import load_frozen_templates


def assemble_requests(root=OUTPUT_ROOT):
    root = Path(root).resolve()
    initialize_output(root)
    sids = sorted(load_frozen_templates()['events'].sid.unique())
    metrics = ['C_A_to_K'] + [f'Gamma_{energy}_{part}' for energy in ['K', 'A'] for part in ['h', 'p', 'total']]
    rows = []
    for sid in sids:
        path = root / 'diagnostics' / sid / 'request_values.parquet'
        receipt = json.loads((path.parent / 'COMPLETE.json').read_text(encoding='utf-8'))
        if receipt.get('sid') != sid or receipt.get('status') != 'COMPLETE' or hashlib.sha256(path.read_bytes()).hexdigest() != receipt['output_sha256']:
            raise ValueError('Unverified request file: ' + str(path))
        if receipt.get('derivation_version') != 2 or receipt.get('mask_schema_version') != 3:
            raise ValueError('Expected derivation 2 / mask schema 3')
        data = pd.read_parquet(path)
        data = data[data.variant.eq('primary') & data.level.eq('500-850') & data.region.eq('r0_300') & data.window.isin([0,1]) & data.metric.isin(metrics)]
        rows.append(data)
    result = pd.concat(rows, ignore_index=True)
    output = root / 'statistics/adjusted_requests.parquet'
    result.to_parquet(output, index=False)
    return output


if __name__ == '__main__': print(assemble_requests())
