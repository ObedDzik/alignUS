import json 
import glob
from warnings import warn
import os 
import pandas as pd


def get_metadata_table(dir):
    file_mapping_suffixes = {
        'rf': 'rf.npy',
        'needle_bm': 'needle_bm.npy',
        'needle': 'needle.npy',
        'prostate': 'prostate.npy',
        'info': 'info.json',
    }

    def get_prefix(file):
        return "_".join(file.split('_')[:2])

    rows = []
    for file in sorted([p for p in os.listdir(dir) if p.endswith('.json')]):
        data = json.load(open(os.path.join(dir, file)))
        prefix = get_prefix(file)

        for name, suffix in file_mapping_suffixes.items():
            fpath = os.path.join(dir, f'{prefix}_{suffix}')
            if not os.path.exists(fpath):
                warn(f'File not found: {fpath}')
                fpath = None
            data[f'{name}_path'] = fpath

        rows.append(data)

    t = pd.DataFrame(rows)
    t = t.set_index('id').sort_index()

    return t


if __name__ == '__main__': 
    t = get_metadata_table(os.environ['BK_QUEENS_RAW_DATA_DIR'])
    breakpoint()