"""Content identity of the only session arrays read for legacy eligibility."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def eligibility_inputs(root, session_ids):
    folder=(Path(root)/'cache/sessions').resolve()
    result={}
    for sid in sorted(set(session_ids)):
        path=(folder/(sid+'.npz')).resolve()
        if not path.is_relative_to(folder) or path.parent!=folder:
            raise ValueError('unsafe session identity')
        record={'present':path.is_file()}
        if record['present']:
            with np.load(path,allow_pickle=False) as arrays:
                for name in ('t_acc','imu_valid'):
                    data=np.ascontiguousarray(arrays[name])
                    record[name]={'dtype':data.dtype.str,'shape':list(data.shape),
                                  'sha256':hashlib.sha256(data.tobytes()).hexdigest()}
        result[sid]=record
    return result
