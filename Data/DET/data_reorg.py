# reorganize the data into a single file

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    subs = np.arange(1,17)
    sessions = np.arange(1,4)

    datadir = 'orig_datafiles'

    df_all = None

    for sub in subs:
        for session in sessions:
            filename = f'SUB{sub}.O{session}'
            filepath = os.path.join(datadir, filename)
            df = pd.read_csv(filepath, sep='\t', header=None)
            df.columns = ['stimulus', 'condition', 'rt1', 'resp1', 'rt2', 'resp2',
                            'rt3', 'resp3', 'rt4', 'resp4', 'rt5', 'resp5', 'enter_rt', 'enter_resp', 'unknown']

            df['subject'] = sub
            df['session'] = session

            if df_all is None:
                df_all = df
            else:
                df_all = pd.concat([df_all, df])
    
    df_all.to_csv('det_expt1_all.csv', index=False)