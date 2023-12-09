# reorganize the data into a single file

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path


def accuracy(row):
    """Calculate accuracy for a given row"""
    correct = 0
    for location in range(5):
        if row[f'resp{location+1}'] == int(str(row['stimulus'])[location]):
            correct += 1
    if correct == 5:
        return 1
    else:
        return 0

if __name__ == "__main__":
    subs = np.arange(1,17)
    sessions = np.arange(1,4)
    nblocks = 12 # blocks per session
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
            df['block'] = np.repeat(np.arange(1, nblocks + 1), 48)

            if df_all is None:
                df_all = df
            else:
                df_all = pd.concat([df_all, df])
    
    df_all = df_all.reset_index()
    del df_all['unknown']
    del df_all['enter_resp']

    # get accuracy
    df_all['correct'] = [accuracy(df_all.loc[i, :]) for i in df_all.index]
    df_all.to_csv('det_expt1_all.csv', index=False)