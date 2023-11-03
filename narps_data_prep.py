# clean up narps behav data

import pandas as pd
import numpy as np
from glob import glob
import os


def parse_bids_filename(fname):
    f_s = os.path.basename(fname).split('.')[0].split('_')
    fdict = {}
    for f in f_s:
        if '-' in f:
            fdict[f.split('-')[0]] = f.split('-')[1]
    return fdict

# load all behav data

basedir = '/data/narps_behav'

files = glob(os.path.join(basedir, 'sub-*/func/*_events.tsv'))
files.sort()

alldata = None

for f in files:
    data = pd.read_csv(f, sep='\t')
    fdict = parse_bids_filename(f)
    data['sub'] = fdict['sub']
    data['run'] = fdict['run']
    if alldata is None:
        alldata = data
    else:
        alldata = pd.concat([alldata, data], axis=0)

del alldata['onset']
del alldata['duration']
alldata = alldata[['sub', 'run',  'gain', 'loss', 'RT', 'participant_response']]
alldata.to_csv('narps_behav_data.csv', index=False)