import os
import os.path as op
import numpy as np

AGE_SLEEP_RECORDS = op.join(op.dirname(__file__), 'age_records.csv')
DATASET_PATH = '/home/raphael_hotter/datasets'

def _fetch_one(fname):
    destination = op.join(DATASET_PATH, fname)
    return destination

def fetch_data(subjects, recording=[1, 2]):  # noqa: D301
    records = np.loadtxt(AGE_SLEEP_RECORDS,
                         skiprows=1,
                         delimiter=',',
                         usecols=(0, 1, 2, 6, 7),
                         dtype={'names': ('subject', 'record', 'type', 'sha',
                                          'fname'),
                                'formats': ('<i2', 'i1', '<S9', 'S40', '<S22')}
                         )
    psg_records = records[np.where(records['type'] == b'PSG')]
    hyp_records = records[np.where(records['type'] == b'Hypnogram')]

    fnames = []
    for subject in subjects:
        for idx in np.where(psg_records['subject'] == subject)[0]:
            if psg_records['record'][idx] in recording:
                psg_fname = _fetch_one(psg_records['fname'][idx].decode())
                hyp_fname = _fetch_one(hyp_records['fname'][idx].decode())
                fnames.append([psg_fname, hyp_fname])

    return fnames