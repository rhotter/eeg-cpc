import os
import os.path as op
import numpy as np
import mne

MAPPING = {'EOG horizontal': 'eog',
	   'Resp oro-nasal': 'misc',
	   'EMG submental': 'misc',
	   'Temp rectal': 'misc',
	   'Event marker': 'misc'}

annotation_desc_2_event_id = {'Sleep stage W': 1,
				'Sleep stage 1': 2,
				'Sleep stage 2': 3,
				'Sleep stage 3': 4,
				'Sleep stage 4': 4,
				'Sleep stage R': 5}
event_id = {'Sleep stage W': 1,
	  'Sleep stage 1': 2,
	  'Sleep stage 2': 3,
	  'Sleep stage 3/4': 4,
	  'Sleep stage R': 5} # unifies stages 3 and 4

def load_labelled_data(subjects, path='/home/raphael_hotter/datasets', recording=[1, 2], filter=False):
  files = _fetch_data(subjects, path, recording)
  epochs = []
  for x in files:
	print("Loading", x[0])
	# load the data
	edf_file = x[0]
	annot_file = x[1]
	raw = mne.io.read_raw_edf(edf_file, verbose='ERROR')
	annot_train = mne.read_annotations(annot_file, verbose='ERROR')

	raw.set_annotations(annot_train, emit_warning=False)
	raw.set_channel_types(MAPPING)
	
	if filter:
	  raw.load_data()
	  raw.filter(None, 30., fir_design='firwin') # low pass filter

	# extract epochs
	events_train, _ = mne.events_from_annotations(
	  raw, event_id=annotation_desc_2_event_id, chunk_duration=30., verbose='ERROR')

	tmax = 30. - 1. / raw.info['sfreq']  # tmax in included
	recording_epochs = mne.Epochs(raw=raw, events=events_train,
				event_id=event_id, tmin=0., tmax=tmax, baseline=None, verbose='ERROR')
	epochs.append(recording_epochs)
  
  epochs = mne.concatenate_epochs(epochs)
  epochs.pick_types(eeg=True, verbose='ERROR') # only keep EEG channels
  return epochs
  return epochs
  return epochs

def load_unlabelled_data(subjects, path='/home/raphael_hotter/datasets', recording=[1, 2]):
  files = fetch_data(subjects, path, recording)
  data = []
  for x in files:
	print("Loading", x[0])
	# load the data
	edf_file = x[0]
	raw = mne.io.read_raw_edf(edf_file, verbose='ERROR')

	raw.set_channel_types(MAPPING)
	
	# filter
	raw.load_data()
	raw.filter(None, 30., fir_design='firwin') # low pass filter
	raw.pick_types(eeg=True, verbose='ERROR') # only keep EEG channels

	data.append(raw.get_data())
  return data

def _fetch_data(subjects, path, recording):  # noqa: D301
  def _fetch_one(fname):
	destination = op.join(path, fname)
	return destination	
  
  records = np.loadtxt(op.join(op.dirname(__file__), 'age_records.csv'),
			 skiprows=1,
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