import numpy as np
import random
import torch
from torch import optim
from torch.utils import data

from .train_helpers import normalize_one
from .models import CPC_EEG
import os
import os.path as op

root = op.dirname(__file__)

def train_cpc_eeg(train_data, test_data, model, n_context_windows, n_predict_windows,
n_negatives, overlap, sampling_freq, window_length, predict_delay, batch_size=128, lr=1e-3, load_last_saved_model=False):
	C = 2 # num channels
	T = int(window_length*sampling_freq) # window length (in samples)

	sampler = SSL_Window_Sampler(n_context_windows, n_predict_windows, n_negatives, overlap, sampling_freq, window_length, predict_delay, batch_size)

	model = CPC_EEG(C, T, n_context_windows, n_predict_windows, n_negatives).cuda()

	if load_last_saved_model:
		model.load_state_dict(torch.load(op.join(root, 'saved_models', 'cpc_eeg_model.pt')))


	train_losses, test_losses = _train_epochs(model, train_data, test_data, sampler,
																			dict(epochs=n_epochs, batch_size=batch_size))

	return train_losses, test_losses, model

class SSL_Window_Sampler():
	def __init__(self, n_context_windows=8, n_predict_windows=4, n_negatives=4, overlap=0.5, sampling_freq=100, window_length=30, predict_delay=60, batch_size=128):
		self.n_context_windows = n_context_windows
		self.n_predict_windows = n_predict_windows
		self.n_negatives = n_negatives
		self.batch_size = batch_size
		self.window_length = window_length
		self.predict_delay = predict_delay
		self.overlap = overlap
		self.sampling_freq = sampling_freq

	def sample_negatives(self, recording, start_sample, sample_length):
		n_available_positions = recording.shape[1] - sample_length - 2*self.window_length
		random_indices = np.random.choice(n_available_positions, self.n_negatives)
		negative_samples = []
		for i in random_indices:
			if i < start_sample - self.window_length:
				negative_samples.append(recording[:, i:i+self.window_length])
			else:
				idx = i + self.window_length + sample_length
				negative_samples.append(recording[:, idx:idx+self.window_length])
		return negative_samples
	
	def get_minibatch(self, train_data):
		"""
		Return list has [{
			context windows: [[],[],...],
			prediction windows: [[],[],...]
			negative windows: [[],[],...]
		}]
		"""
		# sample bs subjects with replacement
		context_time = (1 + (self.n_context_windows-1)*self.overlap)*self.window_length
		predict_time = (1 + (self.n_predict_windows-1)*self.overlap)*self.window_length
		
		sample_length = self.sampling_freq*(context_time + self.predict_delay + self.predict_time)
		
		subjects = random.choices(train_data, k=self.batch_size)
		minibatch = []
		for s in subjects:
			s_length = s.shape[1]
			start_position = np.random.randint(0, s_length-sample_length)
			context_window_start_times = np.arange(start_position,
																						start_position + context_time*self.sampling_freq - overlap*self.sampling_freq*self.window_length,
																						self.overlap*self.sampling_freq*self.window_length)
			predict_window_start_times = np.arange(start_position + self.sampling_freq*context_time + self.sampling_freq*self.predict_delay,
																						start_position + self.sampling_freq*context_time + self.sampling_freq*self.predict_delay + self.sampling_freq*predict_time - overlap*self.sampling_freq*self.window_length,
																						overlap*self.sampling_freq*self.window_length)
			context_windows = [s[:,int(c_time):int(c_time)+self.sampling_freq*self.window_length] for c_time in context_window_start_times]
			predict_windows = [s[:,int(p_time):int(p_time)+self.sampling_freq*self.window_length] for p_time in predict_window_start_times]
			negative_windows = [self.sample_negatives(s, int(start_position), int(sample_length), int(n_negatives)) for i in range(len(predict_windows))]

			minibatch.append({
				"context_windows": [normalize_one(c) for c in context_windows],
				"predict_windows": [normalize_one(c) for c in predict_windows],
				"negative_windows": [normalize_one(c) for vec in negative_windows for c in vec]
			})
		
		return minibatch

def _train_epochs(model, train_data, test_data, sampler, train_args):
	epochs, lr, batch_size = train_args['epochs'], train_args['lr'], train_args['batch_size']
	optimizer = optim.Adam(model.parameters(), lr=lr)

	saved_models_dir = op.join(root, 'saved_models')
	if not os.path.exists(saved_models_dir):
		os.makedirs(saved_models_dir)

	train_losses = []
	test_losses = [_eval_loss(model, test_data, sampler)]
	for epoch in range(1, epochs + 1):
		model.train()
		train_losses.extend(_train(model, train_data, optimizer, epoch, sampler))
		test_loss = _eval_loss(model, test_data, sampler)
		test_losses.append(test_loss)
		print(f'Epoch {epoch}, Test loss {test_loss:.4f}')
		
		# save model every 10 epochs
		if epoch % 10 == 0:
			torch.save(model.state_dict(), op.join(root, 'saved_models', 'cpc_eeg_model_epoch{}.pt'.format(epoch)))
		torch.save(model.state_dict(), op.join(root, 'saved_models', 'cpc_eeg_model.pt'))

	return train_losses, test_losses

def _train(model, train_data, optimizer, epoch, sampler):
	model.train()
	
	train_losses = []
	for i in range(10):
		minibatch = sampler.get_minibatch(train_data)
		loss = model.forward(minibatch)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		train_losses.append(loss.item())
	return train_losses

def _eval_loss(model, test_data, sampler):
	model.eval()
	total_loss = 0
	with torch.no_grad():
		for i in range(2):
			minibatch = sampler.get_minibatch(test_data)
			loss = model.forward(minibatch)
			total_loss += loss * sampler.batch_size
		avg_loss = total_loss / (2*sampler.batch_size)

	return avg_loss.item()