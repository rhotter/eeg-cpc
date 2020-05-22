from sklearn.metrics import balanced_accuracy_score, accuracy_score
import torch
from torch import nn
import numpy as np

def get_test_results(model, test_loader):
	y_true = []
	y_pred = []
	model.eval()
	softmax = nn.Softmax()
	with torch.no_grad():
		for pair in test_loader:
			x, y = pair[0], pair[1]
			x = x.cuda().float().contiguous()
			y = y.cuda().long().contiguous()
			out = model(x)
			_, predicted = torch.max(softmax(out.data), 1)
			y_true.extend(list(y.cpu().numpy()))
			y_pred.extend(list(predicted.cpu().numpy()))
	return y_pred, y_true

def scores(model, test_loader):
	y_pred, y_true = get_test_results(model, test_loader)
	print(f'Performance of the network on the {len(test_loader.dataset)} test images:')
	print(f'\tAccuracy: {100*accuracy_score(y_pred, y_true)}%')
	print(f'\tBalanced accuracy: {100*balanced_accuracy_score(y_pred, y_true)}%')
