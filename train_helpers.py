import numpy as np
import torch

def normalize(x):
  x_normalized = (x - x.mean(2).reshape(x.shape[0],x.shape[1],1))/(x.std(2).reshape(x.shape[0],x.shape[1],1))
  return x_normalized

def normalize_one(x):
  x_normalized = (x - x.mean(1).reshape(x.shape[0],1))/(x.std(1).reshape(x.shape[0],1))
  return x_normalized

def get_loss_weights(epochs_train):
  y_train = epochs_train.events[:, 2] - 1 # start at 0
  _, counts = np.unique(y_train, return_counts=True)
  weights = counts / len(y_train)
  print("Class weights", weights)
  return torch.from_numpy(weights).cuda().float()