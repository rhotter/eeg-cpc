import torch
import numpy as np
from torch import nn
from feature_extractor import EEG_FeatureExtractor

class CPC_EEG(nn.Module):
  def __init__(self, C, T, n_context, n_predict, n_negatives, embedding_dim=100, k=50, m=13, dropout_prob=0.5, n_spatial_filters=8):
    super().__init__()
    self.n_context = n_context
    self.n_predict = n_predict
    self.n_negatives = n_negatives
    self.C = C
    self.T = T
    self.feature_extractor = EEG_FeatureExtractor(C, T, k, m, dropout_prob, embedding_dim, n_spatial_filters)
    self.cross_entropy = nn.CrossEntropyLoss()

  def forward(self, minibatch):
    """
    for each sample:
      encode context windows, prediction windows, negative windows
      aggregate context windows
      compute distances between (1) aggregated context windows and prediction windows encoding
                                (2) aggregated context windows and negative windows encoding
      do softmax with distances
      1 prediction distance and n_neg negative distance
      [prediction, neg1 distance, neg2 distance, , , ]
      
    """
    total_loss = 0
    for recording_dict in minibatch:
      recording_tensor = self.dict_to_tensor(recording_dict)
      embeddings = self.feature_extractor(recording_tensor)
      aggregated_context = self.aggregate_context_embeddings(embeddings[:self.n_context])
      distances = self.compute_distances(aggregated_context, embeddings[self.n_context:])
      loss = self.cross_entropy(distances, torch.zeros(len(distances)).cuda().long())
      total_loss += loss
    return total_loss/len(minibatch)
  
  def compute_distances(self, aggregated_context, comparison_embeddings):
    """
    Returns (n_predict, n_negatives+1) matrix
    """
    distances_vect = torch.matmul(comparison_embeddings, aggregated_context)
    distances = torch.cat((distances_vect[:self.n_predict].unsqueeze(dim=1),
                 distances_vect[self.n_predict:].reshape(self.n_predict, self.n_negatives)), dim=1)
    return distances
  
  def aggregate_context_embeddings(self, context_embeddings):
    return context_embeddings.mean(dim=0)
  
  def dict_to_tensor(self, minibatch_dict):
    tensor = np.zeros((self.n_context + self.n_predict + self.n_predict*self.n_negatives, 1, self.C, self.T))
    tensor[0:self.n_context] = np.array(minibatch_dict["context_windows"]).reshape(-1,1,self.C,self.T)
    tensor[self.n_context:self.n_context+self.n_predict] = np.array(minibatch_dict["predict_windows"]).reshape(-1,1,self.C,self.T)
    tensor[self.n_context+self.n_predict:] = np.array(minibatch_dict["negative_windows"]).reshape(-1,1,self.C,self.T)
    
    return torch.from_numpy(tensor).cuda().contiguous().float()