import torch
from torch import optim
from torch.utils import data
import numpy as np
from .train_helpers import normalize, get_loss_weights

from .models import SupervisedBaseline

def train_supervised_baseline(epochs_train, epochs_test, n_epochs=20):
  X_train = normalize(epochs_train.get_data())
  
  y_train = epochs_train.events[:, 2] - 1 # start at 0

  X_test = normalize(epochs_test.get_data())
  y_test = epochs_test.events[:, 2] - 1
  
  n_classes = y_train.max() - y_train.min() + 1
  C = X_train.shape[1] # num channels
  T = X_train.shape[2] # window length
  loss_weights = get_loss_weights(epochs_train)
  model = SupervisedBaseline(C, T, n_classes, loss_weights).cuda()

  train_dataset = data.TensorDataset(torch.tensor(X_train).unsqueeze(1), torch.tensor(y_train))
  train_loader = data.DataLoader(train_dataset, batch_size=256, shuffle=True)

  test_dataset = data.TensorDataset(torch.tensor(X_test).unsqueeze(1), torch.tensor(y_test))
  test_loader = data.DataLoader(test_dataset, batch_size=256, shuffle=True)

  train_losses, test_losses = _train_epochs(model, train_loader, test_loader, 
                                         dict(epochs=n_epochs, lr=1e-3))

  return train_losses, test_losses, model

def _train_epochs(model, train_loader, test_loader, train_args):
  epochs, lr = train_args['epochs'], train_args['lr']
  optimizer = optim.Adam(model.parameters(), lr=lr)

  train_losses = []
  test_losses = [_eval_loss(model, test_loader)]
  for epoch in range(1, epochs+1):
    model.train()
    train_losses.extend(_train(model, train_loader, optimizer, epoch))
    test_loss = _eval_loss(model, test_loader)
    test_losses.append(test_loss)
    print(f'Epoch {epoch}, Test loss {test_loss:.4f}')
    
		# save model every 10 epochs
    if epoch % 10 == 0:
      torch.save(model.state_dict(), 'saved_models/supervised_baseline_model_epoch{}.h'.format(epoch))
  return train_losses, test_losses

def _train(model, train_loader, optimizer, epoch):
  model.train()
  
  train_losses = []
  for pair in train_loader:
    x, y = pair[0], pair[1]
    x = x.cuda().float().contiguous()
    y = y.cuda().long().contiguous()
    loss = model.loss(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
  return train_losses

def _eval_loss(model, data_loader):
  model.eval()
  total_loss = 0
  with torch.no_grad():
    for pair in data_loader:
      x, y = pair[0], pair[1]
      x = x.cuda().float().contiguous()
      y = y.cuda().long().contiguous()
      loss = model.loss(x, y)
      total_loss += loss * x.shape[0]
    avg_loss = total_loss / len(data_loader.dataset)

  return avg_loss.item()
