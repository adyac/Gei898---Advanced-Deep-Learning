import numpy as np
import matplotlib.pyplot as plt
import torch
from Network import *
from Helpers import *
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

# batch_size = 10
# transform = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Lambda(lambda x: x.view(-1))  # 28*28 = 784
# ])
# training_data = torchvision.datasets.MNIST(root='./data/',
#                                         train=True,
#                                         download=False,
#                                         transform=transform)
# trainloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=0)
#
# testing_data = torchvision.datasets.MNIST(root='./data/',
#                                         train=False,
#                                         download=False,
#                                         transform=transform)
# testloader = torch.utils.data.DataLoader(testing_data, batch_size=batch_size, shuffle=False, num_workers=1)

batch_size = 256

train_data = np.loadtxt("./dataset/shuttle.trn", dtype=np.float32)
test_data = np.loadtxt("./dataset/shuttle.tst", dtype=np.float32)

# Split inputs / targets
train_inputs  = train_data[:, :-1]
train_targets = train_data[:, -1].astype(np.int64)

# Filter class 1
valid_train_idx    = np.where(train_targets == 1)[0]
valid_train_inputs = train_inputs[valid_train_idx]
valid_train_inputs, mu, sigma = normalize(valid_train_inputs)
test_inputs = (test_data[:, :-1] - mu) / sigma
valid_train_targets = train_targets[valid_train_idx]

# Convert to tensors
valid_train_inputs  = torch.from_numpy(valid_train_inputs)
valid_train_targets = torch.from_numpy(valid_train_targets).long()
test_inputs  = torch.from_numpy(test_inputs).float()
test_targets = torch.from_numpy(test_data[:, -1]).long()

valid_train_dataset = TensorDataset(valid_train_inputs, valid_train_targets)
valid_test_dataset  = TensorDataset(test_inputs, test_targets)

trainloader = torch.utils.data.DataLoader(valid_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(valid_test_dataset, batch_size=1, shuffle=False, num_workers=0)

# matrice de covariance
X = valid_train_inputs
C = torch.cov(X.T)
mu = X.mean(dim=0)
# valeurs et vecteurs propres
eigvals, eigvecs = torch.linalg.eigh(C)
print(f"eigenvalues : \n{eigvals}\neigenvectors : \n{eigvecs}")