import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size, n_classes = 5, 3
x = torch.randn(batch_size, n_classes)
print(x.shape)

target = torch.randint(n_classes, size=(batch_size,), dtype=torch.long)

def softmax(x):
    return x.exp() / (x.exp().sum(-1)).unsqueeze(-1)

def nl(input, target):
    return -input[range(target.shape[0]), target].log().mean()

pred = softmax(x)
loss=nl(pred, target)
loss

asds = 3

