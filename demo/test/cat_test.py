import torch

a = torch.randn(2, 3)
print(a)
# dim，如果tensor是二维的，dim=0指在行上连接，dim=1指在列上连接
b = torch.cat((a, a, a), dim=0)
print(b)


