import torch
from functools import reduce

# a = torch.randn(2,583,20)
# a1 = torch.permute(a,(0,2,1))
# print(a1.shape)

a = ["1", "2", "3"]
b = reduce(lambda x, y: x + y, a)
c = "".join(a)
print(b, c)
