from sklearn.metrics import roc_auc_score
import torch
from torch import nn

# a = torch.tensor([
#     [[0., 0, 0], [0, 1, 0]],
#     [[0, 0, 0], [0, 1, 0]]
# ])
list1 = [1,2,3,4,5,6]
a = torch.tensor([0,1,1,0,0,0,1])
index = torch.nonzero(a)
for i in a[index]:
    print(i)

# print(d)
# str1 = "asbd"
# b = str1[1]
# c = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
# zero = torch.zeros(21)
# if c.equal(zero):
#     print("a")
# else:
#     print("b")
# print(zero)
# print(b)
# c,l,v = a.shape
# print(c,l,v)
# for i in range(c):
#     y = 1
#     for x in a[i,:]:
#         print(x)
#         y += 1
