from sklearn.metrics import roc_auc_score
import torch

a = torch.tensor([
    [[0, 0, 0], [0, 1, 0]],
    [[0, 0, 0], [0, 1, 0]]
])
x = 1
for i in a:
    for seq in i:
        print(x)
        print(seq)
        x += 1
