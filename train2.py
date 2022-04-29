from net import FeatureNet
from dataset import Protein_dataset
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
import torch
from sklearn import metrics
import os

train_dataset = Protein_dataset(r"F:\pet\fake_train")
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)

test_dataset = Protein_dataset(r"F:\pet\fake_test")
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=True)

real_train_dataset = Protein_dataset(r"F:\pet\real_train")
real_train_loader = DataLoader(real_train_dataset, batch_size=20, shuffle=True)

net = FeatureNet().cuda()
opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
loss_func = nn.CrossEntropyLoss()

module_path = "module/protein2.pth"

def Train():
    if os.path.exists(module_path):
        net.load_state_dict(torch.load(module_path))
        print("load module")
    else:
        print("no module")

    for epoch in range(100):
        sum_train_loss = 0.
        sum_accuracy = 0.
        for i, (data, tag) in enumerate(real_train_loader):
            train_data = torch.permute(data, (0, 2, 1))
            train_data, train_tag = train_data.cuda(), tag.cuda()
            train_mask, train_out_tag = net(train_data)
            train_loss = loss_func(train_out_tag, train_tag)

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            sum_train_loss += train_loss.item()
            pre_tag = torch.argmax(train_out_tag, dim=1)
            pre_tag = pre_tag.detach().cpu().numpy()
            train_tag = train_tag.detach().cpu().numpy()
            accuracy = metrics.accuracy_score(train_tag,pre_tag)
            sum_accuracy += accuracy
        avg_train_loss = sum_train_loss / len(real_train_loader)
        avg_accuracy = sum_accuracy / len(real_train_loader)
        print("train_loss:{0} train_accuracy:{1}".format(avg_train_loss, avg_accuracy))
        torch.save(net.state_dict(), module_path)
        print("save success")


if __name__ == '__main__':
    Train()
