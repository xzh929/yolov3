from net import TestNet
from dataset import Protein_dataset
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
import torch
import os

train_dataset = Protein_dataset(r"F:\pet\fake_train")
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)

real_train_dataset = Protein_dataset(r"F:\pet\real_train")
real_train_loader = DataLoader(real_train_dataset, batch_size=20, shuffle=True)

test_dataset = Protein_dataset(r"F:\pet\fake_test")
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=True)

net = TestNet().cuda()
opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
loss_func = nn.CrossEntropyLoss()

module_path = "module/protein.pth"


def train():
    if os.path.exists(module_path):
        net.load_state_dict(torch.load(module_path))
        print("load module")
    else:
        print("no module")
    init_module = 0.
    for epoch in range(30):
        train_score_sum = 0.
        train_loss_sum = 0.
        net.train()
        for i, (data, tag) in enumerate(real_train_loader):
            train_data = torch.permute(data, (0, 2, 1))
            train_data, train_tag = train_data.cuda(), tag.cuda()
            train_out = net(train_data)
            train_loss = loss_func(train_out, train_tag)

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            train_score = torch.mean(torch.eq(torch.argmax(train_out, dim=1), train_tag).float())
            train_score_sum += train_score.item()
            train_loss_sum += train_loss.item()
        avg_train_loss = train_loss_sum / len(real_train_loader)
        avg_train_score = train_score_sum / len(real_train_loader)
        print("epoch:{0} train_loss:{1} train_score:{2}".format(epoch, avg_train_loss, avg_train_score))

        test_score_sum = 0.
        test_loss_sum = 0.
        net.eval()
        for i, (data, tag) in enumerate(test_loader):
            test_data = torch.permute(data, (0, 2, 1))
            test_data, test_tag = test_data.cuda(), tag.cuda()
            test_out = net(test_data)
            test_loss = loss_func(test_out, test_tag)

            test_score = torch.mean(torch.eq(torch.argmax(test_out, dim=1), test_tag).float())
            test_score_sum += test_score.item()
            test_loss_sum += test_loss.item()
        avg_test_loss = test_loss_sum / len(test_loader)
        avg_test_score = test_score_sum / len(test_loader)
        print("epoch:{0} test_loss:{1} test_score:{2}".format(epoch, avg_test_loss, avg_test_score))
        # if avg_test_score > init_module:
        #     init_module = avg_test_score
        torch.save(net.state_dict(), module_path)
        print("save success")


if __name__ == '__main__':
    train()
