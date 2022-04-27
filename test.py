from net import TestNet
from dataset import Protein_dataset
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
import torch
import os

test_dataset = Protein_dataset(r"F:\pet\real_test")
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True)

net = TestNet().cuda()
opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
loss_func = nn.CrossEntropyLoss()

module_path = "module/protein.pth"


def Test():
    if os.path.exists(module_path):
        net.load_state_dict(torch.load(module_path))
        print("load module")
    else:
        print("no module")
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
        print("predict:{} tag:{} score:{}".format(torch.argmax(test_out, dim=1).float(), test_tag, test_score))
    avg_test_loss = test_loss_sum / len(test_loader)
    avg_test_score = test_score_sum / len(test_loader)
    print("test_loss:{0} test_score:{1}".format(avg_test_loss, avg_test_score))


if __name__ == '__main__':
    Test()
