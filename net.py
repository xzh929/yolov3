from torch import nn
import torch

"""
    使用conv1d构建网络，输入为元素编码维度，输出为卷积核个数（feature map个数），
卷积核尺寸为（ksize*in_channel），网络输入尺寸为(N,C（编码维度）,L（序列长度）)。定义4个网络层，每层用不同
尺寸卷积进行操作，每层最后用自适应最大池化将输出尺寸归化为1。将4个网络输出在通道上拼接（cat），
最后用全连接层将输出形状转为NV。
"""

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(21, 10, 3, bias=False),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(21, 10, 4, bias=False),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(21, 10, 5, bias=False),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(21, 10, 6, bias=False),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(40 * 1, 2)
        )

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(x)
        out = torch.cat((out1, out2, out3, out4), dim=1)
        out = out.reshape(-1, 40 * 1)
        out = self.fc(out)
        # return out1, out2, out3, out4, out
        return out


if __name__ == '__main__':
    x = torch.randn(2, 21, 742)
    net = TestNet()
    # y1, y2, y3, y4, y5 = net(x)
    # print(y1.shape, y2.shape, y3.shape, y4.shape, y5.shape)
    y = net(x)
    print(y.shape)
