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


class TestNet2(nn.Module):
    def __init__(self):
        super(TestNet2, self).__init__()
        self.down_layer1 = nn.Sequential(
            nn.Conv1d(21, 16, 3, 1, 1, bias=False),  # 800
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2)  # 400
        )
        self.down_layer2 = nn.Sequential(
            nn.Conv1d(16, 16, 3, 1, 1, bias=False),  # 400
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2)  # 200
        )
        self.mid_layer = nn.Sequential(
            nn.Conv1d(16, 32, 3, 2, 1, bias=False),  # 100
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.up_layer1 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, 3, 2, 1, output_padding=1, bias=False),  # 200
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.up_layer2 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, 3, 2, 1, output_padding=1, bias=False),  # 400
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.up_layer3 = nn.Sequential(
            nn.ConvTranspose1d(32, 1, 3, 2, 1, output_padding=1, bias=False),  # 800
            nn.BatchNorm1d(1),
            nn.ReLU()
        )

    def forward(self, x):
        down1 = self.down_layer1(x)
        down2 = self.down_layer2(down1)
        mid = self.mid_layer(down2)
        up1 = self.up_layer1(mid)
        up2 = self.up_layer2(torch.cat((down2, up1), dim=1))
        up3 = self.up_layer3(torch.cat((down1, up2), dim=1))
        return up3


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.mask = TestNet2()
        self.classify = TestNet()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        mask = self.mask(x)
        mask = self.softmax(mask)
        feature_seq = x * mask
        classify = self.classify(feature_seq)
        return mask, classify


if __name__ == '__main__':
    x = torch.randn(2, 21, 800)
    net = FeatureNet()
    # net1 = TestNet2()
    mask, classify = net(x)
    mask = mask.squeeze()
    # out = net1(x)
    print(mask)
    print(mask.shape, classify.shape)
    # print(out.shape)
