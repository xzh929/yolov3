from net import FeatureNet
from dataset import Protein_dataset, anti_one_hot
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
import torch
import os
from sklearn import metrics
import matplotlib.pyplot as plt

test_dataset = Protein_dataset(r"F:\pet\real_test")
test_loader = DataLoader(test_dataset, batch_size=45, shuffle=True)

real_train_dataset = Protein_dataset(r"F:\pet\real_train")
real_train_loader = DataLoader(real_train_dataset, batch_size=150, shuffle=True)

net = FeatureNet()
opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
loss_func = nn.CrossEntropyLoss()

module_path = "module/protein2.pth"


def Test():
    if os.path.exists(module_path):
        net.load_state_dict(torch.load(module_path))
        print("load module")
    else:
        print("no module")
    test_score_sum = 0.
    test_loss_sum = 0.
    net.eval()
    for i, (data, tag) in enumerate(real_train_loader):
        test_data = torch.permute(data, (0, 2, 1))
        test_mask, test_out_tag = net(test_data)
        test_loss = loss_func(test_out_tag, tag)

        anti_seq = anti_one_hot(test_data.permute(0, 2, 1))
        for mask, seq, pre in zip(test_mask, anti_seq, test_out_tag):
            idx = torch.argmax(mask)
            pre = torch.argmax(pre).item()
            feature_seq = seq[idx - 5:idx + 5]
            if pre == 1:
                print(feature_seq, pre)

        test_out = nn.Softmax(dim=1)(test_out_tag)
        test_out2 = torch.argmax(test_out, dim=1)
        test_out1 = torch.argmax(test_out, dim=1).detach().cpu().numpy()
        index = torch.nonzero(test_out2)
        index = index.squeeze()
        test_score = torch.mean(torch.eq(torch.argmax(test_out, dim=1), tag).float())
        test_score_sum += test_score.item()
        test_loss_sum += test_loss.item()
        test_tag = tag.detach().cpu().numpy()
        test_out = test_out.detach().cpu().numpy()[:, 1]

        auc = metrics.roc_auc_score(test_tag, test_out)
        recall = metrics.recall_score(test_tag, test_out1)
        accuracy = metrics.accuracy_score(test_tag, test_out1)
        fpr, tpr, threshold = metrics.roc_curve(test_tag, test_out)
        fpr1, tpr1, threshold1 = metrics.precision_recall_curve(test_tag, test_out)
        cm = metrics.confusion_matrix(test_tag, test_out1)
        display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
        plt.subplot(121)
        plt.title("ROC")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.plot(fpr, tpr)
        plt.subplot(122)
        plt.title("Recall")
        plt.xlabel("TPR")
        plt.ylabel("FPR")
        plt.plot(tpr1, fpr1)
        # plt.subplot(121)
        display.plot()
        plt.subplots_adjust(wspace=0.25)
        plt.show()
        print("AUC:{0} Accuracy:{1} Recall:{2}".format(auc, accuracy, recall))


if __name__ == '__main__':
    Test()
