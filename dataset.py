from torch.utils.data import Dataset
from torchvision import transforms as t
import torch
import os
import numpy as np

amino_acid_str = "GAVLIPFYWSTCMNQDEKRHX"

"""
    先将原始氨基酸序列编码，然后遍历将要编码序列的元素，再用原始序列find该元素的索引，
得到该元素的one-hot编码,最后返回编码后的二维ndarry
"""


def amino_one_hot(encode_data):  # 对序列进行one-hot编码
    amino_acid_encode = []
    encode = []
    for i, amino_acid in enumerate(amino_acid_str):
        zero = np.zeros(21, dtype=np.int32)
        zero[i] = 1
        amino_acid_encode.append(zero)
    amino_acid_encode.append(np.zeros(21, dtype=np.int32))
    amino_acid_encode = np.stack(amino_acid_encode)  # 对原始序列进行编码
    amino_acid_str1 = amino_acid_str + "0"
    for i in encode_data:
        index = amino_acid_str1.find(i)
        i = amino_acid_encode[index]
        encode.append(i)
    return np.stack(encode)


class Protein_dataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.data = []
        self.tag = []
        self.length_list = []
        self.transforms = t.Compose([t.ToTensor()])
        for filename in os.listdir(path):
            file = open(os.path.join(self.path, filename), encoding="utf-8")
            # print(filename)
            # 判断文件类型得到对应标签
            if filename == "fake_negative.txt" or filename == "real_negative.txt":
                tag = 0
            else:
                tag = 1
            for line in file.readlines():
                length = len(line)
                self.data.append(line)
                self.tag.append(tag)
                self.length_list.append(length)
            file.close()
        # print(self.data)
        # print(self.tag)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        protein_data = self.data[item]
        protein_tag = self.tag[item]
        max_len = max(self.length_list)
        if len(protein_data) < max_len:  # 用0补齐序列为最大长度
            extend_len = max_len - len(protein_data)
            for i in range(extend_len):
                protein_data = protein_data + "0"
        protein_data = torch.Tensor(amino_one_hot(protein_data))
        protein_tag = torch.tensor(protein_tag)
        # protein_data.unsqueeze(dim=0)
        return protein_data, protein_tag


if __name__ == '__main__':
    # train_dataset = Protein_dataset(r"F:\pet\fake_train")
    # test_dataset = Protein_dataset(r"F:\pet\fake_test")
    real_dataset = Protein_dataset(r"F:\pet\real_test")
    # print(len(train_dataset))
    print(len(real_dataset))
    # data1, tag1 = train_dataset[90]
    # data2, tag2 = test_dataset[90]
    for i, (data, tag) in enumerate(real_dataset):
        print(data, tag)
    # print(data1, tag1)
    # print(data1.shape, tag1.shape)
    # print(data2.shape, tag2.shape)
