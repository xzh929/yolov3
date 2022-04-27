import os
import random

# 对原始数据集增样
def sample_addition(path, save_path):
    for filename in os.listdir(path):
        length_list = []
        crop_list = []
        file = open(os.path.join(path, filename), encoding="utf-8")
        fake_file = open(os.path.join(save_path, "fake_{}".format(filename)), "w")
        lines = file.readlines()
        print(filename)
        for line in lines:
            line = line.split()
            length = len(line[1])
            # print(length)
            length_list.append(length)  # 存放所有序列长度
        # print(max(length_list))
        # print(min(length_list))
        for line in lines:
            line = line.split()
            rand_crop_length = random.randint(50, min(length_list))  # 计算随机裁剪长度
            l_index = random.randint(0, (len(line[1]) - rand_crop_length))  # 随机裁剪左索引
            r_index = rand_crop_length + l_index  # 随机裁剪右索引
            # print(rand_crop_length, l_index, r_index,len(line[1]))
            crop_line = line[1][l_index:r_index]  # 获取裁剪后的序列
            crop_list.append(crop_line)  # 将裁剪序列存入列表
        print(crop_list)
        # print(max(crop_list), min(crop_list))
        if filename == "negative.txt":  # 将正负样本比例设为3：1
            sample_num = 300
        else:
            sample_num = 100
        for i in range(sample_num):  # 增样
            rand_sample = random.randint(2, 5)
            fake_list = random.sample(crop_list, rand_sample)  # 获取随机的拼接序列
            fake_line = "".join(fake_list)  # 将序列拼接
            fake_file.write("{0}\n".format(fake_line))
            print(fake_line)
            print(len(fake_line))
        file.close()
        fake_file.close()

        # if len(line[1]) < max(length_list):
        #     extend_len = max(length_list) - len(line[1])
        #     for i in range(extend_len):
        #         line[1] = line[1] + "0"
        # print(line[1])
        # print(len(line[1]))

# 对原始数据处理
def sample_real(path, save_path):
    for filename in os.listdir(path):
        file = open(os.path.join(path, filename), encoding="utf-8")
        save_file = open(os.path.join(save_path, "real_{}".format(filename)), "w")
        lines = file.readlines()
        for line in lines:
            line = line.split()
            data_line = line[1]
            save_file.write("{}\n".format(data_line))
        file.close()
        save_file.close()


if __name__ == '__main__':
    train_path = r"F:\pet\train"
    train_fake_path = r"F:\pet\fake_train"
    test_path = r"F:\pet\test"
    test_fake_path = r"F:\pet\fake_test"

    sample_addition(train_path, train_fake_path)
    sample_addition(test_path, test_fake_path)

    # sample_real(r"F:\pet\test", r"F:\pet\real_test")
    # sample_real(r"F:\pet\train", r"F:\pet\real_train")
