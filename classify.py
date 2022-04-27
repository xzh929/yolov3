import os

positive_path = r"F:\pet\Degradable.txt"
negative_path = r"F:\pet\Non_degradable.txt"
train_path = r"F:\pet\train"
test_path = r"F:\pet\test"

train_positive_file = open(os.path.join(train_path, "positive.txt"), "w")
train_negative_file = open(os.path.join(train_path, "negative.txt"), "w")
test_positive_file = open(os.path.join(test_path, "positive.txt"), "w")
test_negative_file = open(os.path.join(test_path, "negative.txt"), "w")

positive_file = open(positive_path, encoding="utf-8")
negative_file = open(negative_path, encoding="utf-8")

positive_lines = positive_file.readlines()
negative_lines = negative_file.readlines()

# 采集正样本
def sample_positive():
    count = 1
    for i, line in enumerate(positive_lines):
        line = line.split()
        length = len(line)
        if length == 0:  # 过滤空行
            continue
        if i < 116:  # 取索引前为训练正样本
            train_positive_file.write("{0} ".format(line[0]))
            if count % 2 == 0:  # 每写入两行换行
                train_positive_file.write("\n")
            print("train", i, line)
            count += 1
        else:  # 取后面的为测试正样本
            test_positive_file.write("{0} ".format(line[0]))
            if count % 2 == 0:
                test_positive_file.write("\n")
            print("test", i, line)
            count += 1
    train_positive_file.close()
    test_positive_file.close()
    positive_file.close()

# 采集负样本
def sample_negative():
    count = 1
    for i, line in enumerate(negative_lines):
        line = line.split()
        length = len(line)
        if length == 0:
            continue
        if i < 390:
            train_negative_file.write("{0} ".format(line[0]))
            if count % 2 == 0:
                train_negative_file.write("\n")
            print("train", i, line)
            count += 1
        else:
            test_negative_file.write("{0} ".format(line[0]))
            if count % 2 == 0:
                test_negative_file.write("\n")
            print("test", i, line)
            count += 1
    train_negative_file.close()
    test_negative_file.close()
    negative_file.close()


if __name__ == '__main__':
    sample_positive()
    sample_negative()
