from torch.utils.data import Dataset, DataLoader
import pandas as pd
from THULAC import Word_splitter_list
import torch

def load_dictionary(file_path):
    dictionary = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()  # 去除行末尾的空白字符
            if line:
                word, index = line.split('\t')
                dictionary[word] = int(index)
    return dictionary

dictionary_file = 'word2id.txt'
word2id = load_dictionary(dictionary_file)

def load_dictionary_from_txt(file_path):
    dictionary = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()  # 去除行末尾的空白字符
            if line:
                key, value = line.split(':')
                words = value.strip().split()
                dictionary[key.strip()] = words
    return dictionary


def collate_fn(batch):
    # 找到最大的样本长度
    max_length = 50
    
    # 对齐样本大小，将较长的样本截断或填充到最大长度
    padded_batch = []
    labels = []
    keywords = []
    for data in batch:
        sequence = data[0]
        keyword = data[1]
        label = data[2]
        
        # 截断或填充样本序列
        if len(sequence) > max_length:
            padded_sequence = sequence[:max_length]
        else:
            padded_sequence = sequence + [0] * (max_length - len(sequence))
        
        # 截断或填充关键词序列
        if len(keyword) > 5:
            padded_keyword = keyword[:5]
        else:
            padded_keyword = keyword + [0] * (5 - len(keyword))
        
        padded_batch.append(padded_sequence)
        labels.append(label)
        keywords.append(padded_keyword)
    
    return torch.tensor(padded_batch, dtype=torch.float32), torch.tensor(keywords, dtype=torch.long), torch.tensor(labels, dtype=torch.float32)


class Comment(Dataset):
    def __init__(self, train=True):
        file = 'dataset.csv' if train else 'testset.csv'
        keyfile = 'train_key.txt' if train else 'test_key.txt'
        key_dict = load_dictionary_from_txt(keyfile)
        # 读取CSV文件
        data = pd.read_csv(file, encoding='ANSI')
        self.data = []
        
        # 提取所需的字段
        self.labels = data["label"]
        movie_names = data["电影名"]
        reviews = data["短评"]
        genre = data["类型"]
        
        # 将字段存储为类的属性
        data = pd.concat([movie_names, reviews, genre], axis=1)
        data = data.apply(lambda x: str(x[0]) + str(x[1]) + str(x[2]), axis=1)
        data = Word_splitter_list(data)
        
        for i, sequence in enumerate(data):
            words = sequence.strip().split()
            sequence_ids = []
            keyword_ids = []
            for word in words:
                id = word2id.get(word, 0)
                sequence_ids.append(id)
            keywords = key_dict[str(i+1)]
            for keyword in keywords:
                id = word2id.get(keyword, 0)
                keyword_ids.append(id)
            self.data.append((sequence_ids, keyword_ids))

    
    def __getitem__(self, index):
        # 根据索引返回相应的数据组合
        return self.data[index][0], self.data[index][1], self.labels[index]  # 返回序列、关键词和标签
    
    def __len__(self):
        # 返回数据集的长度
        return len(self.data)



    