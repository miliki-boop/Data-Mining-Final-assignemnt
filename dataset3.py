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
    max_length = 50
    
    padded_batch = []
    labels = []
    keywords = []
    other_features = []  # 创建11个空列表，用于存储每个特征
    
    for data in batch:
        sequence = data[0]
        keyword = data[1]
        label = data[3]
        other = data[2]  # 获取other_features
        
        if len(sequence) > max_length:
            padded_sequence = sequence[:max_length]
        else:
            padded_sequence = sequence + [0] * (max_length - len(sequence))
        
        if len(keyword) > 5:
            padded_keyword = keyword[:5]
        else:
            padded_keyword = keyword + [0] * (5 - len(keyword))
    
    # 截断或填充其他特征序列        
        if len(other) > 36:
            padded_other = other[:36]
        else:
            padded_other = other + [0] * (36 - len(other))


        padded_batch.append(padded_sequence)
        labels.append(label)
        keywords.append(padded_keyword)
        other_features.append(padded_other)

    return torch.tensor(padded_batch, dtype=torch.float32), torch.tensor(keywords, dtype=torch.long), torch.tensor(other_features, dtype=torch.long), torch.tensor(labels, dtype=torch.float32)

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
        reviews = data["短评"].astype(str)
        
        year = data['上映年份']
        score = data['评分']
        reviews_num = data['评价人数']
        director = data['导演']
        scriptwriter = data['编剧']
        hero = data['主演']
        country = data['制片国家/地区']
        language = data['语言']
        date = data['上映日期']
        time = data['片长']
        genre = data["类型"]
        
        # 将字段存储为类的属性
        data = reviews
        other_features = pd.concat([movie_names, year, score, reviews_num, director, scriptwriter, hero, country, language, date, time, genre], axis=1)
        other_features = other_features.apply(lambda x: ' '.join([str(val) for val in x]), axis=1)
        # print(other_features_ids)
        data = Word_splitter_list(data)
        other_features = Word_splitter_list(other_features)
        data = zip(data, other_features)
        data = list(data)
        
        for i, data_tuple in enumerate(data):
            sequence = data_tuple[0]

            other_features = data_tuple[1]
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
            
            words = other_features.strip().split()
            other_features_ids = []
            for word in words:
                id = word2id.get(word, 0)
                other_features_ids.append(id)
            self.data.append((sequence_ids, keyword_ids, other_features_ids))

    
    def __getitem__(self, index):
        sequence_ids, keyword_ids, other_features = self.data[index]
        return sequence_ids, keyword_ids, other_features, self.labels[index]


    
    def __len__(self):
        # 返回数据集的长度
        return len(self.data)



    