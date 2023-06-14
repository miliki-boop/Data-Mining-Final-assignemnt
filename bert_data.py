from torch.utils.data import Dataset, DataLoader
import pandas as pd
from THULAC import Word_splitter_list
import torch


class Comment(Dataset):
    def __init__(self,train=True):
        file = 'dataset.csv' if train else 'testset.csv'
        
        # 读取CSV文件
        data = pd.read_csv(file,encoding='ANSI')
        self.data = list()
        # 提取所需的字段
        self.labels = data["label"]
        movie_names = data["电影名"]
        reviews = data["短评"]
        # release_years = data["上映年份"]
        # ratings = data["评分"]
        # review_num = data["评价人数"]
        # director = data["导演"]
        # editor = data["编剧"]
        # leader = data["主演"]
        genre = data["类型"]
        # country = data["制片国家/地区"]
        # language = data["语言"]
        # date = data["上映日期"]
        # leng = data["片长"]
        # 提取其他字段...
        
        # 将字段存储为类的属性
        data = pd.concat([movie_names, reviews, genre], axis=1)
        self.data = data.apply(lambda x: str(x[0]) + str(x[1]) + str(x[2]), axis=1)
        
                
    def __getitem__(self, index):
        # 根据索引返回相应的数据组合
        return self.data[index], self.labels[index]
    
    def __len__(self):
        # 返回数据集的长度
        return len(self.data)


testset = Comment(train=False)
testloader = DataLoader(testset, batch_size=32)


trainset = Comment(train=True)
trainloader = DataLoader(trainset, batch_size=32)  # 创建 DataLoader 对象

    