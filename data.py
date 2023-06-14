import pandas as pd
import random

# 读取爬取的数据文件，假设为"comments.csv"，包含电影名称、评论内容和标签（好评或差评）
data = pd.read_csv("comment.csv", encoding='ANSI')
data_alter = pd.read_csv("comment_2.csv", encoding='ANSI')
movie_info = pd.read_csv("movie.csv", encoding='ANSI')
# 随机选择250部电影
selected_movies = data["电影名"].unique()

# 创建空的数据集和测试集
dataset = pd.DataFrame(columns=["电影名"])
testset = pd.DataFrame(columns=["电影名"])
remainingset = pd.DataFrame(columns=["电影名"])
# 构建数据集和测试集
for i,movie in enumerate(selected_movies):
    reviews = data[data["电影名"] == movie]
    positive_reviews = reviews[reviews["label"] == 1 & (reviews["评价"] != "推荐")]
    negative_reviews = reviews[reviews["label"] == 0]
    # print(i,movie,len(positive_reviews),len(negative_reviews))
    
    # 从好评和差评中选择80条和80条作为数据集
    try:
        if len(positive_reviews) < 100:
            additional_positive_reviews = data_alter[(data_alter["电影名"] == movie) & (data_alter["label"] == 1)].sample(n=80-len(positive_reviews), replace=False)
            dataset = pd.concat([dataset, positive_reviews, additional_positive_reviews])
        else:
            dataset = pd.concat([dataset, positive_reviews.sample(n=80, replace=False)])
    except:
        dataset = pd.concat([dataset, positive_reviews.sample(n=min(len(positive_reviews), 60))])

    try:
        if len(negative_reviews) < 80:
            additional_negative_reviews = data_alter[(data_alter["电影名"] == movie) & (data_alter["label"] == 0)].sample(n=80-len(negative_reviews), replace=False)
            dataset = pd.concat([dataset, negative_reviews, additional_negative_reviews])
        else:
            dataset = pd.concat([dataset, negative_reviews.sample(n=80, replace=False)])
    except:
        dataset = pd.concat([dataset, negative_reviews.sample(n=min(len(positive_reviews), 60))])


    # 剩余的好评和差评作为测试集
    remaining_positive_reviews = positive_reviews.drop(dataset.index, errors='ignore')
    remaining_negative_reviews = negative_reviews.drop(dataset.index, errors='ignore')
    print(len(remaining_positive_reviews), len(remaining_negative_reviews))
    try:
        if len(remaining_positive_reviews) < 20:
            additional_positive_reviews = data_alter[(data_alter["电影名"] == movie) & (data_alter["label"] == 1)].sample(n=20-len(remaining_positive_reviews), replace=False)
            testset = pd.concat([testset, remaining_positive_reviews, additional_positive_reviews])
        else:
            testset = pd.concat([testset, remaining_positive_reviews.sample(n=20, replace=False)])
    except ValueError:
        testset = pd.concat([testset, remaining_positive_reviews.sample(n=min(len(remaining_positive_reviews), 10))])

    try:
        if len(remaining_negative_reviews) < 20:
            additional_negative_reviews = data_alter[(data_alter["电影名"] == movie) & (data_alter["label"] == 0)].sample(n=20-len(remaining_negative_reviews), replace=False)
            testset = pd.concat([testset, remaining_negative_reviews, additional_negative_reviews])
        else:
            testset = pd.concat([testset, remaining_negative_reviews.sample(n=20, replace=False)])
    except ValueError:
        testset = pd.concat([testset, remaining_negative_reviews.sample(n=min(len(remaining_negative_reviews), 10))])


    combined_reviews = pd.concat([dataset, testset])
    remaining_positive_reviews = positive_reviews.drop(combined_reviews.index, errors='ignore')
    remaining_negative_reviews = negative_reviews.drop(combined_reviews.index, errors='ignore')
    remaining_reviews = pd.concat([remaining_positive_reviews,remaining_negative_reviews])
    remainingset = pd.concat([remainingset,remaining_reviews])
    
#合并
dataset = pd.merge(dataset, movie_info, on="电影名")
testset = pd.merge(testset, movie_info, on="电影名")   
remainingset = pd.merge(remainingset, movie_info, on="电影名")      
    
# 将数据集和测试集保存为CSV文件
dataset.to_csv("dataset.csv", index=False, encoding='ANSI')
testset.to_csv("testset.csv", index=False, encoding='ANSI')
remainingset.to_csv("remaining.csv",index=False, encoding='ANSI')