import pandas as pd
from THULAC import Word_splitter_list
# 读取 comment.csv 文件

dataset = pd.read_csv('dataset.csv',encoding='ANSI')
testset = pd.read_csv('testset.csv',encoding='ANSI')

comments_text_1 = dataset['短评'].astype(str).tolist()
comments_text_2 = testset['短评'].astype(str).tolist()

# 读取 movie.csv 文件，假设包含电影名、上映年份、评分、评价人数、导演、编剧、主演、类型、制片国家/地区、语言、上映日期和片长的字段
movies = pd.read_csv('movie.csv', encoding='ANSI')
movies_names = movies['电影名'].tolist()
movies_years = movies['上映年份'].astype(str).tolist()
movies_ratings = movies['评分'].astype(str).tolist()
movies_reviews = movies['评价人数'].astype(str).tolist()
movies_directors = movies['导演'].tolist()
movies_writers = movies['编剧'].tolist()
movies_actors = movies['主演'].tolist()
movies_genres = movies['类型'].tolist()
movies_countries = movies['制片国家/地区'].tolist()
movies_languages = movies['语言'].tolist()
movies_release_dates = movies['上映日期'].astype(str).tolist()
movies_durations = movies['片长'].astype(str).tolist()

# 将评论内容、电影名称、上映年份、评分、评价人数、导演、编剧、主演、类型、制片国家/地区、语言、上映日期和片长合并为一个文本列表
text_data = comments_text_1 + comments_text_2 + movies_names + movies_years + movies_ratings + movies_reviews + movies_directors + movies_writers + movies_actors + movies_genres + movies_countries + movies_languages + movies_release_dates + movies_durations

# 构建 word2id 字典
word2id = {'_PAD_': 0}
current_id = 0

words = Word_splitter_list(text_data)
print(words)

# 遍历文本列表，构建 word2id 字典

for item in words:
    k = item.split()
    for word in k:
        if word not in word2id:
            word2id[word] = current_id
            current_id += 1

# 打印 word2id 字典
with open('word2id.txt', 'w', encoding='utf-8') as f:
    for w in word2id:
        f.write(w+'\t')
        f.write(str(word2id[w]))
        f.write('\n')

