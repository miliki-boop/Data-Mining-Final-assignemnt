from gensim.models import Word2Vec
from gensim.models.word2vec import Text8Corpus
import pandas as pd

file = 'dataset.csv'
data = pd.read_csv(file,encoding='ANSI')

movie_names = data["电影名"]
reviews = data["短评"]
genre = data["类型"]
data = pd.concat([movie_names, reviews, genre], axis=1)
text_list_1 = data.apply(lambda x: str(x[0]) + str(x[1]) + str(x[2]), axis=1)

file = 'testset.csv'
data = pd.read_csv(file,encoding='ANSI')

movie_names = data["电影名"]
reviews = data["短评"]
genre = data["类型"]
data = pd.concat([movie_names, reviews, genre], axis=1)
text_list_2 = data.apply(lambda x: str(x[0]) + str(x[1]) + str(x[2]), axis=1)

text_list = text_list_1 + text_list_2
text_list.fillna('', inplace=True)  # 处理缺失值
text_list = list(text_list)

model = Word2Vec(text_list, vector_size=512, window=5, min_count=5, workers=4)
model.save("word2vec_model.bin")