from snownlp import SnowNLP
import jieba.posseg as pseg
import pandas as pd


output_file = 'test_key.txt'

def split_long_phrases(text):
    words = pseg.cut(text)
    phrases = []
    phrase = ''
    for word, tag in words:
        if len(word) == 1 or tag.startswith('a') or tag.startswith('v') or tag.startswith('d'):
            if phrase:
                phrases.append(phrase)
                phrase = ''
            phrases.append(word)
        else:
            phrase += word
    if phrase:
        phrases.append(phrase)
    return phrases

def extract_emotion_keywords(text_list):
    for idx, text in enumerate(text_list, start=1):
        phrases = split_long_phrases(str(text))
        s = SnowNLP(' '.join(phrases))
        keywords = []
        for phrase in phrases:
            sentiment = s.sentiments
            keywords.append((phrase, sentiment))
        keywords = [(word, sentiment) for word, sentiment in keywords if len(word) > 1]
        keywords.sort(key=lambda x: x[1], reverse=True)
        new_list = [item[0] for item in keywords[:5]]
        while len(new_list) < 5:  # 补全关键词至5个
            new_list.append('_PAD_')
        with open(output_file, 'a', encoding='utf-8') as f:    
            line = f"{idx}: {' '.join(new_list)}\n"
            f.write(line)
        
    print("关键词提取完成！")
    return None

file = 'testset.csv'
data = pd.read_csv(file, encoding='ANSI')
reviews = data["短评"]
extract_emotion_keywords(reviews)
