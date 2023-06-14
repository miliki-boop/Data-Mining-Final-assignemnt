import thulac


def Word_splitter_list(text_list):
    # 初始化THULAC分词器
    thu = thulac.thulac(seg_only=True)

    results = []
    for text in text_list:
        # 使用THULAC进行分词和词性标注
        result = thu.cut(text, text=True)
        results.append(result)
    
    return results

# def Word_splitter(text):
#     # 初始化THULAC分词器
#     thu = thulac.thulac(seg_only=True)


#     result = thu.cut(text, text=True)
    
#     return result

# text = "我喜欢这部电影"
# result = Word_splitter(text)

# text_list = ["我喜欢这部电影", "这个产品很好用", "这本书非常有趣"]
# tokenized_texts = Word_splitter(text_list)

# 输出分词结果
# for tokens in tokenized_texts:
#     print(tokens)
