import jieba

print(jieba.__version__)

# 可以查看分词表，分词表是固定的。
print(jieba.__path__)
# 词汇 词频 词性
# 分词模式
text = "我爱中华人民共和国"
# 返回迭代器
res = jieba.cut(text)
for word in res:
    print(word)
# 返回类表
res = jieba.lcut(text)
print(res)

# 全模式,所有可能的词
print(jieba.lcut(text, cut_all=True))

# 搜索引擎模式
print(jieba.lcut_for_search(text))

# 词性标注
from jieba import posseg

res = posseg.lcut(text)
print(res)
print(res[0])

text = "马士兵教育是一个线上培训结构"
print(jieba.lcut(text))

# 加载自定义分词文件 词频影响权重大小,词频越大越容易分在一起
# jieba.load_userdict("dict.txt")

jieba.add_word("马士兵教育")
# jieba.suggest_freq("马士兵教育", True)
print(jieba.lcut(text))

# 关键词提取tf-idf 决定
from jieba import analyse

text = "近日，保靖酉水国家湿地公园科研团队在常规监测中，在酉水河峡谷风光带意外邂逅了一位植物界的“稀客”，列入《世界自然保护联盟濒危物种红色名录》的极危物种——红头索。这是该物种在湖南省内的首次记录，标志着这一珍稀植物成功在保靖酉水湿地“安家”。"
res = analyse.extract_tags(text, topK=10, withWeight=True)
print(res)

res = analyse.tfidf(text, topK=10, withWeight=True)
print(res)

# 获取词表的位置
res = jieba.tokenize(text)
for word in res:
    print(word) # ('近日', 0, 2) 起始位置0 结束位置2，不包含结束位置
