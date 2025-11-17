# function print study
# create by gather
# create time 2025/11/11
# /Users/gather/nltk_data

# nltk.download('punkt')
# nltk.download('punkt_tab')
# res = nltk.find('.')
# print(res)

# word_tokenize函数将文本分割成单词标记
# 参数: 'hello world' - 需要分词的文本字符串
# 返回: ['hello', 'world'] - 分词后的列表
# 基本用法
# tokens = nltk.word_tokenize("Hello world!\nHow are you?")
# print(tokens)  # 输出: ['Hello', 'world', '!']
from nltk.tokenize import sent_tokenize
import string

# 句子切分
paragraph = "At the beginning of the entire article, we first point out the vastness of our ignorance through the example of an ordinary urbanite walking in the countryside, and then further point out that it is our poor observation that leads to our ignorance. Because we are accustomed to these natural phenomena and have never paid attention to observation, we cannot distinguish them. The first paragraph of the article always describes the current situation of our ignorance, and then describes the benefits of ignorance in different paragraphs."
sentences = sent_tokenize(paragraph)
print(len(sentences))
# 单词切分
from nltk.tokenize import word_tokenize

tokens_txt = word_tokenize(paragraph)
print(tokens_txt)
# 停用词，认为是没意义的噪音单词
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
print(len(stop_words))
tokens_txt = [word for word in tokens_txt if word not in stop_words]  # 去掉停用词
tokens_txt = [word for word in tokens_txt if word not in string.punctuation]  # 去掉标点符号
print(tokens_txt)

# 词频提取,单词出现的次数
from nltk.probability import FreqDist
word_freqs = FreqDist([word.lower() for word in tokens_txt])
print(dict(word_freqs))
word_freqs.plot()
