from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "我 爱 深度 学习",
    "我 爱 机器 学习",
    "深度 学习 是 人工智能 的 基础"
]

# 修改参数以确保所有词汇都被包括
vectorizer = TfidfVectorizer(min_df=1, max_df=1.0)  # 包括所有出现过的词
X = vectorizer.fit_transform(corpus)
get_feature_names_out = vectorizer.get_feature_names_out()

print("词汇表:")
print(get_feature_names_out)  # 词汇表
print(f"词汇总数: {len(get_feature_names_out)}")
print("TF-IDF矩阵:")
print(X.toarray())  # 每篇文档的TF-IDF向量

# 显示每篇文档的详细信息
print("\n详细分析:")
feature_names = vectorizer.get_feature_names_out()
for i, doc in enumerate(corpus):
    print(f"文档 {i+1}: {doc}")
    print("TF-IDF值:")
    for j, feature in enumerate(feature_names):
        if X[i, j] > 0:
            print(f"  {feature}: {X[i, j]:.4f}")
    print()