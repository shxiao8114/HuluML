不像一个充满数字的环境，我们现在进入一个文字的世界了，这些是非结构化的数据。

**<font size=4 color= Magenta>Q: 有哪些文本表示模型？它们各有什么优缺点？ (2⭐)</font>**

- 词袋模型和N-gram模型

每篇文章可以用一袋子词语表示(集合)，忽略此出现的顺序。以此为单位切割开，每篇文章是一个长的向量，向量每一维当成一个单词，该维权重代表这个单词的重要程度。常用TF-IDF来计算权重，公式为：

<center>$TF-IDF(t,d)=TF(t,d) \times ID(t)$</center>

其中$TF(t,d)$为单词$t$在文档$d$中出现的频率，$IDF(t)$ 是逆文档频率，用来衡量单词$t$的重要性，表示为：

#### <center>$ID(t) = log(\frac{文章总数}{含单词t的文章总数+1}) $</center>

直观解释：如果它在多篇文章中出现，那么它可能是一个比较通用的词汇，对于区分语义的贡献就比较小，因此对权重做一定惩罚。

将文章进行单词级数的划分并不好，例如natural language processing把三个词拆开，三个单词各自的意思就和整体意思大相径庭。因此甚至得把连续$n$个词共同的词组($n < N$)也放到词袋里，这样就构成$N$-gram模型。一个词性有多种词性变化，却具有相似含义。实际应用中应当成词干(Word-Stemming)处理，即将不同词性的单词当成同一词干来处理。

为了提高复杂关系的拟合能力，我们时常把两个特征变量的$m$个特征和$n$个特征合在一块组成一个由$mn$个特征组成的向量。这种方法当引入ID类特征的时候，问题就出现了。这时需要引入一个较小的数$k \space (k << m,n)$，来达成降维目的。熟悉推荐算法的同学一眼就能看出是等价于矩阵分解(包括lu分解、qr分解和奇异值分解，详见百面机器学习系列的$(4.x)$)，他们都有各自独特的算法决定$k$值，使得特征数下降到$(m + n)k$，甚至到只有$k$个特征(这时的$k$可能还只是个位数)。
<br><br>
1. 在实战之前先做个铺垫:
```Python
import re
import numpy as np
# 这段文字并不通顺，因为我故意加了几个词来测试代码功能的
s = "Natural-language processing (NLP) is an area of \
computer science and of artificial intelligence \
concerned with the interactions between computers and \
and human (natural) languages for artificial intelligence."
```
其代码实现有简单粗暴的纯Python算法：
```Python
def generate_ngrams(s, n):
    # 转小写
    s = s.lower()
    
    # 把非字母字符串变成空格
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
    
    # 把空格拿掉
    tokens = [token for token in s.split(" ") if token != ""]
    
    # 用 zip 函数去生成 n-grams，然后合并，返回
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

def unique_counts(arr):
    arrs = sorted(arr.copy())
    number = [1]*len(arr)
    j = 0
    diff1 = 0
    for i in range(len(arrs)):
        if (not i) or (arrs[i] not in arrs[0 : j]):
            arrs[j] = arrs[i]
            diff2 = i - j
            # print(diff2," ",diff1)
            if diff2 != diff1:
                number[j-1] += diff2 - diff1
            diff1 = diff2
            j += 1
    return arrs[0 : j], number[0 : j]
```
函数部分设计完毕！
```Python
t = generate_ngrams(s,1)

# 先提前生成ac列表
ac = t.copy()
N = 3

# 将小于等于N个单词的字符串全纳入词袋
for i in range(2,N+1):
    t.extend(generate_ngrams(s,i))
a,num = unique_counts(t)

# 生成随机数来举个栗子
total_essay = 1e9
# n_essay的数量占total_essay的10-100%，随机生成，与具体单词没有关系
n_essay = total_essay - np.random.randint(1,9e8,len(a))

# 计算单词权重
tf1 = np.array(num) / sum(num)
idf = np.log(total_essay/(n_essay+1))
tf_idf = tf1 * idf
```
<br><br>
2. 接下来是成熟的sklearn算法：
```Python

corpus = ['This is the first document.','This is the second second document.',\
          'And the third one.','Is this the first document?',\
              "Natural-language processing (NLP) is an area of \
computer science and of artificial intelligence \
concerned with the interactions between computers and \
and human (natural) languages for artificial intelligence."]

transformer = TfidfTransformer(smooth_idf=False)
transformer
TfidfTransformer(smooth_idf=False)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit_transform(corpus).toarray()
```
<br>
我们用sklearn之前可以先用nltk预处理文本字符串：

```Python
import nltk
def preprocessText(data):
    stemmer = nltk.stem.porter.PorterStemmer()
    preprocessed = []
    for each in data:
        tokens = nltk.word_tokenize(each.lower().translate(xlate))
        filtered = [word for word in tokens if word not in stopwords]
        preprocessed.append([stemmer.stem(item) for item in filtered])
    print(Counter(sum([list(x) for x in preprocessed], [])))
    return np.array(preprocessed)
```
<br><br>

3. 还有gensim算法：
```Python
import gensim.downloader as api
from gensim.models import TfidfModel
from gensim.corpora import Dictionary

dataset = api.load("text8")
dct = Dictionary(dataset)  # fit dictionary
corpus = [dct.doc2bow(line) for line in dataset]  # convert corpus to BoW format

model = TfidfModel(corpus)  # fit model
vector = model[corpus[0]]  # apply model to the first corpus document
```

由于每种算法都在处理不同的文本字符串，这三种算法没法直接比较算法复杂度和空间复杂度。

<br>
- 主题模型：

详见**Hulu百面机器学习(6.5)**
<br><br>
- 词嵌入与深度学习模型

词嵌入是一类将词向量化的模型的统称，核心思想是将每个词都
映射成低维空间(通常K=50 ～ 300 维)上的一个稠密向量(Dense
Vector)。K 维空间的每一维也可以看作一个隐含的主题，只不过不像
主题模型中的主题那样直观。

```Python
# 深度学习特征工程
a1, num1 = unique_counts(ac)
deep = np.zeros((len(ac),len(a1)))
for i in range(len(ac)):
    deep[i,a1.index(ac[i])] = 1
print(deep)
# 每排代表每个单词，每列代表去重后的单词(词的数量为1)
In:[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
```

敬请期待下一期：**[Hulu百面机器学习]python实战系列(1.6)——Word2Vec**

<br><br>
过了这个精彩不断的村就没有这个店啦：
