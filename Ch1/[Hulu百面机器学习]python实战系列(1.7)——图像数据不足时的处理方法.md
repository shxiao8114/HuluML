上一章在研究了词袋模型以后，这次的模型上下文的神经网络，就更有挑战性。

**<Font color=Magenta>Q:Word2Vec 是如何工作的？它和LDA有什么区别与联系？(难度：3⭐)</Font>**

CBOW是根据上下文出现的词语来预测当前词的生成概率，Skip-gram是根据当前词来预测上下文各个词的生成概率。

其中$w(t)$是当前关注的词，$w(t-N),w(t-N+1),...,w(t+N-1),w(t+N)$，其中前后滑动窗口大小均设为N。

它们都由输入层、映射层(其他神经网络叫隐藏层)和输出层构成构成的神经网络。

Word2Vec和LDA的区别和联系，首先LDA是按照主题来聚类的，即"文档-单词"矩阵进行分解，得到"文档-主题"和"主题-单词"两个概率分布。而Word2Vec其实是对"上下文-单词"的矩阵进行学习，其上下文由周围简单单词组成。主题模型通过"上下文-单词"主题矩阵进行推理。同理，词嵌入也可用"文档-单词"矩阵学习出词的隐含向量表示。

接下来就实战CBOW和skip-gram两种算法吧！

\
CBOW篇
```Python
import re
import numpy as np
import matplotlib.pyplot as plt
```
\
找个句子
```Python
#%%
sentences = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells."""

# 去掉特殊字符
sentences = re.sub('[^A-Za-z0-9]+', ' ', sentences)

# 去除单字母单词
sentences = re.sub(r'(?:^| )\w(?:$| )', ' ', sentences).strip()

# 改成全部小写
sentences = sentences.lower()
```
\
整理成词汇：
```Python
words = sentences.split()
vocab = set(words)

vocab_size = len(vocab)
embed_dim = 10
context_size = 2

#%%
# 字典词汇
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}
```
\
整理成上下文单词和本单词构成的矩阵：
```Python
def create_windows(context_size):
    data = []
    for i in range(context_size, len(words) - context_size):
        context = [words[i - j] for j in range(-context_size,context_size+1)]
        context.pop(context_size)
        target = words[i]
        data.append((context, target))
    print(data[:10])
    return data

data = create_windows(context_size)
embeddings =  np.random.random_sample((vocab_size, embed_dim))
```
\
定义损失函数:
```Python
def linear(m, theta):
    return np.dot(m,theta)

# Log softmax + NLLloss = Cross Entropy
def log_softmax(x):
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / (e_x.sum() + 0.001))

def NLLLoss(logs, targets):
    out = logs[range(len(targets)), targets]
    return -out.sum()/len(out)

def log_softmax_crossentropy_with_logits(logits,target):

    out = np.zeros_like(logits)
    out[np.arange(len(logits)),target] = 1
    
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
    
    return (- out + softmax) / logits.shape[0]
```
\
设立正向和反向函数：

反向函数是用log_softmax_crossentropy_with_logits(logits,target)的：
```Python
def forward(context_idxs, theta):
    m = embeddings[context_idxs].reshape(1, -1)
    n = linear(m, theta)
    o = log_softmax(n)
    
    return m, n, o

# 通过反向函数去求梯度
def backward(preds, theta, target_idxs):
    m, n, o = preds
    
    dlog = log_softmax_crossentropy_with_logits(n, target_idxs)
    dw = m.T.dot(dlog)
    
    return dw
```
\
优化函数
```Python
# 优化函数
def optimize(theta, grad, lr=0.03):
    theta -= grad * lr
    return theta

# 训练函数
theta = np.random.uniform(-1, 1, (2 * context_size * embed_dim, vocab_size))
epoch_losses = {}

for epoch in range(100):

    losses =  []

    for context, target in data:
        context_idxs = np.array([word_to_ix[w] for w in context])
        preds = forward(context_idxs, theta)

        target_idxs = np.array([word_to_ix[target]])
        loss = NLLLoss(preds[-1], target_idxs)

        losses.append(loss)

        grad = backward(preds, theta, target_idxs)
        theta = optimize(theta, grad, lr=0.03)
        
     
    epoch_losses[epoch] = losses
    if(epoch % 5 == 0):
        print("Cost after epoch {}: {}".format(epoch, epoch_losses[epoch]))
```
\
作图一个新函数：
```Python
# Plot loss/epoch
ix = np.arange(0,80)

fig = plt.figure()
fig.suptitle('Epoch/Losses', fontsize=20)
plt.plot(ix,[epoch_losses[i][0] for i in ix])
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Losses', fontsize=12)
```
\
某个随机结果如下：

![](https://imgkr2.cn-bj.ufileos.com/3494bb93-226f-473d-b2b3-6d3b12635ac9.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=tkqsANt7WGB4W7SHXNwpEF17Lpw%253D&Expires=1605450111)

\
预测并算出精确度：
```Python
# 预测
def predict(words):
    context_idxs = np.array([word_to_ix[w] for w in words])
    preds = forward(context_idxs, theta)
    word = ix_to_word[np.argmax(preds[-1])]
    
    return word

# 评估模型的准确度
def accuracy():
    wrong = 0

    for context, target in data:
        if(predict(context) != target):
            wrong += 1
            
    return (1 - (wrong / len(data)))
```
\
发现竟然为1！是不是很神奇啊？其实不然，这其实因为生成的测试集里面都源于训练集，非常依赖经验主义。这很容易导致过拟合啊，一旦换篇文章就不行了啊。所以解决办法之一是，快速扩充文章数量和句子数量。至于生僻的语境譬如develop作为动词洗胶卷的意思的话，要靠机器学习判断还是自求多福了。

至于上一期有一位童鞋说到的Sentence2Vec和Paragraph2Vec，其实原理很简单，就是把
```Python
words = sentences.split()
```
改成
```Python
sentence = sentences.split(".")
```
这样就ok了。段落向量也是一样的规则。

\
还有一个skip-gram算法(从CBOW生成的词袋words开始)：
```Python
def generate_training_data(words, word_to_id, window_size):
    X, Y = [], []

    for i in range(len(words)):
        nbr_inds = list(range(max(0, i - window_size), i)) + \
                   list(range(i + 1, min(len(words), i + window_size + 1)))
        for j in nbr_inds:
            X.append(word_to_id[words[i]])
            Y.append(word_to_id[words[j]])
            
    return np.array(X), np.array(Y)

def expand_dims(x, y):
    x = np.expand_dims(x, axis=0)
    y = np.expand_dims(y, axis=0)
    return x,y

x, y = generate_training_data(words, word_to_ix, 3)
x, y = expand_dims(x, y)
```
\
然后就是前后向传播和softmax激活函数：
```Python
# 前向传播
def init_parameters(vocab_size, emb_size):
    wrd_emb = np.random.randn(vocab_size, emb_size) * 0.01
    w = np.random.randn(vocab_size, emb_size) * 0.01
    
    return wrd_emb, w

def softmax(z):
    return np.divide(np.exp(z), np.sum(np.exp(z), axis=0, keepdims=True) + 0.001)


def forward(inds, params):
    wrd_emb, w = params
    word_vec = wrd_emb[inds.flatten(), :].T
    z = np.dot(w, word_vec)
    out = softmax(z)
    
    cache = inds, word_vec, w, z
    
    return out, cache

# 垮熵损失函数：
def cross_entropy(y, y_hat):
    m = y.shape[1]
    cost = -(1 / m) * np.sum(np.sum(y_hat * np.log(y + 0.001), axis=0, keepdims=True), axis=1)
    return cost

# softmax的求导
def dsoftmax(y, out):
    return out - y

# 找到损失函数关于z,w,word_vec的偏导
def backward(y, out, cache):
    inds, word_vec, w, z = cache
    wrd_emb, w = params    
    dl_dz = dsoftmax(y, out)
    # deviding by the word_vec length to find the average
    dl_dw = (1/word_vec.shape[1]) * np.dot(dl_dz, word_vec.T)
    # 
    dl_dword_vec = np.dot(w.T, dl_dz)    
    return dl_dz, dl_dw, dl_dword_vec

def update(params, cache, grads, lr=0.03):
    inds, word_vec, w, z = cache
    wrd_emb, w = params
    dl_dz, dl_dw, dl_dword_vec = grads
    
    wrd_emb[inds.flatten(), :] -= dl_dword_vec.T * lr
    w -= dl_dw * lr
    
    return wrd_emb, w
```
\
训练测试词袋(根据单词猜出上下文)：
```Python
vocab_size = len(ix_to_word)

m = y.shape[1]
y_one_hot = np.zeros((vocab_size, m))
y_one_hot[y.flatten(), np.arange(m)] = 1

y = y_one_hot

batch_size = 256
embed_size = 50

params = init_parameters(vocab_size, 50)

costs = []

for epoch in range(5000):
    epoch_cost = 0
    
    batch_inds = list(range(0, x.shape[1], batch_size))
    np.random.shuffle(batch_inds)
    
    for i in batch_inds:
        x_batch = x[:, i:i+batch_size]
        y_batch = y[:, i:i+batch_size]
        
        pred, cache = forward(x_batch, params)
        grads = backward(y_batch, pred, cache)
        params = update(params, cache, grads, 0.03)
        cost = cross_entropy(pred, y_batch)
        
        epoch_cost += np.squeeze(cost)
        
    costs.append(epoch_cost)
    
    if(epoch % 250 == 0):
        print("Cost after epoch {}: {}".format(epoch, epoch_cost))  
```
\
测试集：
```Python
#%% Test
x_test = np.arange(vocab_size)
x_test = np.expand_dims(x_test, axis=0)
softmax_test, _ = forward(x_test, params)
top_sorted_inds = np.argsort(softmax_test, axis=0)[-4:,:]
for input_ind in range(vocab_size):
    input_word = ix_to_word[input_ind]
    output_words = [ix_to_word[output_ind] for output_ind in top_sorted_inds[::-1, input_ind]]
    print("{}'s skip-grams: {}".format(input_word, output_words))
```
\
这里就没有展示它的准确率了，测试结果如下：
```note
they's skip-grams: ['manipulate', 'evolve', 'inhabit', 'as']
programs's skip-grams: ['processes', 'to', 'that', 'direct']
things's skip-grams: ['the', 'we', 'conjure', 'in']
spells's skip-grams: ['computer', 'our', 'with', 'we']
people's skip-grams: ['to', 'programs', 'program', 'create']
about's skip-grams: ['the', 'we', 'conjure', 'in']
manipulate's skip-grams: ['things', 'they', 'evolve', 'other']
that's skip-grams: ['as', 'beings', 'computers', 'inhabit']
is's skip-grams: ['pattern', 'evolution', 'directed', 'by']
called's skip-grams: ['create', 'of', 'people', 'rules']
by's skip-grams: ['rules', 'of', 'directed', 'pattern']
evolution's skip-grams: ['is', 'data', 'processes', 'of']
program's skip-grams: ['programs', 'rules', 'people', 'create']
are's skip-grams: ['that', 'beings', 'computational', 'process']
pattern's skip-grams: ['called', 'by', 'rules', 'is']
effect's skip-grams: ['the', 'we', 'conjure', 'in']
spirits's skip-grams: ['computer', 'the', 'conjure', 'we']
computers's skip-grams: ['evolve', 'they', 'as', 'beings']
inhabit's skip-grams: ['they', 'as', 'computers', 'that']
directed's skip-grams: ['of', 'the', 'in', 'spirits']
in's skip-grams: ['conjure', 'processes', 'direct', 'we']
evolve's skip-grams: ['other', 'they', 'manipulate', 'computers']
other's skip-grams: ['called', 'by', 'rules', 'is']
study's skip-grams: ['of', 'the', 'in', 'spirits']
of's skip-grams: ['the', 'with', 'computer', 'conjure']
computer's skip-grams: ['spells', 'the', 'our', 'with']
beings's skip-grams: ['computers', 'are', 'abstract', 'that']
as's skip-grams: ['processes', 'to', 'that', 'direct']
to's skip-grams: ['in', 'direct', 'processes', 'programs']
process's skip-grams: ['by', 'directed', 'computational', 'evolution']
with's skip-grams: ['spells', 'the', 'our', 'with']
we's skip-grams: ['spirits', 'conjure', 'effect', 'to']
abstract's skip-grams: ['data', 'things', 'other', 'manipulate']
processes's skip-grams: ['we', 'to', 'direct', 'effect']
conjure's skip-grams: ['of', 'the', 'in', 'spirits']
our's skip-grams: ['spells', 'the', 'our', 'with']
the's skip-grams: ['spirits', 'of', 'the', 'our']
direct's skip-grams: ['effect', 'in', 'create', 'to']
idea's skip-grams: ['process', 'computational', 'study', 'effect']
rules's skip-grams: ['people', 'by', 'pattern', 'program']
data's skip-grams: ['of', 'the', 'in', 'spirits']
computational's skip-grams: ['abstract', 'process', 'are', 'processes']
create's skip-grams: ['direct', 'programs', 'called', 'people']
```
看样子这里的准确率不是100%，毕竟1推多比多推1难得多嘛！

fasttext是脸书推出来的源码，有兴趣的点击下方链接：https://github.com/facebookresearch/fastText/

我们的第一章很快就要结束了，现在冷不丁的来个神经网络，才是个特征工程就要搞到这种程度。。。于是我就借用外网的代码来充数了。大家看看第三章机器学习的线性回归的numpy源码就知道这个项目是有多难，而且fasttext肯定比CBOW和skip-gram都更加复杂。

<br><br>
**百面机器学习系列，免费的时日已经不多了！感兴趣想要加粉丝群的欢迎来加微信号：xiaoshida002，非诚勿扰！**
