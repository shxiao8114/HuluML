本章主要讲模型准确度评估

**<font color = Magenta>Q1: 准确率的局限性？(1⭐)</font>**

准确率的定义是预报类别等于观测/实际类别的概率，即公式：

<font size =5><center>$Accuracy = \frac{n_{correct}}{n_{total}}$</font></center>

举个空气质量的例子，预报值有6个等级，分别为优、良、轻微污染、轻度污染、中度污染、重污染，观测值也一样。如果有那么一个$6 \times 6$矩阵$A_{6 \times 6}$来摆放那么准确率经过6变n的归纳推广就是：

<font size =5><center>$Accuracy = \frac{tr(A_{n \times n})}{n_{total}}$</font></center>

其中$A_{ij}$指的是观测空气质量等级为$i$，预报空气质量等级为$j$的实例个数。

<br>

```Python
import numpy as np

n = 6
A = np.zeros((n,n))

# 然后构造confusion matrix
for i in range(n):
    for j in range(n):
        A[i,j] = "观测空气质量等级为i，预报空气质量等级为j的实例个数"
accuracy = np.trace(A) / np.sum(A)
```

<br><br>
当正负样本严重失衡的时候，准确率取决于样本占比大者(如99%)的准确率。不适用于样本比例严重失衡的情况。你可以全预测为同一个样本种类而得到很高的准确率。

<br><br>
**<font color = Magenta>Q2: 精确率和召回率之间的制衡？(1⭐)</font>**

Hulu 提供视频的模糊搜索功能，搜索排序模型返回的 Top 的精
确率非常高，但在实际使用过程中 用户还是经常找不到想要的视频，
特别是一些比较冷门的剧集，这可能是哪个环节出了问题呢？

<br>

| 观测值\预测值| 正 |   负 |
| :----- | :--: | -----: |
| 正     |  TP  |     FN |
| 负     |  FP  |     TN |

\
这里得引入精确率Precision和召回率Recall，其公式为：

<font size =5><center>$Precision = \frac{TP}{TP+FP}$</center></font>

<font size =5><center>$Recall = \frac{TP}{TP+FN}$</center></font>

这对指标是难以同时兼顾的，因为精确率代表着保守，召回率代表着激进。

```Python
n = 2
A = np.zeros((n,n))

# 然后构造confusion matrix
for i in range(n):
    for j in range(n):
        A[i,j] = "观测值为i，预报值为j的实例个数"

precision = A[0,0] / (A[0,0] + A[0,1])
recall = A[0,0] / (A[0,0] + A[1,0])
```

在这个具体的问题中，可以使用topN名次范围内分别计算准确率和召回率曲线，这样就能够绘制出一条精确-召回率曲线反映预报的整体情况。

另外还有一种指标叫F1Score，其公式为：

<font size =5><center>$F1Score = \frac{2 \times Precision \times Recall}{Precision + Recall}$</center></font>

<br><br>
**<font color = Magenta>Q3: 平方根误差的意外？(1⭐)</font>**

我们知道均方误差的公式长这样：

<font size =5><center>$MSE = \frac{1}{n} \sum_{i=1}^{n}|x_{i} - \hat{x}_{i}|^{2}$</center></font>

当均方误差在95%信度区间内预测准确的情况下，均方误差仍然居高不下，很可能是因为outliers造成更高的均方误差贡献。因此有一种损失函数可以替代均方误差，即平均绝对值误差：

<font size =5><center>$MAE = \frac{1}{n} \sum_{i=1}^{n}|x_{i} - \hat{x}_{i}|$</center></font>

甚至推广一下到绝对值的$n$次幂误差的平均值，如果想压低outlier的影响，就把$n$在正数的情况下调小些，想增大对outlier的惩罚就把n调大。

```Python
def mean_square_error(y_predict,y_observation):
    return np.sum((y_predict - y_observation)**2)/len(y_predict)

def mean_absolute_error(y_predict,y_observation):
    return np.sum(np.abs(y_predict - y_observation))/len(y_predict)

def mean_error_of_n_power_difference(y_predict,y_observation,power):
    return np.sum(np.abs(y_predict - y_observation)**power)/len(y_predict)
```

下一期敬请期待：
**[Hulu百面机器学习]python实战系列(2.2)——ROC曲线**
