本文是ICML 2017最佳论文《Understanding Black-box Predictions via Influence Functions》的个人解读。

整体是按照论文行文顺序写的。

# 1. Introduction
提出观点：好的模型不仅仅是表现得好，还要可解释。

解释性指的是我们可以了解模型在干什么，怎样得到最终结果的。模型的解释性可以有助于优化模型。

但是目前在很多领域表现优异的模型大多为深度神经网络，例如其在图像和语音识别方面表现能力显著。这些模型复杂、难以解释。

## 为什么使用影响函数？
原本要观察训练数据的影响，需要做的是将训练数据删除或者细微修改，然后重新训练。但是这个过程计算代价非常昂贵。所以提出稳健统计学中的影响函数，将样本的变化直接映射到模型的变化。

## 影响函数自身存在的问题及解决方法
影响函数Influence function需要昂贵的二阶导数计算，并假设模型的可微性和凸性，这限制了它们在现代环境中的适用性，因为实际应用中模型通常是不可微分的，非凸的和高维的。 

论文证明了可以使用二阶优化技术（Pearlmutter，1994; Martens，2010; Agarwal等，2016）来有效地估计影响函数，并且即使在可微性和凸性的基本假设不成立的情况下，它们仍然是准确可用的。（比较准确的近似）

## Introduction的整体结构
下面将详细介绍每段的主要内容：
1. 提出机器学习中关键的问题:‘模型是如何得出特定的预测结果？’ ，即阐述可解释性的重要性
2. 提出目前存在的问题是表现好的模型如深度学习网络太过复杂、解释性低的问题。提问：“模型到底从何而来？”
3. 提出本文针对“模型从何而来”的问题，回溯到训练数据，分析训练数据的影响
4. 如何衡量训练数据的影响？为避免昂贵的重训练代价，引入影响函数
5. 影响函数存在的问题及解决方法
6. 影响函数的应用

---
# 2. Approach

## 影响函数是什么？
影响函数定义为模型参数的导数：
![Influence function](https://github.com/lidanxu/store-positioning/blob/master/images/Influence_functions.png)

其中海森矩阵 ![Hessian](https://github.com/lidanxu/store-positioning/blob/master/images/Hessian.png)

具体公式含义见论文

## influence function vs Euclidean distance

影响函数可以度量训练样本的改变对测试集中某个测试样本预测结果的影响。（即分析训练数据对测试数据的影响）

那么通过寻找与测试数据相近（根据欧几里得距离）的训练数据，同样可以达到这样的效果。因为一般而言，离测试数据越近的训练数据的影响应该更大一些。

分析可得：
欧几里得可以通过衡量内积
x * x<font size=1>test </font>
来权衡训练数据的影响力.

影响函数与欧几里得两大不同是
1. 影响函数给高训练误差的样本更高的影响力，表明异常点可以支配模型参数
2. 其次，加权协方差矩阵测量其他训练点的“抵抗”，如果∇θL（z，θ）指向一个变化很小的方向，那么它的影响就会更大，因为朝那个方向移动不会显着增加其他训练点上的损失


正因为这两个不同之处，影响函数比最近邻居更准确地捕捉模型训练的效果.

![Figure1](https://github.com/lidanxu/store-positioning/blob/master/images/Figure1.png)

figure1中(a)图分析影响函数的组成元素：`Hessian逆矩阵`和 `training ross` L,比较元素丢失后对影响函数的影响，表明二者缺一不可。

(a) 左图表示如果缺失元素train loss将过大估计很多训练样本点的影响力；中间图表明如果缺失H逆矩阵，则所有正例数据均有利，反例数据均有害。右图两个元素都丢失，则影响函数相当于是放缩后的欧几里得内积，将丧失准确捕捉影响力的能力，而且散点图偏离对角线较远。

(b) 图显示两张相同标记的测试图和有害的训练图片，当这张有害的训练图片用于训练模型时将导致模型错误地判断测试数据，因为测试数据和训练数据非常不一样。使用影响函数可以发现这个问题而欧几里得内积不能发现这个虽然不直观，但重要的有害影响。

-----

# 3. Efficiently Calculating Influence

## 影响函数计算难题
影响函数有两大计算难题，假设影响函数表示如下图：![Iup,loss](https://github.com/lidanxu/store-positioning/blob/master/images/Iup_loss.png)

注意图中L上的横线只是截图残留，没有特殊意义。

第一 ：需要求解Hessian的逆矩阵，当有n个训练样本时，求解Hessian逆矩阵的复杂度约为`O(np^2+p^3)`,其中n是训练样本数，p是模型参数个数。当模型参数个数非常大如深度神经网络时计算代价很高；

第二 ：需要遍历所有的训练样本计算`Iup,loss(zi,ztest)`

注：感觉第一种难题就是一次计算代价大；第二种难题是需要多次计算
## 计算难题解决方法

针对第一个计算难题，可以通过二阶优化(second-order optimization）来解决。主要思想是避免直接计算H逆矩阵，使用implicit Hessian-vector products(HVPs)高效近似Stest.
![Stest](https://github.com/lidanxu/store-positioning/blob/master/images/Stest.png)

这样就可以通过下式计算`Iup,loss`,避免了直接计算H逆矩阵。

这种方法同样可以解决第二个计算难题.

论文中讨论了两种方式来近似计算Stest.两种方法均依赖于Hessian矩阵的HVP。两种方法分别是：
1. Conjugate gradients(CG):共轭梯度，是将矩阵求逆转化为优化问题的标准变换。
2. Stochastic estimation： 当数据量大的时候，标准的CG会变得很慢，因为CG每次迭代都要遍历n个训练样本，第二种方法是2016年提出的，每次迭代只需要一个样本点。计算会快很多。

注:此处计算内容过于复杂，本文省略了具体的计算过程，有需要详见论文。

因此通过这些技术，我们可以将遍历所有训练样本计算`Iup,loss(zi,ztest)`的复杂度控制在`O(np+rtp)`。第四部分将表明选择`rt=O(n)` 将给出准确的结果。

同时论文表明可以使用TensorFlow和Theano来实现 Iup,loss的计算，用户只需要设置损失函数L，其余的工作模块将自动完成。

# 4. Validation and Extensions

本章主要内容主要分为两个部分：
1. 论证在约束条件下，影响函数leave-one-out retraining的渐近逼近
2. 放宽约束条件的情况下，说明影响函数依然有用。

## Influence functions vs. leave-one-out retraining

当以下两个假设成立时，影响函数是leave-one-out retraining的渐近逼近
1. 模型参数θ通过最小化经验风险empirical risk计算得到
2. empirical risk是二阶可微且严格凸函数。

实验论证了influence function与留一法重训练非常相近。如图![Feature2](https://github.com/lidanxu/store-positioning/blob/master/images/Feature2.png)

其中左图使用CG计算Stest;中间图使用stochastic approximation计算Stest;右图使用CNN算法,CNN不凸，不收敛，但是增加阻尼项后，影响函数估计的结果和真实值还是高度相近的（Pearson's R=0.86）。

## Non-convexity and non-convergence 
(非凸性和非收敛性)

当不满足凸约束条件时，影响函数依然可以提供有价值的信息。论文提出的方法是构造凸二次近似，在损失函数中增加阻尼项。这相当于在损失函数中添加L2 regularization.如图Feature2右图所示。

## Non-differentiable losses

当损失函数的可导条件不成立时，可以通过对不可微损失进行光滑逼近计算的影响函数，来预测/估计原始不可微损失进行留一法重训练的结果。就是说可以使用光滑逼近的方法计算原来不可微的损失函数得到影响函数，然后这个影响函数可以预测或者说逼近真实留一法重训练的结果。

这种近似的鲁棒性表明，我们可以训练不可微分的模型，转换不可微分量的平滑版本来计算影响函数。

注：损失函数不可导，那么影响函数的公式无法表示出来，那么采取的方式就是将不可微的损失函数平滑成可微的，再计算得到影响函数。

论文中举的例子是svm算法，这个算法的损失函数是`Hinge(s) = max(0,1-s)`,是不可微的。类似的还有神经网络中的ReLUs。

通过将不可微的svm损失函数`Hinge(s)`平滑化得到`SmoothHinge(s,t)`,利用`SmoothHinge`计算`Iup,loss`可以很好地估计真实代价变化。如图![Feature3](https://github.com/lidanxu/store-positioning/blob/master/images/Feature3.png)

(a)图是当t取值不同时，`SmoothHinge(s)`的变化情况，其中t=0对应`Hinge`,且`t=0.001`和`t=0`图形重叠；

(b)左图表示`t=0`时的影响函数不能准确估计actual diff in loss;中间图 `t=0.001`时，平滑后得到的影响函数实现准确预测；右图`t=0,1`表示相关性在很宽的范围内仍然很高，但当t太大时，相关性会降低。

# 5. Use Cases of Influence Functions
## a. Understanding model behavior
influence function揭示什么样的样本数据对模型影响大，有助于了解模型行为。

比如实验表明rbf svm模型:离测试数据越近的训练数据影响越大，而Inception network 没有这一特点；

Inception network提取数据中比较明显,独特的特征，而RBF SVM更多地是从图片表面的模式匹配。

## b. Adversarial training examples
通过对影响力大的样本增加扰动，可以产生视觉上不可见但是翻转模型预测结果的训练攻击。

细节详见论文。

## c. Debugging domain mismatch

> domain mismatch: where the training distribution does not match the test distribution

如果训练数据和测试数据分布不同，那么可能导致训练误差非常小而测试误差很大。

通过实验获取对错误预测影响力最大的训练样本，观察这些训练样本的分布，有助于模型开发者发现domain mismatch.

## d. Fixing mislabeled examples
在现实世界中，已标记的数据可能损坏。即使一个专家可以识别出错误标记的数据，但是针对不同的数据集，人工检查标记的正确与否工作量太大，也是不切实际的。

论文提出可以借助影响函数帮助专家将他们的注意力放到影响力大的训练样本中，即只检查影响力大的训练样本标记的正确性，这样大大地减少了检查的工作量。

通常在训练模型的过程中，测试数据是未知的，所以可以计算`Iup,loss(zi,zi)`来衡量zi的影响力来估计当zi被删除后可能造成的误差error.
> we measure the influence of zi with Iup,loss(zi
, zi), which approximates the error incurred on zi if we remove zi from the training set.

# 6. Related Work
详见论文

# 7. Discussion
论文提及影响函数依赖于模型只进行微小变化，不能处理模型发生大改动的情况。因此如何估计模型大改动依然是悬而未决的问题。

# 总结
第一章：提出背景及论文工作

第二章：解释什么是影响函数

第三章：提出影响函数的计算难题及解决方案

第四章：验证影响函数的可用性并放宽其使用约束

第五章：影响函数的应用

