---
layout:    post
title:     Generative models for discrete data
subtitle:  Naive Bayes classifiers
date:      2018-09-14
author:     DonGovi
header-img: img/post-bg-ml.jpeg
catalog:   true
mathjax:   true
tags:      
     Machine-Learning
---


# Generative models for discrete data

本系列主要参考Kevin P. Murphy的书籍，Machine learning: A Probabilistic Perspective。该书从理论角度讲述机器学习算法，内容详尽，逻辑流畅，值得机器学习入门阅读。

## 引言

首先需要讲一下几个概率统计的概念，方便后面理解。

### Bayes Theorem（贝叶斯定理）

贝叶斯定理描述了条件概率$P(A \vert B)$与条件概率$P(B \vert A)$之间的关系，即\\[P(A \vert B)P(B)=P(A,B)=P(B \vert A)P(A).\\]

贝叶斯定理可以用在许多问题中都有应用，例如：一位年龄在40多岁的女性进行乳腺X光检查，假设已知该检查的敏感度(sensitivity)为80%，若检查结果为阳性，该女性患病的概率是多少？设事件$x=1$表示乳腺X光检查结果为阳性，事件$y=1$表示患者确患有乳腺癌，则检查的敏感度为80%可表示为$P(x=1|y=1)=0.8$。此外，该乳腺癌在40-50岁的患病率$P(y=1)=0.004$，忽略此先验会导致基本概率谬误(base rate fallacy)。还需了解该检查的假阳率(false positive)，假设假阳率$P(x=1|y=0)=0.1$。由此，可以计算出该名女性确患病的概率为：
\\[P(y=1|x=1)=\dfrac{P(x=1|y=1)P(y=1)}{P(x=1|y=1)P(y=1)+P(x=1|y=0)P(y=0)}=0.031.\\]
因此，即使检查结果为阳性，该名女性患病的概率为0.031。

### Likelihood（似然）

似然概率与通常的概率相反，通常概率指的是在已知参数$\theta$的情况下，预测发生事件$D$的概率。例如：已知单次抛硬币正面朝上(Head)的概率为0.5，现抛三次硬币，求其中有两次正面朝上的概率\\[P(HH|\theta=0.5)=C_3^2\times(0.5)^3=0.375.\\]
而似然函数则是在已知发生事件$D$的前提下，推测参数$\theta$，即$P(\theta|D)$。也就是已知抛三次硬币中，三次正面都朝上，求单次抛硬币正面朝上的概率$\theta=0.5$的可能性是多少。可以表示为\\[L(\theta=0.5|HHH)=P(HHH|\theta=0.5)=0.125,\\]
即，当三次抛硬币都是正面朝上时，单次抛硬币正面朝上概率为0.5的可能性是0.125。

由此，我们可以得到一种推算参数$\theta$的方法，极大似然估计(Maximum Likelihood Estimate, MLE)。即，推测参数$\theta$，使该参数下事件$D$发生的似然概率最大。例如，上述问题可表示为

$$\hat{\theta}_{MLE}=\mathop{argmax}_{\theta}~P(HHH|\theta=0.5).$$

但如果按照MLE估计上述事件中单次抛硬币正面朝上的概率$\theta$，则$\theta=1$，这显然是不合理的，这就是稀疏数据(sparse data)导致的过拟合现象(overfitting)。

### Prior（先验）

这个很好理解，先验概率就是由以往的经验得到的概率或假设。此外，还有一个共轭先验(conjugate prior)的概念，若后验分布(posterior distribution)与先验分布(prior distribution)属于同类分布，则称这组先验和后验分布为共轭分布(conjugate distributions)，这个先验被称为其似然函数的共轭先验。

### Posterior（后验）

后验概率是参数$\theta$给定证据$X$后的概率$P(\theta \vert X)$。根据贝叶斯定理，假设先验服从概率分布函数$P(\theta)$，则后验概率可以表示为
\\[ P(\theta \vert X)=\frac{P(X \vert \theta)P(\theta)}{P(X)}.\\] 又因为，事件$X$的概率$P(X)$是确定的，后验概率可以表示为
\\[ P(\theta \vert X) \propto P(X \vert \theta)P(\theta).\\]

例如，现有五个箱子${A,B,C,D,E}$,每个箱子里有100个球，最多包含黑白两种颜色，具体信息如下表。

|箱子编号|A|B|C|D|E|
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
|球|全白|75白 25黑|50白 50黑|25白 75黑|全黑|

在不知道箱子编号的情况下选择一个箱子，然后以放回抓取的方式随机抽取两个球，得到两个黑球。求这个箱子最可能是哪一个？
令$\theta$代表箱子编号，事件$bb$代表抽取的两个球都是黑球。忽略选取箱子的概率，或每个箱子被选择的概率相同，采用极大似然估计，问题可表示为

$$\hat{\theta}_{MLE}=\arg\mathop{\max}_{\theta}~P(bb \vert \theta).$$

由箱子内球的信息，满足似然概率最大，可以得到该箱子是E的可能性最高。但是，考虑不同箱子被选择的概率不同，如下表

|箱子编号|A|B|C|D|E|
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
|箱子概率|0.1|0.2|0.4|0.2|0.1|

用MLE推测$\theta$就不合理了。这里要加入箱子被选择概率$P(\theta)$，也就是先验，得到这个问题的后验概率$P(\theta \vert bb)$，采用满足此概率最大的策略推测参数$\theta$，也就是最大后验估计(Maximum A Posteriori, MAP),

$$\hat{\theta}_{MAP}=\arg\mathop{\max}_{\theta}~\frac{P(bb \vert \theta)P(\theta)}{P(bb)}.$$

其中，$P(bb)=\sum_{\theta'}P(\theta' \vert bb)P(\theta'),~\theta'={A, B, C, D, E}$。即上式可以简化为

$$\hat{\theta}_{MAP}=\arg\mathop{\max}_{\theta}~P(bb \vert \theta)P(\theta)$$

通过计算，可以得到$\theta=D$时，后验概率最大。
