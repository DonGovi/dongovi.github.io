---
layout:    post
title:     Generative models for discrete data
subtitle:  Bayesian concept learning, Beta-binomial and Dirichlet-multinomial model
date:      2018-09-14
author:     DonGovi
header-img: img/post-bg-ml.jpg
catalog:   true
mathjax:   true
tags:      
     Machine-Learning
---

<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>



# Generative models for discrete data

本系列主要参考Kevin P. Murphy的书籍，Machine learning: A Probabilistic Perspective。该书从理论角度讲述机器学习算法，内容详尽，逻辑流畅，值得机器学习入门阅读。

## 前言

首先需要讲一下几个概率统计的概念，方便后面理解。

### Bayes Theorem（贝叶斯定理）

贝叶斯定理描述了条件概率$P(A|B)$与条件概率$P(B|A)$之间的关系，即
$$
P(A|B)P(B)=P(A,B)=P(B|A)P(A).
$$

贝叶斯定理可以用在许多问题中都有应用，例如：一位年龄在40多岁的女性进行乳腺X光检查，假设已知该检查的敏感度(sensitivity)为80%，若检查结果为阳性，该女性患病的概率是多少？设事件$x=1$表示乳腺X光检查结果为阳性，事件$y=1$表示患者确患有乳腺癌，则检查的敏感度为80%可表示为$P(x=1|y=1)=0.8$。此外，该乳腺癌在40-50岁的患病率$P(y=1)=0.004$，忽略此先验会导致基本概率谬误(base rate fallacy)。还需了解该检查的假阳率(false positive)，假设假阳率$P(x=1|y=0)=0.1$。由此，可以计算出该名女性确患病的概率为：
$$
P(y=1|x=1)=\dfrac{P(x=1|y=1)P(y=1)}{P(x=1|y=1)P(y=1)+P(x=1|y=0)P(y=0)}=0.031.
$$
因此，即使检查结果为阳性，该名女性患病的概率为0.031。

### Likelihood（似然）

似然概率与通常的概率相反，通常概率指的是在已知参数$\theta$的情况下，预测发生事件$D$的概率。例如：已知单次抛硬币正面朝上(Head)的概率为0.5，现抛三次硬币，求其中有两次正面朝上的概率
$$
P(HH|\theta=0.5)=C_3^2\times(0.5)^3=0.375.
$$
而似然函数则是在已知发生事件$D$的前提下，推测参数$\theta$，即$P(\theta|D)$。也就是已知抛三次硬币中，三次正面都朝上，求单次抛硬币正面朝上的概率$\theta=0.5$的可能性是多少。可以表示为
$$
L(\theta=0.5|HHH)=P(HHH|\theta=0.5)=0.125
$$
即，当三次抛硬币都是正面朝上时，单次抛硬币正面朝上概率为0.5的可能性是0.125。

由此，我们可以得到一种推算参数$\theta$的方法，极大似然估计(Maximum Likelihood Estimate, MLE)。即，推测参数$\theta$，使该参数下事件$D$发生的似然概率最大。例如，上述问题可表示为：
$$
\hat{\theta}_{MLE}=\mathop{\arg\max}_{\theta}~P(HHH|\theta=0.5)
$$

但如果按照MLE估计上述事件中，单次抛硬币正面朝上的概率$\theta$，则$\theta=1$，这显然是不合理的，这就是稀疏数据(sparse data)导致的过拟合现象(overfitting)。

### Prior（先验）

这个很好理解，先验概率就是由以往的经验得到的概率。先验和后验是相对的概念，这两者之间就是事件本身。

### Posterior（后验）

未完待续
