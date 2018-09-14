---
layout:    post
title:     Generative models for discrete data
subtitle:  Bayesian concept learning, Beta-binomial and Dirichlet-multinomial model
date:      2018-09-14
author:    DonGovi
header-img:img/ml-bg.p
catalog:   true
tags:      Machine Learning
---




# Generative models for discrete data

本系列主要参考Kevin P. Murphy的书籍，Machine learning: A Probabilistic Perspective。该书从理论角度讲述机器学习算法，内容详尽，逻辑流畅，值得机器学习入门阅读。

## 前言

首先需要讲一下几个概率统计的概念，方便后面理解。

### Bayes Theorem（贝叶斯定理）

贝叶斯定理描述了条件概率$P(A|B)$与条件概率$P(B|A)$之间的关系，即
$$
P(A|B)P(B)=P(A,B)=P(B|A)P(A).
$$

贝叶斯定理可以用在许多问题中都有应用，例如：一位年龄在40多岁的女性进行乳腺X光检查，假设已知该
