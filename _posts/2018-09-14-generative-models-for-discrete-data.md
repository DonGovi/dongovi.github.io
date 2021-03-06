---
layout:    post
title:     Generative models for discrete data
subtitle:  Naive Bayes classifiers
date:      2018-09-14
author:     DonGovi
header-img: img/post-bg-ml.jpeg
catalog:   true
mathjax:   true
tags:    Machine-Learning
---


本系列主要参考Kevin P. Murphy的书籍，Machine learning: A Probabilistic Perspective。该书从理论角度讲述机器学习算法，内容详尽，逻辑流畅，值得机器学习入门阅读。

## 引言

首先需要讲一下几个概率统计的概念，方便后面理解。

#### Bayes Theorem（贝叶斯定理）

贝叶斯定理描述了条件概率$P(A \vert B)$与条件概率$P(B \vert A)$之间的关系，即\\[P(A \vert B)P(B)=P(A,B)=P(B \vert A)P(A)\\]

贝叶斯定理可以用在许多问题中都有应用，例如：一位年龄在40多岁的女性进行乳腺X光检查，假设已知该检查的敏感度(sensitivity)为80%，若检查结果为阳性，该女性患病的概率是多少？设事件$x=1$表示乳腺X光检查结果为阳性，事件$y=1$表示患者确患有乳腺癌，则检查的敏感度为80%可表示为$P(x=1|y=1)=0.8$。此外，该乳腺癌在40-50岁的患病率$P(y=1)=0.004$，忽略此先验会导致基本概率谬误(base rate fallacy)。还需了解该检查的假阳率(false positive)，假设假阳率$P(x=1|y=0)=0.1$。由此，可以计算出该名女性确患病的概率为：
\\[P(y=1|x=1)=\dfrac{P(x=1|y=1)P(y=1)}{P(x=1|y=1)P(y=1)+P(x=1|y=0)P(y=0)}=0.031\\]
因此，即使检查结果为阳性，该名女性患病的概率为0.031。

#### Likelihood（似然）

似然概率与通常的概率相反，通常概率指的是在已知参数$\theta$的情况下，预测发生事件$D$的概率。例如：已知单次抛硬币正面朝上(Head)的概率为0.5，现抛三次硬币，求其中有两次正面朝上的概率\\[P(HH|\theta=0.5)=C_3^2\times(0.5)^3=0.375\\]
而似然函数则是在已知发生事件$D$的前提下，推测参数$\theta$，即$P(\theta|D)$。也就是已知抛三次硬币中，三次正面都朝上，求单次抛硬币正面朝上的概率$\theta=0.5$的可能性是多少。可以表示为\\[L(\theta=0.5|HHH)=P(HHH|\theta=0.5)=0.125\\]
即，当三次抛硬币都是正面朝上时，单次抛硬币正面朝上概率为0.5的可能性是0.125。

由此，我们可以得到一种推算参数$\theta$的方法，极大似然估计(Maximum Likelihood Estimate, MLE)。即，推测参数$\theta$，使该参数下事件$D$发生的似然概率最大。例如，上述问题可表示为：

$$\hat{\theta}_{MLE}=\arg\mathop{\max}_{\theta}~P(HHH|\theta=0.5).$$

但如果按照MLE估计上述事件中单次抛硬币正面朝上的概率$\theta$，则$\theta=1$，这显然是不合理的，这就是稀疏数据(sparse data)导致的过拟合现象(overfitting)。

#### Prior（先验）

这个很好理解，先验概率就是由以往的经验得到的概率或假设。

#### Posterior（后验）

后验概率是参数$\theta$给定证据$X$后的概率$P(\theta \vert X)$。根据贝叶斯定理，假设先验服从概率分布函数$P(\theta)$，则后验概率可以表示为
\\[ P(\theta \vert X)=\frac{P(X \vert \theta)P(\theta)}{P(X)}\\] 又因为，事件$X$的概率$P(X)$是确定的，后验概率可以表示为
\\[ P(\theta \vert X) \propto P(X \vert \theta)P(\theta)\\]

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

#### $\Gamma$函数
$\Gamma$函数是应用中常见的函数，属于欧拉积分，$\Gamma$函数如下：\\[\Gamma(s)=\int_0^{+\infty}x^{s-1}e^{-x}dx,~s>0\\]
对上述积分采用分部积分法，有：

$$\begin{align}
\int_0^{A}x^{s}e^{-x}dx &= -x^se^{-x}\Big \vert_0^A+s \int_0^{A}x^{s-1}e^{-x}dx \\
                        &= -A^se^{-A} + s \int_0^{A}x^{s-1}e^{-x}dx
\end{align}$$

令$A \to +\infty$，可以得到$\Gamma(s+1)=s\Gamma(s)$。因此，$\Gamma$函数可以看做阶乘在实数域的推广。

#### $\text{B}$函数
$B$函数同样是常用的欧拉积分之一，函数式如下：
\\[\text{B}(p,q)=\int_0^1 x^{p-1} (1-x)^{q-1}dx,~p>0,~q>0\\]
此外，$\text{B}$函数和$\Gamma$函数的关系为：

$$
\text{B}(p,q) = \frac{\Gamma(p)\Gamma(q)}{\Gamma(p+q)}
$$

## The beta-binomial model

考虑一个简单的离散问题，抛硬币$N$次，其中正面朝上的次数为$N_1$，背面朝上为$N_0$，即事件$D=\lbrace N_1, N \rbrace $。设$X_i \sim \text{Ber}(\theta)$（Bernoulli分布），其中$X_i=1$表示硬币正面朝上，$X_i=0$表示背面朝上，$\theta \in [0, 1]$是正面朝上的概率，$N_1=\sum_{i=1}^{N} \mathbb{I}(x_i=1)$，$N_0=\sum_{i=1}^{N} \mathbb{I}(x_i=0)$，$N=N_1+N_0$。假设每次抛硬币之间相互独立，若考虑投掷硬币的计数，也就是$N$次中哪几次正面朝上，即$N_1$服从Binomial分布$N_1 \sim \text{Bin}(N, \theta)$，其概率质量函数为\\[\text{Bin}(N_1 \vert N, \theta) \triangleq  \binom{N}{N_1} \theta^{N_1}\theta^{N_0}\\]，由于$ \left( \frac{N}{N_1} \right)$与参数$\theta$无关，其似然可以表达为$p(D \vert \theta) = \theta^{N_1}(1-\theta)^{N_0}$。

#### 先验
假设存在一个先验$a,~b$，即在事件D发生前，已经有$a$次正面朝上，$b$次正面朝上。为了便于计算，假设先验与似然同为Bernoulli分布，即$p(\theta) \propto \theta^a(1-\theta)^b$。由此，可得后验等于\\[p(\theta \vert D) \propto p(D \vert \theta)p(\theta) = \theta^{a+N_1}(1-\theta)^{b+N_0}\\]可见，后验与先验是共轭的，对于Bernoulli分布，其共轭分布为Beta分布，即：\\[\text{Beta}(\theta \vert a,b) \propto \theta^{a-1}(1-\theta)^{b-1}\\]

#### 后验
\\[p(\theta \vert D) \propto \text{Bin}(N_1 \vert \theta, N) \text{Beta}(\theta \vert a,b) = \text{Beta}(\theta \vert N_1+a, N_0+b)\\]
令$\alpha_0=a+b$，代表先验中的样本数。采用最大后验估计参数$\theta$，即为该Beta分布的众数(mode) \\[\hat{\theta}_{MAP}=\dfrac{a+N_1-1}{\alpha_0+N-2}\\] 若采用极大似然估计，或取先验为均匀分布，也就是 $a=1,~b=1$，可得 

\\[\hat{\theta}_{MLE}=\dfrac{N_1}{N}\\]
此外，后验均值可以由Beta分布的公式得到：\\[\bar{\theta}=\dfrac{a+N_1}{\alpha_0 + N}\\]令$m_1=\dfrac{a}{\alpha_0},~\lambda=\dfrac{\alpha_0}{N+\alpha_0}$，后验均值可以表达为：

\\[\mathbb{E}[\theta \vert D]=\lambda m_1+(1-\lambda)\hat{\theta}_{MLE}\\]
其中，$\lambda$越小，也就是先验的数据相对后验越少，后验均值(posterior mean)越趋近于MLE的结果（也就是，随着数据的增多，分布的均值和众数逐渐向MLE收敛）

最后，计算方差度量上述参数估计的不确定性。后验Beta分布的方差为\\[\text{var}[\theta \vert D]=\dfrac{(a+N_1)(b+N_0)}{(\alpha_0+N)^2(\alpha_0 + N + 1)}\\]假设$N \gg \alpha_0$，有\\[\text{var}[\theta \vert D] \approx \dfrac{N_1N_0}{NNN}=\dfrac{\hat{\theta}(1-\hat{\theta})}{N}\\]其中$\hat{\theta}$是MLE，则其标准差为$\sigma = \sqrt{\dfrac{\hat{\theta}(1-\hat{\theta})}{N}}$。也就是说，当$\hat{\theta}=0.5$时，偏差最大，不确定性最高；当$\hat{\theta}$趋近于0或1时，不确定性减小。

#### 后验预测分布
**问题一：**假设一枚硬币已经被投掷了$\alpha_0$次，其中$a$次正面，$b$次反面，求下一次投掷是正面的概率。也就是在后验满足$\text{Beta}(a,b)$的情况下，预测下一次是正面的概率。

$$\begin{align}
p(\tilde{x}=1 \vert D)&=\int_0^1p(x=1 \vert \theta)p(\theta \vert a,b)d\theta \\
                      &=\dfrac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} \int_0^1 \theta^a (1-\theta)^{b-1}d\theta\\
                      &=\dfrac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} \cdot \dfrac{\Gamma(a+1)\Gamma(b)}{\Gamma(a+b+1)}\\
                      &=\dfrac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} \cdot \dfrac{a\Gamma(a)\Gamma(b)}{(a+b)\Gamma(a+b)}=\dfrac{a}{a+b} =\mathbb{E}[\theta \vert a,b]
\end{align}$$

结果等于后验均值。假设$a=0,~b=\alpha_0$，即之前没有抛出过正面，那么下一次是正面的概率就成了0，也就是在数据规模较小的情况下，Bayesian methods的可用性大大降低。对于这种情况，可以使用add-one smoothing，也就是使用均匀先验，设定$a=1,~b=1$，然后再计算后验均值。

**问题二：**现预测之后$M$次投掷中，正面的次数$x$，有：

$$\begin{align}
p(x \vert D,M)&=\int_0^1 \text{Bin}(x \vert \theta,M)Beta(\theta \vert a,b)d\theta \\
              &=\binom{M}{x}\dfrac{1}{B(a,b)}\int_0^1\theta^x (1-\theta)^{M-x} \theta^{a-1} (1-\theta)^{b-1} d\theta \\
              &=\binom{M}{x}\dfrac{B(x+a, M-x+b)}{B(a,b)}
\end{align}$$

即为**beta-binomial分布**，其期望为

$$\begin{align}
\mathbb{E}(x) &= \sum_{x=0}^Mx \int_0^1 \binom{M}{x} \theta^x (1-\theta)^{M-x} \dfrac{\theta^{a-1}(1-\theta)^{b-1}}{\text{B}(a,b)}                d\theta \\
              &= \int_0^1 \sum_{x=0}^Mx \binom{M}{x} \theta^x (1-\theta)^{M-x} \dfrac{\theta^{a-1}(1-\theta)^{b-1}}{\text{B}(a,b)} d\theta\\
              &= \int_0^1 M\theta \dfrac{\theta^{a-1}(1-\theta)^{b-1}}{\text{B}(a,b)} d\theta \\
              &= \dfrac{M}{\text{B}(a,b)} \int_0^1 \theta^a (1-\theta)^{b-1} d\theta \\
              &= M \dfrac{\text{B}(a+1,b)}{\text{B}(a,b)} \\
              &= M \dfrac{a}{a+b}
\end{align}$$

其方差为 $\text{var}(x)=\dfrac{Mab}{(a+b)^2} \dfrac{(a+b+M)}{a+b+1}$

## The Dirichlet-multinomial model

现有一个$K$面的骰子，将其投掷$N$次，$D=\lbrace x_1,...,x_N\rbrace$，其中$x_i=\lbrace 1,...,K \rbrace$，每次投骰子是相互独立的。

#### 似然

$$
p(D \vert \theta)=\prod_{k=1}^K \theta_k^{N_k}
$$

其中$N_k=\sum_{i=1}^N \mathbb{I}(x_i=k)$，即第$k$面朝上的次数。

#### 先验

假设事件$D$发生前，该骰子已经被投掷了$\alpha$次。与上一章问题相近，该先验也需是共轭先验，我们使用狄利克雷分布(Dirichlet distribution)，既满足共轭性质，又满足probability simplex。

$$
Dir(\theta \vert \alpha)=\dfrac{1}{B(\alpha)}\prod_{k=1}^K \theta_k^{\alpha_k-1}
$$

#### 后验

将先验与似然相乘，得到后验，也满足狄利克雷分布。

$$\begin{align}
p(\theta \vert D)  &\propto p(D \vert \theta)p(\theta) \\
              &\propto \prod_{k=1}^K \theta_k^{N_k}\theta_k^{\alpha_k-1}=\prod_{k=1}^K \theta_k^{\alpha_k+N_k-1} \\
              &= \text{Dir}(\theta \vert \alpha_1+N_1,...,\alpha_K+N_K)
\end{align}$$

为了使用MAP解决这个问题，我们首先要约束$\sum_k \theta_k=1$，然后引入拉格朗日乘子(Lagrange multiplier)，将上式转化为似然和先验的对数之和，再加上约束，即：

$$
l(\theta, \lambda)=\sum_k N_k\text{log}\theta_k+\sum_k(\alpha_k-1)\text{log}\theta_k+\lambda\left(1-\sum_k\theta_k\right)
$$

为了使问题简化，令$N_k^{\prime} \triangleq N_k+\alpha_k-1$，$\alpha_0 \triangleq \sum_{k=1}^K\alpha_k$，对$\lambda$求导可得

$$
\dfrac{\partial l}{\partial \lambda}=\left(1-\sum_k \theta_k\right)=0
$$
对$\theta_k$求导可得

$$\begin{align}
\dfrac{\partial l}{\partial \theta_k}&=\dfrac{N_k^{\prime}}{\theta}-\lambda =0\\
 N_k^{\prime}&=\lambda \theta_k \\
\sum_k N_k^{\prime} &= \lambda \sum_k \theta^k  
\end{align}$$

由前面的sum-to-one约束可得$N+\alpha_0-K=\lambda$。因此，MAP估计的结果为

$$
\hat{\theta}_k=\dfrac{N_k+\alpha_k-1}{N+\alpha_0-K}
$$

为狄利克雷分布的众数。假设我们去先验分布为均匀分布，即$\alpha_k=1$，就得到了MLE的结果

$$
\hat{\theta}_k=\dfrac{N_k}{N}
$$

#### 后验预测

现求下一次投掷骰子时，第$j$面朝上的概率。

$$\begin{align}
p(X=j \vert D)&=\int p(X=j \vert \theta)p(\theta \vert D)d\theta \\
&= \int p(X=j \vert \theta_j)\left[ \int p(\theta_{-j}, \theta_j) \vert D)d\theta_{-j} \right]d\theta_j \\
&= \int \theta_j p(\theta_j \vert D) d\theta_j = \mathbb{E}[\theta_j \vert D] =\dfrac{\alpha_j+N_j}{\alpha_0+N}
\end{align}$$

其中，$\theta_{-j}$是除$\theta_j$以外$\theta$中的其他分量。

## 朴素贝叶斯分类器 (naive Bayes classifiers)

现有一个离散数据集$\text{x} \in \lbrace 1,~...,~K\rbrace^D$，其中$D$是特征数，$K$是每项特征可能的取值数。想要得到一个分类模型，首先需要明确这些类别的条件分布$p(\text{x} \vert y=c)$。为了简化问题，假设所有特征相对于类别$y$条件独立，由此该类条件概率密度可以由每一维特征的条件概率相乘得到，即：

$$
p(\text{x} \vert y=c, \theta)=\prod_{j=1}^D p(x_j \vert y=c, \theta_{jc})
$$

如此就得到了该数据集的朴素贝叶斯分类器(NBC)，之所以称之为“朴素(naive)”，原因在于几乎所有问题都无法满足特征关于类别条件独立的假设。

对于不同的特征，类条件概率密度也不相同：
*  若特征是实数特征(real-valued features)，可以使用高斯分布(Gaussion distribution)，即$p(\text{x} \vert y=c, \theta)=\prod_{j=1}^D \mathcal{N}(x_j \vert \mu_{jc}, \sigma_{jc}^2)$，其中$\mu_{jc}$是类别$c$中特征$x_j$在数据集中分布的期望，$\sigma_{jc}^2$是方差。
*  若特征是二进制特征(binary features)，可以使用伯努利分布，即$p(\text{x} \vert y=c, \theta)=\prod_{j=1}^D \text{Ber}(x_j \vert \mu_{jc})$，其中$\mu_{jc}$是类别$c$中特征$x_j$出现的概率。
*  若特征是类别特征(categorical features)，$x_j \in \lbrace1,~...,~K\rbrace$，可以使用multinoulli distribution，即$p(\text{x} \vert y=c, \theta)=\prod_{j=1}^D \text{Cat}(x_j \vert \mu_{jc})$，其中$\mu_{jc}$是类别$c$中特征$x_j$的$K$个可能取值的直方图统计。

#### 模型训练
通常使用MLE或MAP进行参数估计
###### 使用MLE训练NBC

对于数据集$\text{x}$中的每一条数据，其似然概率可以表达为

$$
p(\text{x}_i,y_i \vert \theta)=p(y_i \vert \pi)\prod_j p(x_{ij} \vert \theta_j)=\prod_c \pi_c^{\mathbb{I}(y_i=c)} \prod_j \prod_c p(x_{ij} \vert \theta_{jc})^{\mathbb{I}(y_i=c)}
$$

对其计算对数，可得

$$
\text{log}p(D \vert \theta)=\sum_{c=1}^C N_c \text{log}\pi_c + \sum_{j=1}^D \sum_{c=1}^C \sum_{i:y_i=c} \text{log} p(x_{ij} \vert \theta_{jc})
$$

其中，$N_c \triangleq \sum_i \mathbb{I}(y_i=c)$，代表类别$c$的样本数量。则$y_i$服从多项式分布(multinomial distribution)，$y \sim \text{Mu}(N,~\pi) $，由上一章计算的MLE结果，可以得到$$\hat{\pi}_c=\dfrac{N_c}{N}$$。简单起见，假设所有特征$x_j$都是binary features，我们可以得到$x_j \vert y=c \sim \text{Ber}(\theta_{jc})$，由MLE估计参数$\theta_{jc}$可得$\hat{\theta_{jc}}=\dfrac{N_{jc}}{N_c}$。

###### Bayesian naive bayes

前面提到过，MLE在数据集不充分的情况下可能导致过拟合。接着上面binary features的例子，假设某特征$x_j$在数据集$D$的所有类别的所有样本中均为1，那么

$$
\hat{\theta}_{jc}=1
$$

用这个模型预测时，出现了一个例子，其$x_j=0$，然后模型就发生了错误，因为对于任何类别$c$，有

$$
p(y=c \vert \text{x},\hat{\theta})=0
$$

可以采用**拉普拉斯平滑(Laplace smoothing)**解决这个问题，对数据给定一个先验，对$\pi$给定先验$\text{Dir}(\alpha),~\alpha=\lbrace 1,~...,~1\rbrace^C$，对$\theta_{jc}$给定先验$\text{Beta}(\beta_0,\beta_1),~\beta_0=\beta_1=1$。将先验和上述似然结合，可以得到后验

$$
p(\theta \vert D)=p(\pi \vert D)\prod_{j=1}^D \prod_{c=1}^C p(\theta_{jc} \vert D)
$$

根据前面两章的后验部分，我们可以得到

$$\begin{align}
p(\pi \vert D)&=\text{Dir}(N_1+\alpha_1,~..., N_C+\alpha_C) \\
p(\theta_{jc} \vert D)&=\text{Beta}((N_c-N_{jc})+\beta_0,N_{jc}+\beta_1)
\end{align}$$

#### 使用NBC预测

在测试阶段，目标是计算当前特征$\text{x}$下，每个类别$c$的概率，概率最大的类别即为预测的类别。由贝叶斯定理我们可以得到一个**生成式分类器**（generative classifiers）

$$
p(y=c \vert \text{x},\theta)=\dfrac{p(y=c \vert \theta)p(\text{x} \vert y=c,\theta)}{\sum_{c^{\prime}} p(y=c^{\prime} \vert \theta)p(\text{x} \vert y=c^{\prime}, \theta)}
$$

由于分母已知，可以得到

$$\begin{align}
p(y=c \vert \text{x},D) &\propto p(y=c \vert D)\prod_{j=1}^D p(x_j \vert y=c,D)\\
&\propto \bar{\pi}_c  \prod_{j=1}^D \bar{\theta}_{jc}^{\mathbb{I}(x_j=1)}(1-\bar{\theta}_{jc})^{\mathbb{I}(x_j=0)}
\end{align}$$

根据前面两章中后验预测的计算结果，我们知道狄利克雷分布和beta分布的后验预测即为其分布的**期望**，即

$$\begin{align}
\bar{\pi}_c&=\dfrac{N_c+\alpha_c}{N+\alpha_0}\\
\bar{\theta}_{jk}&=\dfrac{N_{jc}+\beta_1}{N_c+\beta_0+\beta_1}
\end{align}$$

其中$\alpha_0=\sum_{c}\alpha_c$。可以使用对数的方式求解上式

$$\begin{align}
\text{log} p(y=c \vert \text{x},D) &\propto \text{log} \bar{\pi}_c + \text{log} \prod_{j=1}^D \bar{\theta}_{jc}^{\mathbb{I}(x_j=1)}(1-\bar{\theta}_{jc})^{\mathbb{I}(x_j=0)} \\
 &\propto \text{log} \bar{\pi}_c + \sum_{j:x_j=1} \text{log} \bar{\theta}_{jc}+\sum_{j:x_j=0}  \text{log} (1-\bar{\theta}_{jc})
\end{align}$$

算法如下

![1537720805416](/img/ml/nbc_predict.png)

###### The log-sum-exp trick

计算机在计算$\text{log} \sum_c e^{b_c}$时，容易发成下溢(numerical underflow)或上溢(overflow)。解决这个问题最常用的方式是log-sum-exp (LSE)

$$
\text{log} \sum_c e^{b_c}=\text{log}\left[ \left( \sum_c e^{b_c-B}\right)e^B\right]=\text{log} \left( \sum_c e^{b_c-B}\right) + B
$$

其中$B=\max_c b_c$。在上述NBC预测算法中，若只需找到$\hat{y}_i$，只需找到最大的$\text{log}p(y=c \vert \text{x},D) $，无需引入LSE。但若需要计算出$p(y=c \vert \text{x},D) $，则需要LSE计算NBC的分母项，以免因为参数的归一化导致下溢。

## 参考文献
1. Kevin P. Murphy. Machine Learning: A Probabilistic Perspective [M]. The U.S.: Massachusetts Institute of Technology, 2012.
2. 华东师范大学数学系. 《数学分析》[M]. 北京: 高等教育出版社, 2012.



