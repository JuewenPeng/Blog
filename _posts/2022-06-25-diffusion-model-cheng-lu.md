---
layout: post
title: "Diffusion Model 最近在图像生成领域大红大紫，如何看待它的风头开始超过 GAN？"
author: "Juewen Peng"
comments: false
tags: [Diffusion Model, Score Function, GAN, VAE, Normalizing Flow, Autoregressive Model]
excerpt_separator: <!--more-->
sticky: false
hidden: false
katex: true
---

<!-- "highlight language" refer to https://github.com/rouge-ruby/rouge/wiki/List-of-supported-languages-and-lexers -->


本文转自清华朱军教授组 Cheng Lu 博士对于知乎问题 [“diffusion model 最近在图像生成领域大红大紫，如何看待它的风头开始超过 GAN？” ](https://www.zhihu.com/question/536012286/answer/2533146567) 的回答，并在其基础上做了适当修改。该回答详尽阐述了生成模型 diffusion model 的原理以及特点，非常具有启发性。<!--more--> 

（注：以下出现的 “我” 均表示 Cheng Lu）

---

<br>

先自我介绍一下：清华朱军教授组博三在读，博士一直做 deep generative model 的理论工作。之前做 normalizing flow 发表了一篇 ICML 和一篇 ICLR spotlight，最近做了大半年的 diffusion model 的理论工作（一篇关于 diffusion ODE 的最大似然训练的算法发表在了 ICML 2022，另一篇关于 diffusion model 的无需额外训练的 10 步左右的加速采样的工作刚挂 arXiv，后文会详细介绍这两篇工作）。我们组应该是国内做 diffusion model 最早的组之一，跟我同届的 Fan Bao 做的关于 diffusion model 的逆向随机过程的解析方差（Analytic-DPM）还获得了 ICLR 2022 outstanding paper award（相当于以前的 best paper），应该是目前唯一一篇大陆单位独立完成的获奖论文。

我个人感觉，研究 diffusion model 的理论需要接触一些 ODE、SDE 以及采样（比如 MCMC、Langevin dynamics）的相关知识，所以对于刚入门的同学而言门槛可能略高。但就我自己的经验，零基础学这些东西最多不超过半个月，基础好一点可能一周就够了。总的来说，目前这个领域还处于蓬勃发展的状态，个人感觉不管是做理论还是做应用都非常有前景，理论方向有许多基本问题还没有被研究明白，应用方向更是有好多坑可以填。

接下来，我将从我个人的理解角度（偏理论）来仔细分析 diffusion model 的优势。这些理解都是基于我对过去工作的思考，可能有一些不准确的地方，但就当抛砖引玉。

<br>

## 1. 为什么要引入 diffusion model？

生成模型大致可分为以下 4 类（未包含 autoregression model）：

![image]({{site.baseurl}}/images/image-2022-06-26-00-11-32.png){:width="90%"}
<div class="figcap">图 1：generative model 概述（<a href="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/">https://lilianweng.github.io/posts/2021-07-11-diffusion-models/</a>）</div>

varitional auto-encoder（VAE）首先定义了一个隐变量 $$z$$ 满足 $$p(z)=\mathcal{N}(0,I)$$，
接着定义一个条件分布 $$p_\theta(x\vert z)$$（一般参数化成高斯分布或伯努利分布），从而定义了 $$z$$ 和 $$x$$ 的联合分布。当训练好模型后，生成数据 $$x$$ 只需要按照 “祖先采样”（ancestral sampling）的方法：首先采样 $$z\sim p(z)$$，再根据得到的 $$z$$ 来采样 $$x\sim p_\theta(x\vert z)$$，即我们可以把 $$p_\theta(x\vert z)$$ 理解成 “生成器”，把标准高斯噪声 $$z$$ 通过某种 “随机映射” 映射到原始图像的数据分布 $$x$$ 上。

训练 VAE 通常基于最大似然，即 $$\max_\theta{\,\log{p_\theta(x)}}$$。根据贝叶斯公式，我们可以得到 $$\log{p_\theta(x)}=\log{p_\theta(x\vert z)}+\log{p(z)}-\log{p_\theta(z\vert x)}$$，然而，$$\log{p_\theta(x)}$$ 的计算是 intractable 的（因为真实后验 $$\log{p_\theta(z\vert x)}$$ 无法计算），因此我们通常需要借助 variational inference 的技巧，即采用 $$q_\phi(z\vert x)$$ 来近似真实后验，此时可以推出模型似然有一个下界：

$$
\log{p_\theta(x)}\geq\mathbb{E}_{q_\phi(z\vert x)}[\log{p_\theta(x\vert z)}+\log{p(z)}-\log{q_\phi(z\vert x)}]
$$

训练 VAE 本质上就是最大化不等式的右边关于数据分布的期望，这需要同时训练 $$p_\theta(x\vert z)$$ 和 $$q_\phi(z\vert x)$$，并且当且仅当 $$q_\phi(z\vert x)$$ 等于真实后验时等号成立。然而，VAE 领域这么多年的核心问题就是这个变分后验（variational posterior）$$q_\phi(z\vert x)$$ 很难选择。如果选得比较简单，那很可能没办法近似真实后验，导致模型效果不好；而如果选得比较复杂，$$\log{q_\phi(z\vert x)}$$ 又会很难计算，导致难以优化。例如，几年前 Kingma（VAE 和 Adam 的作者）的一篇经典工作 [Improving Variational Inference with Inverse Autoregressive Flow](https://arxiv.org/abs/1606.04934) 就是通过引入 normalizing flow来作为变分后验分布，以提升变分后验分布的表达能力。然而不管怎么样，**变分后验分布的表达能力与计算代价的权衡一直是 VAE 领域的核心痛点。**

反观 GAN 和 normalizing flow，都是只需要一个 “生成器”，先采样高斯噪声，然后用 “生成器” 把这个高斯噪声映射到数据分布就完事了。而且大多数情况下我们只关心生成图像（也就是模型的边缘分布 $$p_\theta(x)$$），并不关心这个后验分布到底是啥。但是 GAN 和 normalizing flow 也有其他缺陷，比如 GAN 还需要额外训练判别器，这导致训练很困难；而 normalizing flow 需要模型是可逆函数，不能随便用一个图像分类或分割领域的 SOTA 神经网络，这也导致模型表达能力受限。**那么，是否存在一种生成模型，训练目标函数简单，只需要训练 “生成器”，不需要训练其他网络（判别器/后验分布等），并且这个 “生成器” 没有特殊限制，可以随便选择表达能力极强的神经网络？答案就是 diffusion model。**

Diffusion model 最初提出是 2015 年的 [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585)，这篇文章写作上跟目前 diffusion model 非常不一样，偏理论且效果较差，不建议新手读。真正把 diffusion model 发扬光大的工作是2020 年提出的 [DDPM](https://arxiv.org/abs/1503.03585)，这篇文章对之前的理论推导做了一定的简化，实现了非常好、甚至超越 GAN 的效果，同时，文章也包含许多实现上的细节。抛开这些实现细节，站在 2022 年回看，我们可以揣测出 2015 年原作者提出这种模型的初衷。以下内容是我个人的理解。

回想 VAE，最大的难题是变分后验很讨厌，这是因为我们首先定义了 “生成器”（条件分布 $$p_{\theta(x\vert z)}$$），然后才定义了变分后验来适配这个生成器。那能不能反过来呢？我们能否先定义一个简单的 “变分后验”（加引号是因为这里是不准确的描述），再定义 “生成器” 去适配它呢？如果可以做到，我们就可以避免优化 “变分后验”，而是直接优化生成器！回忆一下，“生成器” 想要做的是把标准高斯分布映射到数据分布，那么反过来，“变分后验” 其实就是想要把数据分布映射到标准高斯。换句话说，我们能否先定义某种简单的过程，把数据分布映射到标准高斯？这样一来，我们的生成器只需要去拟合这个过程对应的逆过程即可！

再直观理解一下：VAE 同时优化条件分布（“生成器”）和变分后验，我们的目标函数只是要让预测的边缘分布 $$p_\theta(x)$$ 尽可能与真实数据分布接近，然而让模型自己寻找同时适配的条件分布和变分后验是很困难的，因为搜索的空间太大了。然而，如果我们先定义了一个简单的过程，把数据分布映射到高斯分布，那我们的生成器就可以抄作业，直接拟合这个过程每一小步的逆过程即可！**这就是 diffusion model 的核心思想：匹配简单前向过程对应的逆过程的每一小步。**

如果你本科学过随机过程，接触过马尔可夫链（markov chain），那么你就知道可以构造适当的马尔科夫链，使得不管从什么分布出发，沿着马尔可夫链一直采样下去最终可以得到某个你想要的平稳分布（stationary distribution），这也是马尔可夫链蒙特卡洛（MCMC）算法的核心。如果不限制采样的步数，把数据分布映射到标准高斯其实非常容易：**我们只需要构造一个平稳分布是标准高斯分布的马尔可夫链即可。**对于离散的时间，2015 年的那篇文章提出可以如下构造每一步的转移概率（transition distribution）：

$$
q(x_t\vert x_{t-1})=\mathcal{N}(\sqrt{1-\beta_t}\,x_{t-1},\beta_t I),\quad 0<\beta_t<1
$$

其中，$$q(x_0)$$ 为数据分布（初始时刻分布）。由于条件高斯分布 $$q(x_t\vert x_{t-1})$$ 的均值和方差都是关于条件变量地线性函数，我们可以轻松地证明，任意时刻关于初始分布的条件分布依然是均值和方差都是线性的高斯分布：

$$
q(x_t\vert x_0)=\mathcal{N}(\sqrt{\bar{\alpha}_t}\,x_0,(1-\bar{\alpha}_t)I),\quad \bar{\alpha}_t=\prod^t_{s=1}(1-\beta_t)
$$

由于 $$1-\beta_t$$ 小于 1，$$\bar{\alpha}_t$$ 是关于 $$t$$ 的递减函数。如果选择合适的 $$\beta_t$$，可以得到 $$\lim_{t\to\infty}\,\bar{\alpha}_t=0$$，这意味着：

$$
\lim_{t\to\infty}\,q(x_t\vert x_0)=\mathcal{N}(0,I)
$$

也就是说这个条件分布最终会收敛到一个和 $$x_0$$ 无关的分布，因此可以证明边缘分布也会收敛到标准高斯分布，即：

$$
\lim_{t\to\infty}\,q(x_t)=\mathcal{N}(0,I)
$$

这样一来，只要我们取一个足够大的终止时刻 $$N$$ ，就可以逐渐把数据分布 $$x_0$$ 映射到一个非常接近高斯分布的 $$x_N$$ 。我们把这样的随机过程称为前向过程（forward process）。并且，由于 $$q(x_t\vert x_{t-1})$$ 本质上就是把 $$x_{t-1}$$ 放缩后加上一点小小的噪声，我们也可以把这个过程理解为逐渐给数据加噪声，这样的过程也称为扩散过程（diffusion process），这也是 diffusion model 名字的由来。

重点来了：**这样的前向过程的每一小步的逆过程都可以近似为高斯分布！**准确地讲，如果我们选取比较合适的 $$\beta_t$$ ，使得 $$x_t$$ 与 $$x_{t-1}$$ 对应的分布变化非常微小时，我们可以用高斯分布来近似它们的逆过程 $$q(x_{t-1}\vert x_t)$$ 。精确的推导需要借助随机微分方程（SDE）的相关知识，可以参考 Yang Song 发表在 ICLR 2021 的 outstanding paper：[Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)。下面从不太严谨的角度来推导：

$$
\begin{aligned}
q(x_t-1\vert x_t) & = q(x_t\vert x_{t-1}) \exp{\big(\log{p_{t-1}(x_{t-1})-\log{p_t(x_t)}}\big)} \\
& \approx q(x_t\vert x_{t-1}) \exp{\big(\log{p_t(x_{t-1})-\log{p_t(x_t)}}\big)} \\
& \approx q(x_t\vert x_{t-1}) \exp{\big((x_{t-1}-x_t)\nabla_x\log{p_t(x_t)}\big)} \\
& = \frac{1}{\sqrt{2\pi}\,\beta_t}\exp{\bigg( \frac{\Vert x_t-\sqrt{1-\beta_t}\,x_{t-1} \Vert^2_2}{2\beta^2_2} + (x_{t-1}-x_t)\nabla_x\log{p_t(x_t)} \bigg)} \\
\end{aligned}
$$

其中，第一行到第二行的近似是 $$q_t(\cdot)\approx q_{t-1}(\cdot)$$（假设前向过程每一步的变化非常微小），第二行到第三行的近似是对 $$\log{p_t}(\cdot)$$ 做一阶泰勒展开。在给定 $$x_t$$ 后，最后一行括号里的公式可以看成是一个关于 $$x_{t-1}$$ 的二次函数，我们可以通过简单的配方来把它配方成 $$\exp{\Big( \frac{\Vert x_{t-1}-\cdot \Vert^2_2}{\cdot} \Big)}$$ 的形式，也就得到了一个高斯分布的形式。这也是为什么 diffusion model 里经常会见到用一个参数化的高斯分布 $$p_\theta(x_{t-1}\vert x_t)$$ 来近似逆向过程。更重要的是，这个高斯分布的均值大部分是解析的，唯一不知道的是 $$\nabla_x\log{p_t(x_t)}$$，这个函数也被称为 “score function”，这也是为什么 diffusion model 还有个名字是 score-based generative model。

![image]({{site.baseurl}}/images/image-2022-06-26-21-59-27.png){:width="90%"}
<div class="figcap">图 2：diffusion model 原理的直观展示</div>

我们可以用上图来直观感受一下。Diffusion model 定义了一个简单的前向过程（下面那行），不断地加噪来把真实数据映射到标准高斯分布；然后又定义一个逆向过程来去噪（上面那行），并且逆向过程的每一步同样只是一个很简单的高斯分布。

总而言之，从理论角度来看，diffusion model 的成功在于我们训练的模型只需要 “模仿” 一个简单的前向过程对应的逆向过程，而不需要像其它模型那样 “黑盒” 地搜索模型。并且，这个逆向过程的每一小步都非常简单，只需要用一个简单的高斯分布来拟合。这为 diffusion model 的优化带来了诸多便利，这也是它 empirical performance 非常好的原因之一。

<br>

## 2. Diffusion model 的定义：去噪模型，噪声估计模型，分数模型

现在大部分关于 diffusion model 的介绍都是基于 [DDPM](https://arxiv.org/abs/1503.03585) 中 “去噪模型” 的描述出发的。然而，DDPM 提出之时，研究者其实并不完全清楚这个模型背后的数学原理，所以文章里的描述有很多不严谨的地方。直到 Yang Song 在 [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) 中才首次揭示了 diffusion model 的连续版本对应的数学背景，并且将统计机器学习中的 denoising score matching 方法与 DDPM 中的去噪训练统一起来。

这部分需要的前置知识略多，需要有 denoising score matching 相关的基础才可以，这里就不展开说了。直观而言，其实就是考虑我们上一节介绍的平稳分布为标准高斯分布的马尔科夫链的连续版本：随机微分方程（SDE）。前向过程就是一个不断加噪声的 SDE，它唯一对应了一个逆向过程（reverse SDE），其中的未知量只有每一时刻的 score function：

![image]({{site.baseurl}}/images/image-2022-06-26-22-15-05.png){:width="90%"}
<div class="figcap">图 3：score-based generative model</div>

用一个 score model（分数模型）去拟合 score function，可以证明，所谓的 “去噪模型” 与 “分数模型” 其实采用的是等价的 loss，这其实只需要用 denoising score matching 的一个性质就可以证明。有些文章里会出现 “去噪”，而有些文章又会出现 “估计噪声”，有些文章又会出现 “score function”，但其实它们都是等价的参数化。这一点在 Kingma 发表在 NeurlPS 2021 的 [Variational Diffusion Models](https://arxiv.org/abs/2107.00630) 中也有提到：

![image]({{site.baseurl}}/images/image-2022-06-26-22-25-25.png){:width="90%"}
<div class="figcap">图 4：diffusion model 的三种等价参数化形式</div>

其中，$$x_\theta$$ 是 “去噪模型”，$$\epsilon_\theta$$ 是 “噪声估计模型”（DDPM 用的就是这种参数化方法），$$s_\theta$$ 是 “分数模型”。除了这篇论文，CVPR 2022 关于 diffusion model 的 [tutorial](https://drive.google.com/file/d/1noQ2d4-nzSh3Yp3-mOOYN8YGJoSa4pmz/view) 中 Part 2 部分也证明了 DDPM 这种 “噪声估计模型” 与 “分数模型” 的损失函数是完全等价的。

<br>

## 3. 连续时间的 diffusion model：diffusion SDE 与 diffusion ODE

深度生成模型除了可以生成数据以外，还有一类核心任务是估计数据的概率密度（可以用模型计算的数据似然（likelihood）来刻画）。GAN 被诟病的一点就是无法计算似然，因为它是隐式生成模型（implicit generative model），这导致 GAN 无法被用来数据压缩等领域。而 VAE 只能计算数据似然的一个下界，也不太令人满意。在 diffusion model 提出之前，人们通常都是用 autoregressive model 或者 normalizing flow来计算数据的精确似然，然而这类模型的表现都不够好。我博一时候与组里另一位博士后学长合作的 [VFlow](https://arxiv.org/abs/2002.09741) 就是为了提高 normalizing flow 的似然，达到了当时 CIFAR-10 最好的似然估计水平。然而，这类模型非常难训练，模型的参数量也巨大，很不美观。

再来看 diffusion model，上文提到的连续版本的 reverse SDE 又可以称为 diffusion SDE 或 score-based SDE，它们都是基于 SDE 定义的生成模型。这类模型与 VAE 一样，无法计算精确的数据似然（likelihood），只能计算 ELBO（evidence lower bound objective）。但是，diffusion model 的一个优势就在于，只要训练好了 score model（上文提到的三种等价形式都可以转换为 score model），那么就可以导出一个 [Neural ODE](https://arxiv.org/abs/1806.07366)，它是一类 continuous normalizing flow，可以精确计算数据的似然。这个推导也是由 Yang Song 在 [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) 中首次提出。这里就不展开详细的推导了，我们可以简单理解为，**diffusion model 可以定义一个 Neural ODE，它可以被用来做密度估计和似然计算。**基于 diffusion ODE，diffusion Model 在密度估计领域里也达到了 SOTA，吊打了以往所有的 normalizing flow。

目前大部分应用领域的文章都是基于 diffusion SDE（大多是其离散版本，如 DDPM）来发的，这是因为这类模型的生成效果往往比 diffusion ODE 好。然而，diffusion SDE 有一个致命问题是采样速度极慢，因为 SDE 的离散化比 ODE 困难得多。目前基于 diffusion SDE 的最快的采样方法应该还是 Fan Bao 的 [Analytic-DPM](https://arxiv.org/abs/2201.06503) 和它的[扩展版本](https://arxiv.org/abs/2206.07309)，可以做到 25-50 步就达到比较好的效果。然而，步数再低，采样效果就急剧下降了。

在这个月之前，diffusion ODE 的生成效果一直都不如 diffusion SDE。然而，这个问题在这个月被大名鼎鼎的 StyleGAN 的作者给解决了。在最近挂 arXiv 的 [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) 中，大佬们直接把 diffusion ODE 的生成效果调到比 diffusion SDE 还要好，这给 diffusion ODE 的后续下游应用也带来了非常大的希望。

<br>

## 4. 离散时间的 diffusion model：DDPM 与 DDIM

关于离散时间的 diffusion model，可以先看看 [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) 这个 tutorial 来学习一下基本的概念。然而，我个人建议，学习离散时间的 diffusion model 一定要站在 “**连续时间模型的离散化**” 角度去看，这样能把知识系统地串起来。

### 4.1. DDPM 与其解析形式 Analytic-DPM：对 diffusion SDE 的离散化

DDPM 提出要用一个特殊参数化的高斯分布来建模逆过程 $$p_\theta(x_{t-1}\vert x_t)$$ ，这个高斯分布的均值特殊地取成了 $$q(x_{t-1}\vert x_t,x_0)$$ 的均值，并把 $$x_0$$ 换成了一个参数化模型（即上文提到的三种等价参数化模型中的 “去噪” 模型）：

![image]({{site.baseurl}}/images/image-2022-06-27-13-19-52.png){:width="90%"}
<div class="figcap">图 5：离散时间 diffusion model 逆过程的均值参数化</div>

具体的推导基于最小化前向过程和逆向过程的联合分布的 KL。然而，DDPM 原文中的推导非常不严谨，有一点 “硬凑” 的感觉，并且也没有给出怎么确定高斯分布的方差，也是强行凑了一个方差。在这之后原作者又做了个后续工作叫 [Improved DDPM](https://arxiv.org/abs/2102.09672)，使用了一个参数化网络去学习高斯分布的协方差矩阵，但优化起来也非常麻烦。

在这之后，Yang Song 在 [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) 的附录中证明了，DDPM 的采样过程（基于 $$p_\theta(x_{t-1}\vert x_t)$$ 的祖先采样（ancestral sampling），实际上等价于 diffusion SDE 的一阶离散化，这就统一了 diffusion SDE 的离散与连续。然而，实际上针对 SDE 的一阶离散化有无数种等价形式（只需要保证离散化误差的阶是一阶即可）。有一个非常重要的问题还没有被解决：**DDPM 这种看似有点巧妙的 “硬凑” 出来的均值到底对应了怎样的 SDE 离散化？有没有更好的方差建模方式？**

接着在 ICLR 2022 有两篇独立研究的工作出现，一篇是我实验室同届的同学 Fan Bao 提出的 [Analytic-DPM](https://arxiv.org/abs/2201.06503)（Oral，outstanding paper award），另一篇是 [Diffusion-Based Voice Conversion with Fast Maximum Likelihood Sampling Scheme](https://arxiv.org/abs/2109.13821)（Oral），这两篇结合起来证明了 **DDPM 的均值实际上是对 diffusion SDE 的 maximum likelihood SDE solver，并且最优方差有解析形式，且可以被 score function 唯一确定。**

具体而言，Analytic-DPM 从严谨的数学角度证明了 DDPM 中最小化 KL 得到的 $$p_\theta(x_{t-1}\vert x_t)$$ 的**均值和方差都是有解析形式的！**这直接颠覆了以前搞 DDPM 那些人的观念。具体而言，最优的均值和方差长这样：

![image]({{site.baseurl}}/images/image-2022-06-27-13-34-56.png){:width="90%"}
<div class="figcap">图 6：Analytic-DPM 中的最优均值与方差的解析形式</div>

Analytic-DPM 在附录中证明了，这里的最优均值等价于 DDPM 中的均值的参数化（把 score function 替换成 score model $$s_\theta$$，再把 score model 等价转换成噪声预测模型 $$\epsilon_\theta$$），并且由于方差里只有 score function 是未知的，也可以直接替换成我们预训练的 score model（或者等价的噪声预测模型）。这是一个非常漂亮的理论：**不但从数学角度严谨支持了 DDPM 原来选择的均值的数学背景，还证明了最优方差也可以用 score model 来近似！**基于这个发现，Analytic-DPM 极大地提高了原始 DDPM 的采样速度，在几十步就可以达到和原来 1000 步的效果。在 Fan Bao 后续发表在 ICML 2022 的工作 [Estimating the Optimal Covariance with Imperfect Mean in Diffusion Probabilistic Models](https://arxiv.org/abs/2206.07309) 中，把 Analytic-DPM 扩展到了对角协方差矩阵的形式，并且考虑了均值网络的训练误差来进一步优化协方差矩阵，在原始的 Analytic-DPM 的基础上把效果又提高了一些。

此外，上文提到的那篇同时独立的工作证明了 DDPM 其实是对 diffusion SDE 的 maximum likelihood SDE solver（其实就是 Analytic-DPM 中说的 “最小化每一小步的KL”），但这篇工作并没有给出最优方差，就不多做介绍了。

综上所述，离散时间的 DDPM 其实基本被研究清楚了：**DDPM 对应了 diffusion SDE 的 maximum likelihood SDE solver，并且最优方差由 Analytic-DPM 来解析地给出。**


### 4.2. DDIM 与其高阶形式 DPM-Solver：对 diffusion ODE 的离散化

Jiaming Song 在 ICLR 2021 提出了 [DDIM](https://arxiv.org/abs/2010.02502)，对应的是另一种离散时间的 diffusion model。在那时，diffusion model 的连续形式还没有被提出，因此 DDIM 的原始推导也像 DDPM 那样 “硬凑” 了一个逆向过程的均值与方差，并且当方差设置为 0 时变成了一个 implicit generative model。巧的是，这种方差为 0 的模型由于没有噪声的干扰，可以在几十步到 100 步内就可以达到原始 DDPM 接近 1000 步的采样质量。这是针对 DDPM 的加速采样的开山之作。

然而，当 diffusion model 的连续版本提出之时，DDIM 到底对应了一种怎样的连续版本，一直是一个未解之谜。直到 ICLR 2022 中，DDPM 的原作者们提出了基于 DDIM 的蒸馏来为 diffusion model 做加速：[Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512)，在这篇文章的附录中证明了 DDIM 是 diffusion ODE 在换元到 log-SNR 后的一阶 ODE solver。然而，一直以来大家都发现使用那些 general 的 ODE solver（例如 RK45）在步数少的时候采样质量远远不如 DDIM，而这篇文章的证明里并没有能给出一个直观的解释：为什么 DDIM 能比普通的ODE solver 好？

这个问题直到最近才被我和实验室同学提出的 [DPM-Solver](https://arxiv.org/abs/2206.00927) 所解决：我们证明了，DDIM 其实是基于 diffusion ODE 的半线性结构（semi-linear ODE）的一阶 ODE solver，并且给出了对应的高阶 solver 形式。具体地，我们给出了 **Diffusion ODE 的解的解析形式**：

![image]({{site.baseurl}}/images/image-2022-06-27-16-13-14.png){:width="90%"}
<div class="figcap">图 7：diffusion ODE 的解析解</div>

我们可以看到，diffusion ODE 的离散化其实就是对这个解析解的积分做离散化近似。一个最简单的方法就是直接让 $$\epsilon_\theta(\tilde{x}_\lambda,\lambda)\approx\epsilon_\theta(\tilde{x}_{\lambda_{t_{i-1}}},\lambda_{t_{i-1}})$$ （零阶泰勒展开），那么可以得到：

![image]({{site.baseurl}}/images/image-2022-06-27-16-13-20.png){:width="90%"}
<div class="figcap">图 8：DDIM 是 DPM-Solver 的一阶形式</div>

如果忽略后面的误差项，这其实就是 DDIM！基于这个观察，我们进一步做了更高阶的 ODE solver（类似数值分析中 Runge-Kutta 法的推导，这里就不展开了），我们把他们统一称为 DPM-Solver。

综上所述，离散时间的 DDIM 其实也基本被研究清楚了：**DDIM 对应了 diffusion ODE 的一阶 ODE solver，它的加速效果好是因为它考虑了 ODE 的半线性结构，而 DPM-Solver 给出了对应的更高阶的 solver，可以让 10 步左右的采样达到与 DDPM 的 1000 步的采样相当。**

<br>

## 5. Diffusion model 的优势与热点研究问题

Diffusion model 最大的优势是训练简单。例如 “预测噪声”，其实就是用一个二范数来训练。diffusion model 借助了图像分割领域的 UNet，训练 loss 稳定，模型效果非常也好。相比于 GAN 需要和判别器对抗训练或者 VAE 需要变分后验，diffusion model 的 loss 真的是太简单了。究其本质，其实就是我在第一节中提到的，diffusion model 只需要 “模仿” 一个非常简单的前向过程对应的逆过程即可。这样简单高效的训练也使得 diffusion model 在许多任务中的表现都非常好，甚至超过了 GAN。

然而，目前 diffusion model 的理论研究中最大的问题就是它的采样速度较慢。目前有许多针对这个问题的研究，开山之作是 Jiaming Song 提出的 DDIM，包括 Fan Bao 的 Analytic-DPM 的主要应用也是加速 DDPM 的采样。我们最新的研究提出了一个叫 DPM-Solver 的加速采样算法，证明了 DDIM 是 diffusion ODE 的一阶 ODE solver，并且提出了二阶、三阶 solver，可以做到 10 步采样质量很不错，20 步几乎收敛。我们的方法不需要任何额外训练，任给一个 pretrained model 都可以直接用。据我所知，这应该是目前 training-free sampler 里最快的了。

<br>

## 6. 对 diffusion model 的一些探索

在这里介绍一下我和朋友们在这个领域的一些探索。目前做了两个工作，一个关于快速采样，一个关于 diffusion ODE 的最大似然训练。

### 6.1. DPM-Solver：10步左右的快速采样算法

![image]({{site.baseurl}}/images/image-2022-06-27-17-12-01.png){:width="90%"}
<div class="figcap">图 9：DPM-Solver 与 DDIM 对比效果图</div>

我们提出的 DPM-Solver 可以在 10 步左右就能达到非常好的采样效果，相比于 DDIM 而言有巨大的速度提升。事实上，我们在这篇工作中提出了 diffusion ODE 的解的解析形式，并且证明了 DDIM 只是 DPM-Solver 的一阶 solver。我们提出的二阶、三阶 solver 不需要任何额外的训练，只需要对 DDIM 的采样算法稍作修改，即可得到巨大的性能提升：

![image]({{site.baseurl}}/images/image-2022-06-27-17-14-23.png){:width="90%"}
<div class="figcap">图 10：DPM-Solver 性能（越靠左下越好）</div>

其中涉及到的数学推导较多，这里就不展开了，欢迎对 diffusion model 加速采样感兴趣的同学关注！我们的代码也会在论文接收后放出（但其实代码真的没几行，核心算法都是10行结束，着急的同学其实可以看 paper 很快就能复现出来）。

### 6.2. 对 diffusion ODE 的最大似然训练方法：高阶 denoising score matching

针对 diffusion ODE 的最大似然训练问题一直是个 open problem，我们发表在 ICML 2022 的 [Maximum Likelihood Training for Score-Based Diffusion ODEs by High-Order Denoising Score Matching](https://arxiv.org/abs/2206.08265) 从理论角度彻底解决了这个 open problem。我们的理论贡献可以浓缩为一个图：

![image]({{site.baseurl}}/images/image-2022-06-27-17-18-28.png){:width="90%"}
<div class="figcap">图 11：diffusion SDE 与 diffusion ODE 的联系与区别</div>

这里面涉及到的公式较多，就不展开了，总的来讲，我们从理论角度彻底分析出了 diffusion SDE 与 diffusion ODE 的最大似然训练的联系和区别，并且提出了新的二阶、三阶 denoising score matching 算法来解决了 diffusion ODE 的最大似然训练问题。这里给两个简单数据的示意图，来说明我们方法的有效性：

![image]({{site.baseurl}}/images/image-2022-06-27-17-21-18.png){:width="90%"}
<div class="figcap">图 12：diffusion ODE 的最大似然训练（一维）</div>

![image]({{site.baseurl}}/images/image-2022-06-27-17-22-09.png){:width="90%"}
<div class="figcap">图 13：diffusion ODE 的最大似然训练（二维）</div>

结论就是，之前训练 diffusion model 的 “去噪” 方法其实是一阶 score matching，它只适合训练 diffusion SDE 的最大似然估计，但并不适合训练 diffusion ODE 的最大似然估计。想要对 diffusion ODE 做最大似然估计，需要采用我们提出的方法。


<br>

## 7. 总结
总的来看，diffusion model 领域正处于一个百花齐放的状态，这个领域有一点像 GAN 刚提出来的时候，但目前的训练技术让 diffusion model 直接跨越了 GAN 领域调模型的阶段，而是直接可以用来做下游任务。这个领域有一些核心的理论问题还需要研究，这给我们这些做理论的人提供了个很有价值的研究内容。并且，哪怕对理论研究不感兴趣，由于这个模型已经很 work 了，它和下游任务的结合也才刚刚起步，有很多地方都可以赶紧占坑。我相信 diffusion model 的加速采样肯定会在不久的将来彻底被解决，从而让 diffusion model 占据深度生成模型的主导。

另外，介绍两个diffusion model的入门博客：

1. 离散 diffusion model（DDPM与DDIM）：[What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
2. 连续 diffusion model（score-based model）：[Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.github.io/blog/2021/score/)

还有一个最新的 diffusion model 论文的 collection list：[What's the score?](https://scorebasedgenerativemodeling.github.io/)
