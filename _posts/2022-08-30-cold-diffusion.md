---
layout: post
title: "Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise"
author: "Juewen Peng"
comments: false
tags: [Diffusion Model, Generative Model]
excerpt_separator: <!--more-->
sticky: false
hidden: false
katex: true
---

<!-- "highlight language" refer to https://github.com/rouge-ruby/rouge/wiki/List-of-supported-languages-and-lexers -->

<!-- 标准的 diffusion model 通常包含添加高斯噪声的前向过程以及去噪的反向过程，而在 [Cold Diffusion](https://arxiv.org/pdf/2208.09392.pdf) 这篇文章中，作者认为 diffusion model 有效的原因并不在于使用高斯噪声对图像做 degradation，并展示了采用其他类型的 deterministic degradation 操作（如 blur，masking 等）依然能完成图像生成的任务。 -->

As we know, a standard diffusion model involves a forward process with gradually adding noise to a pure image and a backward process with denoising. This paper makes an argument that the generative behavior of diffusion models is not strongly dependent on the choice of image degradation, and shows that with other deterministic degradation operations (e.g., blur, masking), the model is also able to implement image generation. <!--more-->

![image]({{site.baseurl}}/images/image-2022-08-30-23-44-01.png){:width="90%"}

---

<br>

# 1. Method

## 1.1. Model Components and Training

Given $$x_0$$, consider a degradation operator $$D$$, which meets 

$$
x_t=D(x_0,t)\,,\ D(x_0,0)=x_0\,.
$$

We also consider a restoration operator $$R$$, which approximately inverts $$D$$ and meets

$$
R(x_t,t)\approx x_0\,.
$$

The training objective is

$$
\min_\theta\ \mathbb{E}_{x\sim \mathcal{X}} \lVert R_\theta(D(x,t), t) - x \rVert\,,
$$

where $$\lVert \cdot \rVert$$ represents $$\ell_1$$ in this paper, and $$R_\theta$$ represents the restoration model.

## 1.2. Sampling Algorithms

Propose two sample algorithms:

![image]({{site.baseurl}}/images/image-2022-08-30-23-49-09.png){:width="60%"}

For cold diffusion with smooth/differentiable degradations (other than adding noise), Alg. 2 is superior to Alg. 1. The reason is shown in the next section.

![image]({{site.baseurl}}/images/image-2022-08-30-23-57-00.png){:width="60%"}

## 1.3. Properties and Derivations

For a linear degradation function, we can assume 

$$
D(x,s)\approx x+s\cdot e+\mathrm{HOT}\,,
$$

where $$\mathrm{HOT}$$ denotes higher order terms. Therefore, the update in Alg. 2 can be written as

$$
\begin{aligned}
x_{s-1} 
& = x_s - D(R(x_s,s),s)+D(R(x_s,s),s-1) \\
& = D(x_0,s)-D(R(x_s,s),s)+D(R(x_s,s),s-1) \\
& = x_0+s\cdot e-R(x_s,s)-s\cdot e+R(x_s,s)+(s-1)\cdot e \\
& = x_0+(s-1)\cdot e \\
& = D(x_0,s-1)\,.
\end{aligned}
$$

We can see that $$x_s=D(x_0,s)$$ regardless of the choice of $$R$$. In other words, for any choice of $$R$$, the iteration behaves the same whether $$R$$ is a perfect inverse for the degradation $$D$$ or not. By contrast, Alg. 1 does not enjoy this behavior. The predicted $$\hat{x}_0$$ may be really bad in the first several steps, which will lead to the failure of the whole inverse process.

(Although the assumption and the derivation are not strict, I think the conclusions make sense intuitively.)

<br>

# 2. Experiments of Image Reconstruction

## 2.1. Deblurring

Perform a sequence of Gaussian kernels $$\{G_s\}$$ on the input image $$x_0$$, and $$x_t$$ can be represented as

$$
x_t=G_t*x_{t-1}=G_t*\cdots*G_1*x_0=\bar{G}_t*x_0=D(x_0,t)\,.
$$

![image]({{site.baseurl}}/images/image-2022-08-31-00-26-23.png){:width="90%"}

![image]({{site.baseurl}}/images/image-2022-08-31-00-26-53.png){:width="90%"}

Quantitatively we can see that using the iterative sampling process is worse than the direct reconstruction in terms of the reconstruction metrics such as RMSE and PSNR, but the qualitative improvements and decrease in FID show the benefits of the iterative sampling routine, which brings the learned distribution closer to the
true data manifold. These phenomena can be seen in the following experiments as well. 

## 2.2. Inpainting

Input image $$x_0$$ is iteratively masked via multiplication with a sequence of masks $$\{z_{\beta_i}\}$$ with increasing $$\beta_i$$. $$\beta_i$$ controls the Gaussian variance of each Gaussian mask.

$$
D(x_0,t)=x_0\cdot\prod^t_{i=1}z_{\beta_i}\,.
$$

![image]({{site.baseurl}}/images/image-2022-08-31-00-38-06.png){:width="90%"}

![image]({{site.baseurl}}/images/image-2022-08-31-00-38-29.png){:width="90%"}

## 2.3. Super Resolution

The degradation operator downsamples the image by a factor of two in each iteration.

![image]({{site.baseurl}}/images/image-2022-08-31-00-46-10.png){:width="90%"}

![image]({{site.baseurl}}/images/image-2022-08-31-00-46-22.png){:width="90%"}

## 2.4. Snowification

Apply snowification transform in each iteration.

![image]({{site.baseurl}}/images/image-2022-08-31-00-48-54.png){:width="90%"}

![image]({{site.baseurl}}/images/image-2022-08-31-00-49-15.png){:width="90%"}

<br>

# 3. Experiments of Image Generation

## 3.1. Deblurring

Start from an 3-channel RGB image with random and uniform color for each channel. To increase the diversity of the generated images, break the symmetry between pixels by adding a small amount of Gaussian noise to each sampled $$x_T$$.

![image]({{site.baseurl}}/images/image-2022-08-31-00-57-43.png){:width="90%"}

![image]({{site.baseurl}}/images/image-2022-08-31-00-57-55.png){:width="90%"}

## 3.2. Other Transformations

For inpainting configuration, paint the mask to random color during the training. At inference time, start from random color image.

For super resolution configuration, start from a $$2\times2$$ (project to the required resolution) image with random colors.

For a new defined transformation deemed animorphosis, iteratively transform a human face to an animal face during the training, and invert it during the inference. The fusion strategy is the same as the noising procedure, but replace the noise with an animal face. 

$$
x_t=\sqrt{\alpha_t}x+\sqrt{1-\alpha_t}z\,.
$$

![image]({{site.baseurl}}/images/image-2022-08-31-01-10-17.png){:width="90%"}

<br>

## 4. References

[1] Arpit Bansal et al., "[Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise](https://arxiv.org/pdf/2208.09392.pdf)", arXiv 2022.
