---
layout: post
title: "DreamFusion: Text-to-3D Using 2D Diffusion"
author: "Juewen Peng"
comments: false
tags: [Text-to-3D, Diffusion Model, Generative Model]
excerpt_separator: <!--more-->
sticky: false
hidden: false
katex: true
---

<!-- "highlight language" refer to https://github.com/rouge-ruby/rouge/wiki/List-of-supported-languages-and-lexers -->

This paper proposes to use a pretrained 2D text-to-image diffusion model to optimize Neural Radiance Fields (NeRF), achieving remarkable text-to-3D synthesis results. <!--more-->

![image]({{site.baseurl}}/images/image-2023-08-30-11-38-34.png){:width="100%"}

---

<br>

## 1. Diffusion Models and Score Distillation Sampling

Training the generative model with a (weighted) evidence lower bound (ELBO) simplifies to a weighted denoising score matching objective for parameters $$\phi$$:

$$
\mathcal{L}_{\rm Diff}(\phi,x) = \mathbb{E}_{t\sim\mathcal{U},\epsilon\sim\mathcal{N}(\bm{0,\mathbf{I}})}\big[w(t)\lVert\epsilon_\phi(\alpha_t\mathbf{x}+\sigma_t\epsilon;t)-\phi\rVert^2_2\big]\,.
$$

This work builds on text-to-image diffusion models that learn $$\epsilon_\phi(\mathbf{z}_t;t,y)$$ conditioned on text embeddings $$y$$. These models usually use classifier-free guidance, which jointly learns an unconditional model to enable higher quality generation via a guidance scale parameter $$w$$: $$\hat{\epsilon}_\phi(\mathbf{z}_t;y,t)=(1+w)\epsilon_\phi(\mathbf{z}_t;y,t)-w\epsilon_\phi(\mathbf{z}_t;t)$$. In practice, setting $$w>0$$ improves sample fidelity at the cost of diversity.

Diffusion models trained on pixels have traditionally been used to sample only pixels. However, what this work wants to do is to **create 3D models that look like good images when rendered from random angles**. Such 3D models can be specified as a differentiable image parameterization (DIP), where a differentiable generator $$g$$ transforms parameter $$\theta$$ to create an image $$\mathbf{x}=g(\theta)$$. 

The authors optimize over parameters $$\theta$$ such that $$\mathbf{x}=g(\theta)$$ looks like a sample from the frozen diffusion model. To perform this optimization, a differentiable loss function where plausible images have low loss and implausible images have high loss is required. Therefore, the authors try to minimize the diffusion training loss with respect to a generated datapoint $$\mathbf{x}=g(\theta)$$:

$$
\theta^*=\argmin_\theta\mathcal{L}_{\rm Diff}(\phi,\mathbf{x}=g(\theta))\,.
$$

The gradient of $$\mathcal{L}_{\rm Diff}$$ is

$$
\nabla_\theta\mathcal{L}_{\rm Diff}(\phi,\mathbf{x}=g(\theta))=\mathbb{E}_{t,\epsilon}\bigg[w(t)\ \underbrace{\vphantom{\frac{\partial{\hat\epsilon_\phi}(\mathbf{z}_t;y,t)}{\partial{\mathbf{z}_t}}}(\hat\epsilon_\phi(\mathbf{z}_t;y,t)-\epsilon)}_{\text{Noise Residual}}\ \underbrace{\frac{\partial{\hat\epsilon_\phi}(\mathbf{z}_t;y,t)}{\partial{\mathbf{z}_t}}}_{\text{U-Net Jacobian}}\ \underbrace{\vphantom{\frac{\partial{\hat\epsilon_\phi}(\mathbf{z}_t;y,t)}{\partial{\mathbf{z}_t}}}\frac{\partial\mathbf{x}}{\partial\theta}}_{\text{Generator Jacobian}}\bigg]\,,
$$

where the constant $$\alpha_t\mathbf{I}=\partial{\mathbf{z}_t}/\partial\mathbf{x}$$ is absorbed into $$w(t)$$. In practice, the U-Net Jacobian term is expensive to compute and poorly conditioned for small noise levels as it is trained to approximate the scaled Hessian of the marginal density. The authors found that omitting the U-Net Jacobian term leads to an effective gradient for optimizing DIPs with diffusion models:

$$
\nabla_\theta\mathcal{L}_{\rm SDS}(\phi,\mathbf{x}=g(\theta))\triangleq\mathbb{E}_{t,\epsilon}\bigg[w(t)\ (\hat\epsilon_\phi(\mathbf{z}_t;y,t)-\epsilon)\ \frac{\partial\mathbf{x}}{\partial\theta}\bigg]\,,
$$

Since the diffusion model directly predicts the update direction, **it is unnecessary to backpropagate through the diffusion model**; the model simply acts like an efficient, frozen critic that predicts image-space edits.

## 2. The DreamFusion Algorithm
For the diffusion model, the authors use the Imagen model, which has been trained to synthesize images from text. We only use the $$64 \times 64$$ base model (not the super-resolution cascade for generating higher-resolution images), and use
this pretrained model as-is with no modifications. To synthesize a scene from text, the authors initialize a NeRF-like model with random weights, then repeatedly render views of that NeRF from random camera positions and angles, using these renderings as the input to the proposed score distillation loss function that wraps around Imagen. Refer to the original paper for more details.

![image]({{site.baseurl}}/images/image-2023-09-03-23-45-31.png){:width="100%"}

<br>
