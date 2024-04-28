# Diffusion Models

In my previous blog, I covered the concept of [Generative Adversarial Networks (GAN)](https://www.notion.so/Generative-Adversarial-Networks-ebfba7327d264466b8496a52cf98e453?pvs=21). Since its inception in 2014, it became the state-of-the-art on image generation tasks and we saw an era of GAN improvement in subsequent years with introduction of models such as [StackGAN](https://arxiv.org/abs/1612.03242), [AttnGAN](https://arxiv.org/abs/1711.10485) and [GAN-Control](https://arxiv.org/abs/2101.02477).

Although the concept of Diffusion Models was introduced by [Sohl-Dickstein et al.](https://arxiv.org/abs/1503.03585) early in 2015, it was not until a [paper by OpenAI in 2021](https://arxiv.org/pdf/2105.05233) showed that they could achieve superior performance on image synthesis tasks. Since then, notable diffusion-based generative models have been released such as [DALL-E](https://arxiv.org/abs/2102.12092), [Stable Diffusion](https://arxiv.org/abs/2112.10752) and [Imagen](https://arxiv.org/abs/2205.11487). Today, I will be covering the concept underlying diffusion models.

## What are Diffusion Models?

Diffusion models are inspired by non-equilibrium thermodynamics and the idea behind it as described by Sohl-Dickstein et al. (2015) is this:

> The essential idea, inspired by non-equilibrium statistical physics, is to systematically and slowly destroy structure in a data distribution through an iterative forward diffusion process. We then learn a reverse diffusion process that restores structure in data, yielding a highly flexible and tractable generative model of the data.
> 

To better understand this, let’s focus on the forward diffusion process and reverse diffusion process separately.

### Forward Diffusion Process

![Image by [Karagiannakos and Adaloglou (2022)](https://theaisummer.com/diffusion-models/) modified from [Ho et al. (2020)](https://arxiv.org/abs/2006.11239)](Images/Untitled.png)

Image by [Karagiannakos and Adaloglou (2022)](https://theaisummer.com/diffusion-models/) ****modified from [Ho et al. (2020)](https://arxiv.org/abs/2006.11239)

In the forward trajectory, we want to gradually “corrupt” the training images. As such, we iteratively apply Gaussian noise to images sampled from the true data distribution, i.e. $x_0 \sim q(x)$, in over $T$ steps to produce a sequence of noisy samples $x_0, x_1, \dots , x_T$. 

The diffusion process is fixed to a Markov chain which simply means that each step is only dependent on the previous one (memoryless). Specifically, at each step, we apply Gaussian noise with variance $\beta_T \in (0,1)$ to $x_{t-1}$ to produce a latent variable $x_t$ of the same dimension. As such, each transition is parameterized as a diagonal Gaussian distribution that uses the output of the previous state as its mean:

$$
q(x_t|x_{t-1}) = \mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1}, \beta_t\mathbf{I})
$$

Note:

- $\beta_T$ is known as the “diffusion rate” and it can be sampled according to a variance schedule $\beta_1, \dots, \beta_T$, which means the amount of noise applied at each time step is not necessarily constant
- $\mathbf{I}$ is the identity matrix. We use the identity matrix because our images are multi-dimensional and we want each dimension to be independent of each other

The posterior after $T$ steps, conditioned on the original data distribution, can be represented as a product of single step conditionals as such:

$$
q(x_{1:T} | x_0) = \prod^T_{t=1} q(x_t | x_{t-1})
$$

As $T\rightarrow \infty$, $x_T$ is equivalent to an isotropic Gaussian distribution, losing all information about the original sample distribution.

## Reverse Diffusion Process

![Image by [Karagiannakos and Adaloglou (2022)](https://theaisummer.com/diffusion-models/) modified from [Ho et al. (2020)](https://arxiv.org/abs/2006.11239)](Images/Untitled%201.png)

Image by [Karagiannakos and Adaloglou (2022)](https://theaisummer.com/diffusion-models/) ****modified from [Ho et al. (2020)](https://arxiv.org/abs/2006.11239)

In the reverse process, we aim to learn a model that can denoise the pure Gaussian noise, i.e. $x_T \sim \mathcal{N}(x_T; 0,\mathbf{I})$, to recover the original sample image. As mentioned by Sohl-Dickstein et al. (2015),

> Estimating small perturbations is more tractable than explicitly describing the full distribution with a single, non-analytically-normalizable, potential function.
> 

Directly describing the original distribution from the pure Gaussian noise can be intractable. Rather, what we can do is train a model $p_\theta$ (e.g. using a neural network) to approximate $q(x_{t-1}|x_{t})$ such that we can iteratively recover the original data distribution in small time steps. Therefore, the reverse trajectory can also be formulated as a Markov chain and can be represented as such:

$$
p_\theta(x_{0:T}) = p(x_T) \prod^T_{t=1} p_\theta(x_{t-1} | x_t)
$$

where $p(x_T) = \mathcal{N}(x_T; 0,\mathbf{I})$.

Moreover, since $q(x_t|x_{t-1})$ follows a Gaussian distribution, if $\beta_t$ is small, then the reversal of the diffusion process has the identical functional form as the forward process [(Feller, 1949)](https://www.semanticscholar.org/paper/On-the-Theory-of-Stochastic-Processes%2C-with-to-Feller/4cdcf495232f3ec44183dc74cd8eca4b44c2de64), which means that $q(x_{t-1}|x_{t})$ will also be Gaussian. Therefore, to approximate $q(x_{t-1}|x_{t})$, our model $p_\theta$ only needs to estimate the Gaussian parameters $\mu_\theta(x_t, t)$ and $\Sigma_\theta(x_t, t)$ for timestep $t$.

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$

Note:

- $p_\theta$ not only takes in $x_t$, but also $t$, as inputs because each time step is associated with different noise levels due to the non-linear $\beta_t$.

### Training the Neural Network

In summary, we have defined the forward trajectory as a steady noisification of the sample distribution over time and the reverse trajectory as “tracing back” these steps to recover the original distribution. 

But how exactly do we teach our neural network to approximate the conditional probabilities for each time step in the reverse trajectory? To do so, we need to define a loss function. 

Naively, we can use a maximum likelihood objective where we maximise the likelihood assigned to $x_0$ by the model, i.e.

$$
\begin{align} \notag
p_\theta(x_0) &= \int p_\theta(x_{0:T})dx_{1:T}  \\\ \notag
L &= -\log(p_\theta(x_0))
\end{align}
$$

This objective is unfortunately intractable as it requires us to marginalize over all possible trajectories we could have taken from $x_{1:T}$.

Rather, we can take inspiration from Variational Autoencoders (VAE) and reformulate the training objective using a variational lower bound (VLB), also known as “evidence lower bound” (ELBO).

$$
\begin{aligned}\log p_\theta(x_0)
&\geq \log p_\theta(x_0) - D_\text{KL}(q(x_{1:T} | x_0) \| p_\theta(x_{1:T} | x_0) ) \\
&= \log p_\theta(x_0) - \mathbb{E}_{q(x_{1:T} | x_0)} \Big[ \log\frac{q(x_{1:T} | x_0)}{\frac{p_\theta(x_{0:T})}{p_\theta(x_0)}} \Big] \\
&= \log p_\theta(x_0) - \mathbb{E}_{q(x_{1:T} | x_0)} \Big[ \log\frac{q(x_{1:T} | x_0)}{p_\theta(x_{0:T})} + \log p_\theta(x_0) \Big] \\
&= - \mathbb{E}_{q(x_{1:T} | x_0)} \Big[ \log\frac{q(x_{1:T} | x_0)}{p_\theta(x_{0:T})}  \Big] \\
&= \mathbb{E}_{q(x_{1:T} | x_0)} \Big[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} | x_0)}  \Big]
\end{aligned}
$$

Therefore, the last term becomes the VLB of the likelihood assigned to $x_0$, a proxy objective to maximise. However, this VLB term is still not tractable so further reformulations is needed. Before we proceed, it is important to note that we can rewrite each transition as $q(x_{t}|x_{t-1}) = q(x_{t}|x_{t-1}, x_0)$, where the extra conditioning term is superfluous due to the Markov property. Using Bayes’ rule, we can rewrite each transition as:

$$
q(x_{t}|x_{t-1}, x_0) = \frac{q(x_{t-1}|x_{t}, x_0)q(x_{t}|x_0)}{q(x_{t-1}|x_0)}
$$

This trick will be useful for reducing the variance and derive a more elegant variational lower bound expression. Continuing from where we left off earlier, we have:

$$
\begin{aligned}\log p_\theta(x_0)
&\geq\mathbb{E}_{q(x_{1:T} | x_0)} \Big[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} | x_0)}  \Big] \\
&= \mathbb{E}_{q(x_{1:T} | x_0)} \Big[\log \frac{p_\theta(x_{T})\prod_{t=1}^T p_\theta(x_{t-1} | x_{t})}{\prod_{t=1}^T q(x_{t} | x_{t-1})}  \Big] \\
&= \mathbb{E}_{q(x_{1:T} | x_0)} \Big[\log \frac{p_\theta(x_{T})p_\theta(x_0 | x_1)\prod_{t=2}^T p_\theta(x_{t-1} | x_{t})}{q(x_1 |x_0)\prod_{t=2}^T q(x_{t} | x_{t-1})}  \Big] \\
&= \mathbb{E}_{q(x_{1:T} | x_0)} \Big[\log \frac{p_\theta(x_{T})p_\theta(x_0 | x_1)\prod_{t=2}^T p_\theta(x_{t-1} | x_{t})}{q(x_1 |x_0)\prod_{t=2}^T q(x_{t} | x_{t-1}, x_0)}  \Big] \\
&= \mathbb{E}_{q(x_{1:T} | x_0)} \Big[\log \frac{p_\theta(x_{T})p_\theta(x_0 | x_1)}{q(x_1 |x_0)} + \log\prod_{t=2}^T \frac{p_\theta(x_{t-1} | x_{t})}{q(x_{t} | x_{t-1}, x_0)}  \Big] \\
&= \mathbb{E}_{q(x_{1:T} | x_0)} \Big[\log \frac{p_\theta(x_{T})p_\theta(x_0 | x_1)}{q(x_1 |x_0)} + \log\prod_{t=2}^T \frac{p_\theta(x_{t-1} | x_{t})}{\frac{q(x_{t-1}|x_{t}, x_0)q(x_{t}|x_0)}{q(x_{t-1}|x_0)}}  \Big] \\
&= \mathbb{E}_{q(x_{1:T} | x_0)} \Big[\log \frac{p_\theta(x_{T})p_\theta(x_0 | x_1)}{q(x_1 |x_0)} + \log\prod_{t=2}^T \frac{\cancel{q(x_{t-1}|x_0)}}{\cancel{q(x_{t}|x_0)}} + \log\prod_{t=2}^T \frac{p_\theta(x_{t-1} | x_{t})}{{q(x_{t-1}|x_{t}, x_0)}}  \Big] \\
&= \mathbb{E}_{q(x_{1:T} | x_0)} \Big[\log \frac{p_\theta(x_{T})p_\theta(x_0 | x_1)}{\cancel{q(x_1 |x_0)}} + \log \frac{\cancel{q(x_{1}|x_0)}}{{q(x_{T}|x_0)}} + \log\prod_{t=2}^T \frac{p_\theta(x_{t-1} | x_{t})}{{q(x_{t-1}|x_{t}, x_0)}}  \Big] \\
&= \mathbb{E}_{q(x_{1:T} | x_0)} \Big[\log \frac{p_\theta(x_{T})p_\theta(x_0 | x_1)}{{q(x_{T}|x_0)}} + \log\sum_{t=2}^T \frac{p_\theta(x_{t-1} | x_{t})}{{q(x_{t-1}|x_{t}, x_0)}}  \Big] \\
&= \mathbb{E}_{q(x_{1:T} | x_0)}[\log p_\theta(x_0 | x_1)] + \mathbb{E}_{q(x_{1:T} | x_0)}\Big[\log \frac{p_\theta(x_{T})}{{q(x_{T}|x_0)}} \Big] + \log\sum_{t=2}^T \mathbb{E}_{q(x_{1:T} | x_0)} \Big[\frac{p_\theta(x_{t-1} | x_{t})}{{q(x_{t-1}|x_{t}, x_0)}}  \Big] \\
&= \mathbb{E}_{q(x_{1} | x_0)}[\log p_\theta(x_0 | x_1)] + \mathbb{E}_{q(x_{T} | x_0)}\Big[\log \frac{p_\theta(x_{T})}{{q(x_{T}|x_0)}} \Big] + \log\sum_{t=2}^T \mathbb{E}_{q(x_{t}, x_{t-1} | x_0)} \Big[\frac{p_\theta(x_{t-1} | x_{t})}{{q(x_{t-1}|x_{t}, x_0)}}  \Big] \\
&= \underbrace{\mathbb{E}_{q(x_{1} | x_0)}[\log p_\theta(x_0 | x_1)]}_{\text{reconstruction term}} - \underbrace{D_\text{KL} (q(x_{T}|x_0) \parallel p_\theta(x_{T}))}_{\text{prior matching term}}  -\sum_{t=2}^T \mathbb{E}_{q(x_{t} | x_0)} [\underbrace{D_\text{KL} ({q(x_{t-1}|x_{t}, x_0)} \parallel p_\theta(x_{t-1} | x_{t}))}_{\text{denoising matching term}}]   \\
\end{aligned}
$$

- $D_\text{KL} (q(x_{T}|x_0) \parallel p_\theta(x_{T}))$ is a constant because $q$ has no trainable parameters and $p_\theta(x_T)$ is a standard Gaussian. Therefore, we can ignore it.
- $\mathbb{E}_{q(x_{1} | x_0)}[\log p_\theta(x_0 | x_1)]$ is the reconstruction term which predicts the log likelihood of the original data sample given the first-step latent. Since the original data distribution may not be Gaussian, we cannot compute this in closed form. We will approximate and optimize this term using a Monte Carlo estimate.
- $D_\text{KL} ({q(x_{t-1}|x_{t}, x_0)} \parallel p_\theta(x_{t-1} | x_{t}))$ measures the KL divergence between the learnt transition step $p_\theta(x_{t-1} | x_{t})$ and the ground-truth denoising transition step $q(x_{t-1}|x_{t}, x_0)$. $q(x_{t-1}|x_{t}, x_0)$ can act as a ground-truth signal because it defines how to denoise a noisy image $x_t$ with access to what the final, completely denoised image $x_0$ should be.

Given that the denoising matching term is the only term we are interested in, to maximise the likelihood objective, we will need to minimize the KL divergence between the learnt denoising step and ground-truth denoising step. By Bayes’ Rule, we can reformulate the ground-truth denoising transition step as such:

$$
q(x_{t-1}|x_{t}, x_0) = \frac{q(x_{t}|x_{t-1}, x_0)q(x_{t-1}|x_0)}{q(x_{t}| x_0)}
$$

Using our definition of the forward transition step and the Markov property, we already know that $q(x_{t}|x_{t-1}, x_0) = q(x_{t}|x_{t-1}) = \mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1}, \beta_t\mathbf{I})$. Let $\alpha_t = 1 - \beta_t$, under the reparameterization trick used in VAEs, samples $x_t \sim q(x_t|x_{t-1})$ can be rewritten as:

$$
x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1- \alpha_t}\epsilon
$$

where $\epsilon \sim \mathcal{N}(\epsilon; 0, \mathbf{I})$. To express $q(x_{t}| x_0)$ in closed form, we recursively apply the reparameterization trick. That is, for any $x_t \sim q(x_t | x_0)$,

$$
\begin{align} \notag
x_t &= \sqrt{\alpha_t}x_{t-1} + \sqrt{1- \alpha_t}\epsilon^*_{t-1} \\\ \notag
&= \sqrt{\alpha_t a_{t-1}}x_{t-2} + \sqrt{\alpha_t - \alpha_t a_{t-1}}\epsilon^*_{t-2} + \sqrt{1-\alpha_t}\epsilon^*_{t-1} \\\
&=  \sqrt{\alpha_t a_{t-1}}x_{t-2} + \sqrt{\alpha_t - \alpha_t a_{t-1} + 1 - \alpha_t}\epsilon_{t-2} \\\ \notag
&= \dots \\\ \notag
&= \sqrt{\prod_{i=1}^t a_i}x_0 + \sqrt{1-\prod_{i=1}^t a_i}\epsilon_0 \\\ \notag
&= \sqrt{\bar{a}_t}x_0 + \sqrt{1-\bar{a}_t}\epsilon_0 \\\ \notag
&\sim \mathcal{N}(x_t; \sqrt{\bar{a}_t}x_0, (1-\bar{a}_t)\mathbf{I})
\end{align}
$$

where $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$. We can merge two Gaussians in (1) since the sum of two independent Gaussian random variables is a Gaussian with mean being the sum of the two means and variance being the sum of the two variances. We have therefore derived $q(x_{t}| x_0)$ and we can reuse the parameterization trick to yield $q(x_{t-1}| x_0) = \mathcal{N}(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}}x_0, (1-\bar{\alpha}_{t-1})\mathbf{I})$. Substituting both expressions into the Bayes rule expansion of the ground truth denoising step (intermediate steps have been omitted for brevity):

$$
\begin{aligned}
q(x_{t-1}|x_{t}, x_0)
&= \frac{q(x_{t}|x_{t-1}, x_0)q(x_{t-1}|x_0)}{q(x_{t}| x_0)}\\
&= {\frac{\mathcal{N}(x_{t} ; \sqrt{\alpha_t} x_{t-1}, (1 - \alpha_t)\textbf{I})\mathcal{N}(x_{t-1} ; \sqrt{\bar\alpha_{t-1}}x_0, (1 - \bar\alpha_{t-1}) \textbf{I})}{\mathcal{N}(x_{t} ; \sqrt{\bar\alpha_{t}}x_0, (1 - \bar\alpha_{t})\textbf{I})}}\\
&\propto {\text{exp}\left\{-\left[\frac{(x_{t} - \sqrt{\alpha_t} x_{t-1})^2}{2(1 - \alpha_t)} + \frac{(x_{t-1} - \sqrt{\bar\alpha_{t-1}} x_0)^2}{2(1 - \bar\alpha_{t-1})} - \frac{(x_{t} - \sqrt{\bar\alpha_t} x_{0})^2}{2(1 - \bar\alpha_t)} \right]\right\}}\\
&= \dots \\
&\propto {\mathcal{N}(x_{t-1} ;} \underbrace{{\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})x_{t} + \sqrt{\bar\alpha_{t-1}}(1-\alpha_t)x_0}{1 -\bar\alpha_{t}}}}_{\mu_q(x_t, x_0)}, \underbrace{{\frac{(1 - \alpha_t)(1 - \bar\alpha_{t-1})}{1 -\bar\alpha_{t}}\textbf{I}}}_{\bm{\Sigma}_q(t)})
\end{aligned}
$$

We have therefore shown that at each step, $x_{t-1} \sim q(x_{t-1}|x_{t}, x_0)$ follows a normal distribution with mean $\mu_q(x_t, x_0)$, a function of $x_t$ and $x_0$, and variance $\Sigma_q(t)$, a function of $\alpha$ coefficients. We can further leverage the reparameterization trick to express $x_0$ as $\epsilon_0$:

$$
\begin{aligned}
\mu_q(x_t, x_0) &= {\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})x_{t} + \sqrt{\bar\alpha_{t-1}}(1-\alpha_t)x_0}{1 -\bar\alpha_{t}}} \\
&= {\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})x_{t} + \sqrt{\bar\alpha_{t-1}}(1-\alpha_t)\frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_0}{\sqrt{\bar{\alpha}_t}}}{1 -\bar\alpha_{t}}} \\
&= \dots \\
&= \frac{1}{\sqrt{\alpha_t}}x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}\sqrt{\alpha_t}}\epsilon_0
\end{aligned}
$$

Following [Ho et al. (2020)](https://arxiv.org/abs/2006.11239), we can set $\Sigma_q(t)$  as a constant at each timestep by modelling the $\alpha$ coefficients as fixed hyperparameters. With that, we can rewrite the variance equation as $\Sigma_q(t) = \sigma^2_q(t)\mathbf{I}$, where:

$$
\sigma^2_q(t) = \frac{(1 - \alpha_t)(1 - \bar\alpha_{t-1})}{1 -\bar\alpha_{t}}
$$

As we have kept the variance constant, minimizing the KL divergence is simply minimizing the difference between $\mu_q(x_t, x_0)$ and $\mu_\theta(x_t, t)$. Note that we have no choice but to parameterize the mean of the learnt model $\mu_\theta(x_t, t)$ as a function of $x_t$ since $p_\theta(x_{t-1}|x_t)$ does not depend on $x_0$. 

The tue denoising transition mean is expressed as:

$$
\mu_q(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}}x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}\sqrt{\alpha_t}}\epsilon_0
$$

To optimize the model’s mean $\mu_\theta(x_t, t)$, we set it to have the following form:

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}\sqrt{\alpha_t}}\hat{\epsilon_\theta}(x_t, t)
$$

where $\hat{\epsilon}_\theta(x_t, t)$ is parameterized by a neural network that seeks to predict the source noise $\epsilon_0 \sim \mathcal{N}(\epsilon_0; 0, \mathbf{I})$ that lead to $x_t$ from $x_0$. As such, the optimization problem simplifies to:

$$
\begin{aligned}
&\quad \argmin_{{\theta}} D_\text{KL}({q(x_{t-1}|x_t, x_0)}\parallel{p_{{\theta}}(x_{t-1}|x_t)}) \\
&= \argmin_{{\theta}} D_\text{KL}({\mathcal{N}\left(x_{t-1}; {\mu}_q,{\Sigma}_q\left(t\right)\right)} \parallel {\mathcal{N}\left(x_{t-1}; {\mu}_{{\theta}},{\Sigma}_q\left(t\right)\right)})\\
&=\argmin_{{\theta}}\frac{1}{2\sigma_q^2(t)}\left[\left\lVert\frac{1}{\sqrt{\alpha_t}}x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar\alpha_t}\sqrt{\alpha_t}}{\hat\epsilon}_{{\theta}}(x_t, t) -
\frac{1}{\sqrt{\alpha_t}}x_t + \frac{1 - \alpha_t}{\sqrt{1 - \bar\alpha_t}\sqrt{\alpha_t}}{\epsilon}_0\right\rVert^2\right]\\
&=\argmin_{{\theta}}\frac{1}{2\sigma_q^2(t)}\left[\left\lVert \frac{1 - \alpha_t}{\sqrt{1 - \bar\alpha_t}\sqrt{\alpha_t}}{\epsilon}_0 - \frac{1 - \alpha_t}{\sqrt{1 - \bar\alpha_t}\sqrt{\alpha_t}}{\hat\epsilon}_{{\theta}}(x_t, t)\right\rVert^2\right]\\
&=\argmin_{{\theta}}\frac{1}{2\sigma_q^2(t)}\left[\left\lVert \frac{1 - \alpha_t}{\sqrt{1 - \bar\alpha_t}\sqrt{\alpha_t}}({\epsilon}_0 - {\hat\epsilon}_{{\theta}}(x_t, t))\right\rVert^2\right]\\
&=\argmin_{{\theta}}\frac{1}{2\sigma_q^2(t)}\frac{(1 - \alpha_t)^2}{(1 - \bar\alpha_t)\alpha_t}\left[\left\lVert{\epsilon}_0 - {\hat\epsilon}_{{\theta}}(x_t, t)\right\rVert^2\right]
\end{aligned}
$$

We can formulate our variational lower bound loss function as

$$
L = \mathbb{E}_{x_0, \epsilon} \Big[ \frac{(1 - \alpha_t)^2}{2\sigma_q^2(t)(1 - \bar\alpha_t)\alpha_t}\left \lVert{\epsilon}_0 - {\hat\epsilon}_{{\theta}}(x_t, t)\right\rVert^2 \Big]
$$

Empirically, [Ho et al. (2020)](https://arxiv.org/abs/2006.11239) found that a simplified loss function without the weighting term performs better:

$$
L_{\text{simple}}  = \mathbb{E}_{x_0, \epsilon} \Big[\|{\epsilon}_0 - {\hat{\epsilon}}_\theta(x_t, t \|^2 \Big]
$$

which is basically just the **mean squared error between the noise added in the forward process and the noise predicted by the model**!

![Training and sample algorithm from [Ho et al. (2020)](https://arxiv.org/abs/2006.11239)](Images/Untitled%202.png)

Training and sample algorithm from [Ho et al. (2020)](https://arxiv.org/abs/2006.11239)

## Summary
To wrap up, diffusion models represent a fascinating approach to data generation and denoising tasks in the realm of deep learning. At their core, these models leverage a process called diffusion, which involves progressively applying Gaussian noise to an original sample distribution. The diffusion model then aims to learn the parameters that maximize the Variational Lower Bound (VLB) during the denoising step. 

Notably, diffusion models often excel in terms of generalizability and training stability, steering clear of pitfalls like mode collapse commonly encountered by GANs. However, GANs are still considerably more computationally efficient compared to diffusion models. This is due to the fact that GANs can generate an image in a single forward pass but diffusion models rely on a long Markov chain of denoising steps.

Still, diffusion models have proved their incredible capabilities in generating realistic samples. It's highly likely that we'll witness the emergence of even more powerful diffusion-based generative models, pushing the boundaries of generative modeling to new heights in the years ahead!

## References

[1] Goodfellow, Ian & Pouget-Abadie, Jean & Mirza, Mehdi & Xu, Bing & Warde-Farley, David & Ozair, Sherjil & Courville, Aaron & Bengio, Y.. (2014). Generative Adversarial Networks. Advances in Neural Information Processing Systems. 3. 10.1145/3422622. 

[2] Dhariwal, Prafulla & Nichol, Alex. (2021). Diffusion Models Beat GANs on Image Synthesis. 

[3] Zhang, H., Xu, T., Li, H., Zhang, S., Wang, X., Huang, X., & Metaxas, D.N. (2016). StackGAN: Text to Photo-Realistic Image Synthesis with Stacked Generative Adversarial Networks. *2017 IEEE International Conference on Computer Vision (ICCV)*, 5908-5916.

[4] Xu, T., Zhang, P., Huang, Q., Zhang, H., Gan, Z., Huang, X., & He, X. (2017). AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks. *2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 1316-1324.

[5] Shoshan, A., Bhonker, N., Kviatkovsky, I., & Medioni, G.G. (2021). GAN-Control: Explicitly Controllable GANs. *2021 IEEE/CVF International Conference on Computer Vision (ICCV)*, 14063-14073.

[6] Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N. & Ganguli, S.. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics. Proceedings of the 32nd International Conference on Machine Learning, in Proceedings of Machine Learning Research 37:2256-2265

[7] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *ArXiv, abs/2006.11239*.

[8] Feller, W. (1949). On the Theory of Stochastic Processes, with Particular Reference to Applications.

[9] Weng, Lilian. (Jul 2021). What are diffusion models? Lil’Log. [https://lilianweng.github.io/posts/2021-07-11-diffusion-models/](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

[10] Luo, C. (2022). Understanding Diffusion Models: A Unified Perspective. *ArXiv, abs/2208.11970*.