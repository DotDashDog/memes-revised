# Research Note — Hawkes & Neural Spatio-Temporal Point Processes

**Status:** speculative (exploratory; nothing here is an accepted decision)
**Feeds:** `../open-questions.md` → "Holistic spatio-temporal point-process model"
**Relation to current plan:** This describes a possible *alternative/extension* to the cluster-then-fit-SIR pipeline. The cluster-SIR pipeline is the limiting (mean-field, independent-reservoir) case of the model described here. Do **not** build this before the cluster pipeline produces a baseline.

---

## 1. Why this is relevant

The project outline floats a "holistic" idea: instead of treating each meme as a cluster, model the whole stream of posts as a process on an underlying **rate field over embedding space**, and predict that field's evolution in time. That object is exactly the **conditional intensity of a marked spatio-temporal point process (STPP)**, and **Hawkes processes** are the contagion-flavored way to specify it. This note distills what that connection is and how one would parameterize it neurally.

## 2. Core concepts (established)

A temporal point process is fully characterized by its **conditional intensity**
$$\lambda^*(t) = \lambda(t \mid \mathcal{H}_t),\qquad \lambda^*(t)\,dt = \mathbb{E}[\#\text{events in }[t,t{+}dt) \mid \mathcal{H}_t].$$

- **Poisson:** $\lambda$ independent of history (constant or deterministic $\mu(t)$). No event-to-event influence.
- **Hawkes (self-exciting):** each past event raises the current intensity, then it relaxes:
$$\lambda^*(t) = \mu + \sum_{t_i<t}\phi(t-t_i),\qquad \phi\ge 0\ \text{(e.g. }\alpha e^{-\beta\tau}).$$
- **Marked / spatio-temporal Hawkes:** attach a mark $\mathbf{x}_i$ (for us, the post embedding); the intensity becomes a field over (embedding space) × time:
$$\lambda^*(t,\mathbf{x}) = \mu(\mathbf{x}) + \sum_{t_i<t}\phi\big((t-t_i),(\mathbf{x},\mathbf{x}_i)\big).$$
**This $\lambda^*(t,\mathbf{x})$ is the "rate field." Forecasting it forward is what the conditional intensity is for.**

### Branching interpretation (the key bridge)
A Hawkes process = a branching process: "immigrant" events arrive at rate $\mu$; each event spawns offspring via rate $\phi$. The **branching ratio** $n=\int_0^\infty\phi(\tau)\,d\tau$ is the Hawkes analogue of $R_0$:
- $n<1$ subcritical → outbreaks die out (the regime real memes live in).
- $n=1$ critical; $n>1$ supercritical → explosive.
The kernel timescale (e.g. $1/\beta$) is the analogue of the recovery timescale $1/\gamma$.

### Rate-field evolution (the "physics" intuition, established)
For the exponential kernel, between events the excitation $g(t)=\sum_{t_i<t}\alpha e^{-\beta(t-t_i)}$ obeys $\dot g = -\beta g$, with a jump $+\alpha$ at each event. So the rate field follows **piecewise-deterministic dynamics**: smooth ODE relaxation punctuated by jumps at event locations. The spatio-temporal version is a decay/diffusion operator over embedding space with point sources deposited at each new event. This is the closest thing to "a differential equation for the rate field."

## 3. Relation to SIR (established, with one important caveat)

Taking expectations of the linear Hawkes intensity gives a deterministic integro-differential equation for the **expected** rate field — a mean-field limit.

- **SIR** is self-exciting **with depletion**: $\dot I = \beta IS - \gamma I$; the shrinking susceptible pool $S$ bends the curve into a single hump and forces decay to zero.
- **Linear Hawkes has no depletion**, so it only reproduces the *early/contagion* phase and is stable only when subcritical; it does not naturally peak-and-decay from depletion.
- To get SIR-like rise–peak–fall you need a **finite-population / self-limiting Hawkes** ("SIR-Hawkes" / "HawkesN"). Its mean-field limit *is* the SIR ODE system.

**Takeaway:** cluster-SIR and the holistic Hawkes view are the microscopic vs. mean-field descriptions of the same contagion. SIR ≈ coarse-grained, independent-reservoir, depleting Hawkes.

## 4. Neural parameterization — the three families (established)

What we're choosing is *how history → $\lambda^*(t,\mathbf{x})$*.

- **Family A — Additive neural kernel.** Learn $\mu_\theta(\mathbf{x})$ and $\phi_\theta$ but keep the additive Hawkes form. Most interpretable: branching ratio and kernel survive as named quantities (the SIR bridge). Tractable-integral variants use low-rank / displacement-reparameterized kernels.
- **Family B — Autoregressive state-summary.** Encode history into a state $\mathbf{h}(t)$, set $\lambda^*(t,\mathbf{x})=g_\theta(\mathbf{h}(t),\mathbf{x})$. Encoders: RNN (RMTPP), continuous-time LSTM that decays between events (Neural Hawkes), or self-attention (Transformer / Self-Attentive Hawkes). Most scalable; attention naturally learns *which past memes influence which*.
- **Family C — Intensity-free / generative.** Skip $\lambda^*$; directly model the predictive density of the next event's (time, location). Includes mixture/flow densities (Intensity-Free), Neural-ODE + continuous-time normalizing flows (Neural STPP), and diffusion/score-matching STPPs. Sidesteps the normalizing integral.

## 5. Suggested starting architecture for the meme problem (our speculation)

**Family B encoder + Family A intensity head**, to keep an interpretable kernel while scaling:

1. **Event token:** $[\mathbf{x}_i \,\|\, \tau(t_i)\,\|\, \text{sub-id emb}\,\|\,\log(1{+}\text{upvotes}_i)]$. Engagement enters as a feature, not a hack on the rate.
2. **Encoder:** causally-masked self-attention → state $\mathbf{h}(t)$; window by time and/or by $k$-nearest past events in embedding space (a learned-locality assumption to ablate).
3. **Intensity head with explicit decomposition:**
$$\lambda^*(t,\mathbf{x}) = \mu_\theta(\mathbf{x}) + \sum_{t_i<t} a_\theta(\mathbf{h}(t),\mathbf{x}_i)\, s_\theta(\mathbf{x},\mathbf{x}_i)\, e^{-\beta_\theta(\mathbf{x}_i)(t-t_i)}.$$
   - $\mu_\theta$: background "birth" field. $s_\theta$: non-negative spatial kernel (sets meme granularity implicitly — no HDBSCAN). $\beta_\theta(\mathbf{x}_i)$: **per-content decay = learned analogue of $\gamma$, predicted from content** (the project goal in miniature). $a_\theta$: excitation magnitude.
4. **Tractability of the compensator integral** $\int_0^T\!\!\int \lambda^*\,d\mathbf{x}\,dt$ in high-$d$: the main obstacle. Options: analytic (Gaussian) kernels for closed-form spatial integral; go intensity-free (Family C); or MC / score-matching estimation. **Recommended:** define the kernel on a learned low-dim projection $\mathbf{z}=W\mathbf{x}$ ($\sim$8–16 d) so the spatial integral is doable. Directly attacks the curse-of-dimensionality concern.
5. **Objective:** TPP negative log-likelihood $\sum_i\log\lambda^*(t_i,\mathbf{x}_i) - \int_0^T\!\!\int_\mathcal{Z}\lambda^*\,d\mathbf{z}\,dt$. Evaluate via held-out future log-likelihood and next-event time/location error.

### SIR-Hawkes inductive bias (PINN-style hook)
Because the head exposes per-content decay and excitation, add a soft penalty $\mathcal{L} - \eta\,\mathcal{R}_{\text{SIR}}$ nudging local aggregate dynamics toward a depleting (SIR-Hawkes / HawkesN) law. $\eta\to\infty$ ≈ cluster-SIR; $\eta=0$ = free neural STPP; the preferred $\eta$ tells you how much SIR structure the data wants.

## 6. Why this is hard *for this project specifically* (our assessment)

1. **High-dimensional mark.** Existing STPP work uses low-d marks (2-D maps, a few event types). A $\sim10^3$-d embedding mark is at/past the frontier; the low-dim projection is load-bearing and unvalidated.
2. **Background vs. excitation identifiability** = the endogeneity problem (R2) in disguise ("happened because a similar one just did" vs. "baseline rate high here"). Hard even in 1-D.
3. **Corrupted event stream.** Reposts, mod deletions, posts ≠ infections (R5). NLL treats every observed event as a true draw; deletion is missing-not-at-random.
4. **Cost.** Million-event NLL with a compensator integral is expensive; windowing/locality approximations are necessary but unproven here.

## 7. If/when promoted to real work

Plausible entry experiments (each could become a work package):
- **WP-candidate:** SIR-Hawkes (HawkesN) mean-field derivation + synthetic check that its mean-field limit reproduces SIR fits — connects directly to WP-02.
- **WP-candidate:** 1-D temporal-only neural Hawkes on a *single known meme's* post stream, to validate the likelihood/training machinery before adding marks.
- **WP-candidate:** low-dim-projection + Gaussian-kernel STPP on a small embedding subset, measuring whether held-out future log-likelihood beats independent per-cluster SIR.

## 8. Curated literature (verified)

Foundations & surveys:
- Hawkes (1971), *Spectra of some self-exciting and mutually exciting point processes.*
- Shchur, Türkmen, Januschowski, Günnemann (2021), *Neural Temporal Point Processes: A Review*, IJCAI. (temporal-only taxonomy)
- *Deep Spatio-temporal Point Processes: Advances and New Directions* (2025), arXiv:2504.06364; Mukherjee et al. (2025), *Neural Spatiotemporal Point Processes: Trends and Challenges.* (spatio-temporal taxonomy)
- Benchmark: *On the Predictive Accuracy of Neural TPP Models* (arXiv:2306.17066).

Encoders (Family B):
- Du et al. (2016), *Recurrent Marked Temporal Point Processes* (RMTPP), KDD.
- Mei & Eisner (2017), *The Neural Hawkes Process*, NeurIPS (continuous-time LSTM).
- Zuo et al. (2020), *Transformer Hawkes Process*, ICML; Zhang et al. (2020), *Self-Attentive Hawkes Process*, ICML.

Spatio-temporal / kernel / generative (Families A, C):
- **Chen, Amos, Nickel (2021), *Neural Spatio-Temporal Point Processes*, ICLR** — canonical neural STPP (Neural ODE + continuous-time normalizing flows); code: `facebookresearch/neural_stpp`. **Start here.**
- Jia & Benson (2019), *Neural Jump SDEs* (basis for the above).
- Zhu et al. (2022) and *Spatio-temporal point processes with deep non-stationary kernels* (DNSK, arXiv:2211.11179) — tractable low-rank neural kernels (Family A).
- Shchur et al. (2020), *Intensity-Free Learning of Temporal Point Processes*, ICLR.
- Yuan et al. (2023), *Spatio-temporal Diffusion Point Processes*, KDD; recent score-matching neural STPPs (likelihood-free + uncertainty).

Application precedent (social-media cascades / popularity):
- Cao et al. (2017), *DeepHawkes*, CIKM.
- *Dynamic Hawkes Processes* (DHP, arXiv:2105.11152) — latent time-evolving state modulating the kernel; closest to the depleting-reservoir idea.
- TCAN (IPM 2023) and related cascade-popularity attention models.
