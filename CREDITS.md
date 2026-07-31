# Credits

This project is a **reproduction**. The method, the implementation, the environment and the
tooling are other people's work. Our contribution is the CoinRun reproduction, the fixes needed
to make it run end to end, and the measurements.

---

## The implementation we built on

**Open Dreamer** — <https://github.com/next-state/open-dreamer>
Diego Marti Monso, Francesco Sacco, Edward Hu · contributions from Dere-Wah

Their JAX/Flax implementation of Dreamer 4 is what we actually ran: the model code, the configs,
the training and evaluation scripts. Their requested citation:

```bibtex
@misc{marti2026opendreamer,
  title  = {How to Train a Frontier-level World Model},
  author = {Marti Monso, Diego and Sacco, Francesco and Hu, Edward},
  month  = {jul}, year = {2026}, publisher = {Zenodo},
  doi    = {10.5281/zenodo.21475232},
  url    = {https://next-state.github.io/open-dreamer/}
}
```

Inference harness: <https://github.com/reactor-team/open-dreamer>

> ### ⚠️ Licence status — important
> Open Dreamer ships an explicit **all-rights-reserved** notice:
> *"No license or permission is granted, whether express or implied, to use, copy, modify, merge,
> publish, distribute, sublicense, or sell copies of this software, in whole or in part, without
> the prior written consent of the copyright holders."*
> It adds: *"This notice is provisional: it is expected to be replaced by a formal license in a
> future release."*
>
> Consequences for us:
> - We do **not** redistribute their code, or any weights trained with it.
> - We publish our **measurements, figures and write-up** only.
> - Releasing our pipeline/weights is **gated on a licence that permits it** (or written consent).

---

## The method

**Dreamer 4 — Training Agents Inside of Scalable World Models**
Danijar Hafner, Wilson Yan, Timothy Lillicrap · [arXiv:2509.24527](https://arxiv.org/abs/2509.24527)
The causal tokenizer + action-conditioned dynamics architecture we trained.

**One Step Diffusion via Shortcut Models**
Kevin Frans, Danijar Hafner, Sergey Levine, Pieter Abbeel · ICLR 2025 ·
[arXiv:2410.12557](https://arxiv.org/abs/2410.12557)
Conditioning the flow on step size as well as noise level — the mechanism behind few-step
(real-time-capable) generation, and the `bootstrap` phase of our training run.

**Genie: Generative Interactive Environments**
Jake Bruce et al., Google DeepMind · ICML 2024 (best paper) ·
[arXiv:2402.15391](https://arxiv.org/abs/2402.15391)
The original idea, our first architecture, and the tokenizer PSNR (35.7) we compare against.

---

## Baseline and earlier build

**GenieRedux / Exploration-Driven Generative Interactive Environments**
Nedko Savov, Naser Kazemi, Mohammad Mahdi, Danda Pani Paudel, Xi Wang, Luc Van Gool — INSAIT · CVPR 2025
<https://github.com/insait-institute/GenieRedux>
Our first world model was built on GenieRedux, and its CoinRun tokenizer (PSNR 38.25) is our
main reference point. MIT licensed.

---

## Environment and data

**Procgen Benchmark / CoinRun** — Karl Cobbe and collaborators, OpenAI · MIT ·
<https://github.com/openai/procgen>
Every one of our 9.6M training frames was rendered by procgen. No scraped video is used anywhere
in this project.

---

## Infrastructure and libraries

| | |
|---|---|
| **Modal** | GPU training and evaluation — <https://modal.com> |
| **JAX / Flax (NNX)** | Google DeepMind — the framework Open Dreamer is written in |
| **Optax** | optimisation, including the Muon optimiser used for the dynamics model |
| **Grain / ArrayRecord** | Google — data loading and the shard format |
| **ott-jax** | optimal-transport coupling used in the flow-matching objective |
| **LPIPS** (jaxlpips) | perceptual loss in tokenizer training — Zhang et al. |
| **FVD / I3D** | video-quality metric used in our evaluation — Unterthiner et al. |
| **Remotion** | the announcement videos — <https://remotion.dev> |
| **Gemini** | generation of the process figures |
| **Fraunces · Inter · JetBrains Mono** | typefaces used on the site |

---

## What is ours

- The procgen → ArrayRecord **data engine** (9.6M frames, self-generated, action-labelled)
- The **Modal training harness** (`modal_dreamer.py`)
- **Five fixes** to make the CoinRun path run end to end (see `OPEN_DREAMER_NOTES.md`)
- The CoinRun **configuration choices** (`n_latents` 32, batch sizes, 80k/40k schedule)
- The **measurements**: tokenizer PSNR 40.41, FVD 32.19 / 16.59 / 23.29, 144-frame rollouts
- The **write-up, figures and videos**

Reproduction, fixes and measurements by **Vizuara AI Labs**.
