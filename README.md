<div align="center">

# Training a Dreamer 4 world model from scratch on CoinRun

**A world model that dreams a playable platformer, frame by frame — trained end to end
on 9.6M frames we generated ourselves, on one GPU, for about $150.**

[**Read the write-up →**](https://site-dreamer.vercel.app) ·
[Rollouts](rollouts/) · [The five things that broke](OPEN_DREAMER_NOTES.md) · [Credits](CREDITS.md)

| tokenizer PSNR | end-to-end FVD | rollout | cost |
|:---:|:---:|:---:|:---:|
| **40.41** | **32.19** | **144 frames, no collapse** | **~$150** |

</div>

---

## ⚠️ Read this first — what this repo is, and is not

**This is not a redistribution of Open Dreamer.**

We trained on [`next-state/open-dreamer`](https://github.com/next-state/open-dreamer), which ships an
explicit **all-rights-reserved** licence:

> *"No license or permission is granted, whether express or implied, to use, copy, modify, merge,
> publish, distribute, sublicense, or sell copies of this software, in whole or in part, without the
> prior written consent of the copyright holders."* (marked provisional)

Out of respect for that, this repo contains **none of their code and no weights derived from it**.
`open-dreamer/` is in `.gitignore` — you clone it yourself.

What **is** here is our own work, MIT licensed:

- `modal_dreamer.py` — the Modal harness (data engine, training, eval, the patches)
- `make_charts.py`, `gen_figs.sh` — figures from our real training logs
- `site-dreamer/` — the write-up site
- `rollouts/` — generated rollouts + ground-truth comparisons
- `announcement/` — Remotion announcement films
- the `.md` files — the full recipe, every number, and every failure

## What we did

Trained the full [Dreamer 4](https://arxiv.org/abs/2509.24527) recipe on
[procgen CoinRun](https://github.com/openai/procgen):

1. **Data engine** — 10,000 CoinRun episodes rolled out on Modal CPU containers →
   **9.6M frames**, 160-frame records, with ground-truth action labels. No scraped video anywhere.
2. **Tokenizer** — causal video tokenizer (MAE, continuous latents, tanh bottleneck), 10k steps,
   1×H100, 1h35m. **PSNR 40.41.**
3. **Dynamics** — 1.57B-parameter action-conditioned transformer trained with flow matching +
   **shortcut forcing**, 80k steps on 1×H200. `flow_mse` 0.53 → 0.014.
4. **Eval** — FVD on a held-out test split, 144-frame rollouts from 4 context frames.

### Results

| metric | value | what it means |
|---|---|---|
| Tokenizer PSNR | **40.41** | vs GenieRedux 38.25, Genie paper 35.7 |
| FVD (original → dream) | **32.19** | end-to-end |
| FVD (original → recon) | 16.59 | tokenizer ceiling |
| FVD (recon → dream) | 23.29 | dynamics only |

The FVD decomposition is the useful bit: roughly half the error is the tokenizer's compression loss
and half is the dynamics model's prediction error — neither dominates.

### Still unmeasured

**Controllability.** Everything above shows the model *simulates* CoinRun well. It does **not** yet
prove that pressing → makes the character go right. The action-swap test is still outstanding, and
we say so rather than implying otherwise.

## The five things that broke

Their CoinRun path does not run end to end as shipped. All five found by reading the source:

1. `generate_coinrun_dataset.py` passes a kwarg `ShardWriter` doesn't accept — instant crash
2. That writer emits **msgpack**; the CoinRun reader does `pickle.loads` — incompatible
3. `tokenize_minecraft_dataset.py` hardcodes the Minecraft transform — no CoinRun path
4. `train_dynamics.py` asserts the **Minecraft action space** — fails immediately on CoinRun
5. `eval_fvd.py` writes MP4 via `plugin="pyav"` but only `imageio[ffmpeg]` is declared

Plus environment traps that cost real time — a CUDA base image making JAX **silently fall back to
CPU**, an 8×H200 **NCCL deadlock** that looked healthy while burning $36/hr, and Modal's 24h job cap.
All documented in [`OPEN_DREAMER_NOTES.md`](OPEN_DREAMER_NOTES.md).

## Reproducing it

You need: a Modal account, and your own clone of Open Dreamer (subject to their licence).

```bash
git clone https://github.com/next-state/open-dreamer.git   # their terms apply
pip install modal && modal setup

modal run --detach modal_dreamer.py::gen_data                                   # 9.6M frames, ~$15
modal run --detach modal_dreamer.py::train_tokenizer --steps 10000 --batch 32   # ~$6
modal run --detach modal_dreamer.py::latent_stats
modal run --detach modal_dreamer.py::pipeline                                   # dynamics + eval
```

Two things that will bite you: **use H200, not H100** (batch 16 OOMs by 26 GiB), and **stay on a
single GPU** — multi-GPU deadlocked on NCCL rendezvous for us.

## Credits

Built on [Open Dreamer](https://github.com/next-state/open-dreamer) by Diego Marti Monso,
Francesco Sacco and Edward Hu — an implementation of **Dreamer 4** (Hafner, Yan & Lillicrap), which
builds on **shortcut models** (Frans, Hafner, Levine & Abbeel). Environment:
[OpenAI Procgen](https://github.com/openai/procgen). Baseline:
[GenieRedux](https://github.com/insait-institute/GenieRedux) (INSAIT). Trained on
[Modal](https://modal.com). Full attributions in [CREDITS.md](CREDITS.md).

Reproduction, fixes and measurements by **Vizuara AI Labs**.
