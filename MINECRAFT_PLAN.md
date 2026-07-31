# Minecraft via Open Dreamer — feasibility & staged plan

**Short answer: yes, and Rajat's instinct was right — Minecraft is the repo's *native* path, not a port.
But the default config will not fit on one GPU, and that is the real blocker.**

---

## 1. Why this is easier than CoinRun was

Every CoinRun breakage we hand-patched either **vanishes** or is **already solved by code we wrote**:

| CoinRun breakage | Minecraft status |
|---|---|
| `generate_coinrun_dataset.py` crashes | N/A — Minecraft has its own path |
| msgpack-vs-pickle writer/reader mismatch | Same bug exists → **reuse our pickle writer** |
| No CoinRun tokenization script | `tokenize_minecraft_dataset.py` is Minecraft-only **by design** |
| Action-space asserts fail | **Pass natively** (27 binary / 121 camera classes) |

Configs default to Minecraft (`configs/tokenizer.yaml:10`, `dynamics.yaml:9`). Resolution matches
**exactly** — VPT ships 640×360 @ 20 fps h264; the config expects 360×640 padded to 368×640, patch 16.
No resampling.

**And we have already run the identical dynamics model.** Our CoinRun run passed no dynamics overrides,
so it used the *Minecraft* architecture: depth 30, d_model 1920, packing_factor 2 = **1.5704 B params**
(matches our logged 1.57 B exactly). Only `n_latents` (32→512) and `long_T` (64→256) change.

## 2. The data — free, public, action-labelled

Five contractor indexes at `openaipublic.blob.core.windows.net/minecraft-rl/snapshots/<name>.json`:

| Index | Task | Segments | Hours | Size |
|---|---|---|---|---|
| `all_6xx_Jun_29` | free play | 71,868 | ~1,000 | 2.4 TB |
| `all_7xx_Apr_6` | free play (long) | 17,886 | ~1,360 | 3.8 TB |
| `all_8xx_Jun_29` | build house from scratch | 5,096 | ~354 | 0.8 TB |
| `all_9xx_Jun_29` | build house from materials | 8,698 | ~587 | 1.3 TB |
| `all_10xx_Jun_29` | obtain diamond pickaxe | 5,661 | ~348 | 0.8 TB |
| **total** | | **109,209** | **≈3,650 h** | **≈9.1 TB** |

The Open Dreamer authors trained on **≈1,580 h** (derived from `index_max: 89` × 5,000 records ×
256 frames ÷ 20 fps) — i.e. **43% of the public set**. We could match or exceed them.

> The ~70k hours of YouTube-scraped VPT video is **NOT distributed**. Only the paid-contractor data is
> downloadable — the cleanest provenance available for Minecraft.

## 3. 🔴 The real blocker: activation memory

| Config | tokens/step | vs CoinRun |
|---|---|---|
| Our CoinRun run (B8, T64, S50) — **needed an H200** | 25,600 | 1× |
| CoinRun at B16 — **OOM'd on H100** (26 GiB short) | 51,200 | 2× |
| **Minecraft default (B8, T256, S290)** | **593,920** | **23×** |

Stored block inputs alone ≈ **68 GB** across 30 layers, before optimizer state (~25 GB).
**The default config will not fit on a single H200.**

Options, none free:
- **Cut `long_T` 256→64 and B 8→2** — fits, but loses long-context training and needs 16× more steps
- **Multi-GPU FSDP** — plausible, **but this is our one known-unfixed failure**: the 8×H200 NCCL
  `Acquire clique` deadlock at step 4187, silent, while Modal reported healthy
- **`n_latents` 512→256** — halves cost, departs from the published recipe, needs its own tokenizer

Compounding it: Modal's 24 h cap means **~40–50 sequential resumes** for 200k steps at the default
config — 40 chances to hit that silent stall.

## 4. Cost

| Item | Estimate |
|---|---|
| Tokenizer (B32, 50k steps) | $400–900 |
| Dynamics, default config, 200k steps | $4,200–5,700 |
| Dynamics, reduced (T=64) | $780–1,400 |
| Storage (author-parity latents) | ~$75/mo (raw + latents ≈ $450/mo) |
| **Credible end-to-end run** | **$5,000–7,500** |

Latents are **1.18 GB per video-hour** (16 KB/frame) — a 16× per-frame blowup vs our CoinRun setup.

## 5. Converter — the one piece of net-new code (~300 lines)

No Minecraft converter exists in the repo. Spec: fetch index → download `.mp4` + `.jsonl` → align →
chunk to 256 frames → re-encode each chunk to self-contained mp4 → store **raw VPT action dicts**
(parsed at load time by `transforms.py:313`) → write with our pickle writer.

**Verified gotcha:** on a real segment the mp4 has **1201 frames, the jsonl has 1200 lines.**
Systematic, not a fluke. `transforms.py:309-321` slices actions with video indices ⇒ silent
misalignment. **Truncate both to the action count**, and re-verify with decord (whose frame count
can disagree with ffprobe).

Slim the action dicts — `inventory`/`stats`/`xpos` are 90%+ of jsonl bytes and are never read.

## 6. Legal

| Scenario | Minecraft risk | open-dreamer risk |
|---|---|---|
| Internal research training | **Low** — not "shared" ⇒ not commercial use under Mojang's own definition; contractor data is consented | **Low-med** — technically unlicensed |
| Publish paper / blog | **Low** — huge unchallenged prior art | **Low** — citation is what they ask |
| Publish weights | **Medium** — arguably embeds derivatives of Mojang assets | **Med-high** |
| Ship playable demo | **High** | **High — blocking** |

- **VPT is MIT** for the *code*, and OpenAI explicitly disclaims Minecraft rights:
  *"Minecraft is the intellectual property of Microsoft, not OpenAI. OpenAI is not purporting to
  license any intellectual property rights with respect to Minecraft."* The MIT license covers the
  code, not the pixels.
- **Mojang's Usage Guidelines** define "our assets" to include *"any videos or screenshots taken from
  our games"*, and define commercial use as sharing *"regardless of whether you receive payment"*.
- **Precedent:** Decart/Etched's **Oasis** shipped a browser-playable Minecraft world model with open
  MIT weights in Oct 2024 — **no takedown in 21 months**, only Microsoft clarifying it is "not
  officially sanctioned". Non-enforcement is a signal, not a licence.
- **Counter-signal:** Microsoft Research's own **MineWorld** (VPT-trained) released weights then
  **pulled them in May 2025** citing "certain reasons" — still down.

**The exposure is concentrated in open-dreamer's licence, not Minecraft.** Emailing the authors for
consent / an Apache-2.0 relicense is a **one-day, zero-cost de-risk with a high chance of yes.**


### 6b. What the rest of the field actually does (verified, full-text)

A second pass over **Malmo, MineRL, MineDojo and VPT** — reading the licence files and extracting the
full paper texts — found something clarifying: **not one of them documents permission from Mojang or
Microsoft.** Machine-counted across every paper full-text: `Mojang` = 0, `EULA` = 0, `licen` = 0.

| Project | Licence | Minecraft carve-out? |
|---|---|---|
| **Malmo** (Microsoft Research) | plain **MIT**, unmodified | none — no NOTICE, no EULA reference |
| **MineRL** | ⚠️ **LICENSE = CC BY-NC-SA 4.0**, but `setup.py` says `license='MIT'` | none; GitHub reports `NOASSERTION` |
| **MineDojo** (NVIDIA) | code **MIT**; data CC BY 4.0 / CC BY-NC-SA 3.0 | none |
| **VPT** (OpenAI) | code **MIT** | explicit **non-claim** disclaimer |

Notable specifics:
- **MineRL is widely mis-cited as MIT.** Its actual LICENSE file is **non-commercial** (CC BY-NC-SA 4.0)
  in every ref checked (`dev`, `master`, v1.0.0, v0.4.4, v0.3.7). Don't build commercial work on it.
- **Malmo is *not* verifiably "Mojang-sanctioned".** That claim traces only to an **uncited sentence on
  a community fan wiki**; no Mojang/Microsoft primary source exists. (Malmo also patches decompiled
  MCP source with no legal discussion anywhere in the repo — Microsoft owns Mojang, which is presumably
  why it was never an issue for them.)
- **MineDojo is the useful precedent for us.** It deliberately **withholds the MP4s** — *"we do not
  release the actual MP4 files and transcripts due to legal concerns"* (Appendix D.1) — and ships only
  URLs + metadata. Publish the recipe and the measurements, not the copyrighted pixels.
- **VPT's contractor data has real, documented consent**: recruited on Upwork at **$20/hour**, job
  posting quoted in the paper. The 70k scraped hours have none — and are **not distributed**. We would
  only ever touch the consented half.

**What this changes:** the entire Minecraft-AI research field operates without documented Mojang
permission, across Microsoft's own lab, NVIDIA, CMU and OpenAI, for a decade, unchallenged. That makes
**internal research + a published paper/blog** a well-trodden path. It does **not** make publishing
weights safe — and MineDojo's "ship metadata, withhold pixels" norm is the model to copy.

## 7. Staged plan

| Stage | Scope | Gate | Cost | Time |
|---|---|---|---|---|
| **0. Licence ask** | Email open-dreamer authors. Parallel to all else. | A reply. Publish nothing until it lands. | $0 | 1 d |
| **1. Converter + 20 h pilot** | ~250 segments from `all_10xx`. Build `vpt_to_arrayrecord.py`. | Records round-trip; sane action histograms; `len(vr)==len(actions)` after truncation | ~$20 | 2–3 d |
| **2. Memory probe** ⭐ | 200 steps at B∈{1,2,4}, T∈{64,256} on 1×H200; try FSDP on 2×H200 | Largest config that survives 200 steps **+ a `log_every` materialisation**. If nothing ≥B2/T256 fits and FSDP won't hold → **stop and rethink** | ~$50 | 1 d |
| **3. Tokenizer, 300 h** | 8xx+9xx subset, B=32, 50k steps | PSNR ≥ ~30 (do **not** expect CoinRun's 40.41 — 84× compression vs our 24×) | ~$600 | 2 d |
| **4. Tokenize + stats** | 300 h → ~0.35 TB latents | `latent_stats.npz` std in a sane band — **compute ours, never reuse theirs** | ~$100 | 1 d |
| **5. Dynamics** | Largest config from stage 2; rescale steps to wall-clock as we did for CoinRun | flow_mse descending, boot_mse non-zero after bootstrap, **step advancing between check-ins** | $1.5–3k | 5–10 d |
| **6. Eval** | FVD (ctx 4 + horizon 240) + **the action-swap test we still owe from CoinRun** | Coherence at 240 frames; divergence under swapped actions | ~$50 | 1 d |

**Do stages 0 and 2 first.** Together they cost $50 and one day, and they retire the two risks that
actually decide this: the licence and the memory wall.

**Two things to avoid:** don't start at `n_latents=512, T=256, B=8` — it will OOM and burn a day
discovering what stage 2 tells you for $50. And don't scale the tokenizer to 1,580 h before stage 3
proves reconstruction is acceptable at 84× compression — that ratio is the one number here with no
precedent in our own results.

## Flagged as unverified
- Per-subset hours/TB extrapolated from 7–12 file samples per index + one measured bitrate.
  Segment counts, 640×360@20fps, and the off-by-one are exact/measured.
- The 26 GiB OOM in our CoinRun run was never root-caused ⇒ §3 is a scaling argument, not a model.
  **Stage 2 exists to replace it with a measurement.**
- Whether decord's frame count agrees with ffprobe on VPT mp4s — untested, and it gates the fix.
