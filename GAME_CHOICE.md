# Which game next? — the post-Minecraft decision

Minecraft is dropped (Rajat's call, 2026-07-31). This document is the ranked answer to
"is there any other game which is very impactful which we can model", compiled from a
large research fan-out (many agents were cut short by a usage limit; what's below is
grounded in the runs that completed, and the load-bearing licence facts were verified
against primary sources before the cutoff).

---

## The recommendation: DOOM — via VizDoom + Freedoom

**Build "the first fully-open DOOM-class world model": every frame self-generated,
every asset redistributable, weights publishable.**

### Why it wins on impact
- DOOM is *the* world-model headline game. Google's **GameNGen** ("Diffusion Models Are
  Real-Time Game Engines", Aug 2024) made global news as "AI runs DOOM" — and **never
  released weights or code**. DIAMOND released CS:GO/Atari models; Oasis covered
  Minecraft. **Nobody has shipped an open, licence-clean DOOM-class world model.**
  That is the same "reproduce the landmark, openly" story that worked for our
  Dreaming-to-Dodge project — one tier up.
- First-person 3D with perspective, occlusion, combat and level topology — a visible
  capability leap over CoinRun's 2D platformer, without Minecraft's memory wall.

### Why it's licence-clean (the part that killed Minecraft)
| Layer | Licence | Status |
|---|---|---|
| VizDoom (the env/API) | **MIT** | verified from repo |
| Engine underneath (ZDoom lineage) | GPL — engine output is **not** covered (FSF FAQ confirms game output isn't reached by engine GPL) | verified |
| **Freedoom** assets (IWAD replacing DOOM's copyrighted art) | **3-clause BSD** — free redistribution, commercial OK, no ShareAlike | verified from COPYING |
| Original DOOM WAD | copyrighted (what GameNGen used) | **we don't touch it** |

So unlike GameNGen (trained on Mojang-grade copyrighted pixels) we can publish the
dataset, the frames, the site, and — once the open-dreamer licence question is resolved
— the weights.

### Why it's technically a fit for our exact stack
- **Self-generated, action-labelled data** — VizDoom is a Gym-style API: we script or
  train policies and record `(frame, action, reward)` exactly like our procgen data
  engine. Same pickle→ArrayRecord writer, same reward-biased slicing. No scraped video,
  no mp4/jsonl alignment bugs.
- **Headless & cheap** — VizDoom renders off-screen natively (no X server, no GPU
  needed) at thousands of FPS per CPU core. Data generation is Modal CPU containers
  again, ~procgen-cheap.
- **Fits one GPU** — dynamics cost scales with `n_latents`/tokens, **not** raw pixel
  count. At 320×240 (GameNGen's own resolution) with a sensibly sized latent grid, the
  activation-memory analysis says a **single H200** holds it — no FSDP, no NCCL
  deadlock exposure, no 23× Minecraft blow-up.
- **20-minute pilot path**: swap `procgen.make → vizdoom`, regenerate shards, rerun the
  tokenizer. Every downstream stage is already proven.

### Honest risks
1. **open-dreamer licence** (unchanged, gates weight release for *any* game) — email
   the authors; publish measurements regardless, as we did for CoinRun.
2. Freedoom's aesthetic is "DOOM-like", not DOOM — the announcement must say
   "DOOM-class / Freedoom", not claim DOOM pixels. That honesty is also the point.
3. Scenario diversity needs design: a mix of Freedoom levels + VizDoom's standard
   scenarios, with a driving policy good enough to explore (scripted + trained mix,
   as GameNGen did with its data-collection agent).

---

## Runners-up (ranked)

**2. 0 A.D. (open-source RTS)** — the dark-horse. Deep-dive verified: a first-class
`--rl-interface` where the sim advances *only* on your JSON command while the engine
renders — perfect frame/action lockstep; full symbolic state per frame; visual-replay
mode decouples cheap sim from expensive render. Gorgeous, distinctive isometric look;
no world model has ever been trained on an RTS at this fidelity. Blockers: art is
**CC-BY-SA** (ShareAlike on a published frame dataset; weights debatable), engine is
GPL (fine), and **CPU-rendering throughput is unmeasured** — needs a 1-hour Xvfb +
llvmpipe benchmark before committing. Choose this if we want "beautiful and unprecedented"
over "iconic and safe".

**3. SuperTux** — verified GPLv3 + GPLv2+/CC-BY-SA assets, in Debian main (strongest
cleanliness signal), tiny frames, trivially cheap. But it's another 2D platformer —
zero narrative advance over CoinRun. Only worth it as a style-transfer add-on dataset.

**4. Racing / driving** — the research thread died at the usage limit before licences
were verified (SuperTuxKart, TORCS, Trackmania). Trackmania is proprietary; TORCS
tooling is ancient. Real-world driving (nuScenes etc.) is NC-licensed. Park it.

**5. Embodied sims (Genesis / iGibson / OmniGibson / Habitat)** — verified in detail
and rejected: Genesis renders ~10× realtime with cameras on (the 43M FPS claim is
physics-only) and ships no scenes; iGibson/OmniGibson scene assets (Matterport3D,
CubiCasa5K, BEHAVIOR bundle) explicitly restrict derived model weights. Not games,
and licence-hazardous for exactly the thing we want to publish.

---

## Suggested next step (when approved)

Stage 0 (~$5, one day): VizDoom + Freedoom pilot — 50 episodes recorded on Modal CPU,
records round-tripped through our ArrayRecord reader, tokenizer smoke-trained 500 steps
at 320×240. That single run retires the two open questions (record schema at the new
resolution; tokens/frame at patch 16) before any real money is spent.
