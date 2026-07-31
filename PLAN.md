# Genie-for-Platformers — Build & Ship Plan (on Modal)

**Goal.** Reproduce DeepMind's *Genie: Generative Interactive Environments* (Bruce et al., ICML 2024, arXiv:2402.15391) pipeline — the part that learns a **controllable 2D-platformer world model from video with no action labels** — train it on **Modal**, and **ship it as a browser product people can play** (upload/choose a starting frame, press keys, the model dreams the next frames).

**Guardrails (from Rajat).** Speed is *not* the priority — correctness and doing it *properly* is. Faithful to Genie's **architecture and its systematic dataset method**, at a **tractable, shippable scale** — not the 10.7B/256-TPU frontier run.

---

## 1. What Genie actually is (confirmed from the paper)

Three components, all on a shared **Spatiotemporal (ST) Transformer** (spatial attention within a frame + *causal* temporal attention across frames → compute scales **linearly** in #frames, the key efficiency trick):

| Component | Job | Key specs (paper) | Params |
|---|---|---|---|
| **Video Tokenizer** ("ST-ViViT" VQ-VAE) | frames → discrete tokens | codebook **1024**, emb 32, **patch 4**, input **160×90**, T=16 @10fps | 200M |
| **Latent Action Model (LAM)** | infer discrete action *between* frames, **unsupervised** | VQ codebook **|A|=8**, emb 32, patch 16; encoder sees future frame, decoder reconstructs it | 300M |
| **Dynamics Model** (decoder-only **MaskGIT**) | past tokens + latent action → next-frame tokens | actions added as **additive** embeddings; **25 MaskGIT steps/frame**, temp 2 | 10.1B |

- **Train** (2-phase): (1) train tokenizer; (2) co-train LAM (reads pixels) + dynamics (reads tokens). LAM's encoder peeks at the next frame to infer the action that produced it.
- **Play** (inference): discard the LAM encoder/decoder, **keep only its 8-code action codebook**. User supplies a starting image → tokens; then each step picks an action in `{0..7}`; dynamics + tokenizer-decoder roll frames autoregressively.
- **The wall:** Genie runs at **~1 FPS**. This single fact defines the "ship it as a playable product" problem.
- **Full-scale (reference only, NOT our target):** 10.7B params, 942B tokens, 256 TPUv5p, 125k steps. Scaling study swept dynamics 41M→2.7B. **CoinRun case study** (the reproducible one): 64×64, 16 frames, 7 actions.

### The dataset method (Rajat's emphasis — the part that made it controllable)
1. Source: public internet videos.
2. **Keyword filter** on titles: 2D-platformer names + action words (`speedrun`, `playthrough`); exclude negatives (`movie`, `unboxing`).
3. Raw: **55M clips** ×16s @10fps @160×90 = **~244k hours**.
4. **Quality filter:** hand-label **10k** videos → train an **11M-param ResNet-18** classifier → apply to all.
5. Final curated set: **6.8M clips = 30k hours** (~12% of raw).
6. **Curation was a measured experiment:** FVD improved **54.8 vs 61.4**. Not a preprocessing afterthought.

---

## 2. Strategic decisions (recommended — the forks that shape everything)

### Decision A — Data route: **self-generate from CC0/permissive open platformers. Do NOT scrape YouTube.**
Genie scraped internet video. **For a shippable *commercial* product that route is disqualifying**, and there's a technical reason to avoid it too:

- **Legal (from the research):** scraped gameplay stacks four independent liabilities — YouTube ToS breach, the game publisher's copyright in the footage, the uploader's copyright, and *output-side* infringement/trademark when the model emits recognizable game visuals. 2025 case law (Bartz v. Anthropic, Kadrey v. Meta) + the U.S. Copyright Office Part 3 report point the *wrong* way for exactly our fact pattern: **commercial, expressive output that competes in the same market, obtained via unauthorized access.** Self-play from an open *engine* (e.g. Doom/GPL) does **not** free the proprietary *assets*.
- **Technical bonus:** self-generated data comes with **ground-truth action labels for free.** Genie *had* to learn latent actions blind (no labels) and could only validate controllability on CoinRun's labels. If we self-generate, we get labels on the *whole* corpus — which lets us **prove the unsupervised 8-code LAM actually recovered the real controls.** That directly serves the "do the dataset systematically" instinct.
- **The concrete stack:** **Procgen/CoinRun** (MIT, 64×64, unlimited procedural levels, native action labels) as the fast backbone; a **custom renderer using CC0 asset packs** (Kenney, Quaternius, Poly Haven; Procgen's own CC0/CC-BY manifest is the template) for higher-res fidelity + visual diversity; aggressive **texture/palette/level randomization** to close the diversity gap that Genie got "for free" from hundreds of real games. Keep a **`gym-super-mario-bros` probe research-only** (ROM is Nintendo's — never in a shipped checkpoint) purely to validate the latent-action space against real controls.

> Faithful to Genie's *method* (learn latent actions unsupervised from video + curate systematically), but on data we own. We apply Genie's exact curation discipline (quality/diversity filter, dedup, tokenize) to *our* frames.

### Decision B — Scale: **start CoinRun-scale for correctness, then climb the ladder to ~100M → 300M → (maybe) ~1B.**
No open reproduction has matched Genie's *emergent* generality; all validated results are on the small CoinRun case study. We follow the same ladder. Sub-1B is where "playable" is demonstrated (Matrix-Game 2.0 1.8B @25fps, MineWorld 300M). **11B and open-domain generality are explicitly out of scope.**

### Decision C — Base: **fork `GenieRedux` (INSAIT), borrow the tokenizer, keep `Jasmine` as the scaling reference.**
- **`GenieRedux`** (github.com/insait-institute/GenieRedux, PyTorch, MIT) — the *only* open reproduction with published, results-validated CoinRun numbers (17.91 PSNR, beats others), a proper controllability metric, the RetroAct dataset, and an exploration agent. Reproducing *its* CoinRun result is our correctness gate.
- **Tokenizer:** `Open-MAGVIT2` (TencentARC/SEED-Voken, Apache-2.0 — faithful MAGVIT-v2 LFQ, video variant, pretrained ckpts) + `lucidrains/vector-quantize-pytorch` (MIT — EMA, dead-code reset, LFQ; **the single most important knob against codebook collapse**).
- **`Jasmine`** (p-doom, JAX) — best-engineered/scalable Genie codebase (CoinRun <9h/1GPU, released checkpoints, MaskGIT+diffusion+AR baselines). Keep as the scaling reference / fallback.
- Avoid open-genie, genie-bottle, TinyWorlds (~3M, educational), 1X (robot POV, ground-truth actions) as *production* bases.

### Decision D — Treat "faithful Genie" and "playable product" as **two separate engineering tracks.**
Genie is ~1 FPS *by design* (25 MaskGIT steps × ~900 tokens/frame). Shipping requires the **post-Genie real-time playbook** (few-step distillation + causal KV-cache + low-res). These are different problems with different gates.

---

## 3. The build, staged

### Phase 0 — Foundations & the correctness gate *(low cost, high value)*
- Modal account/env: `Volume` for token shards + checkpoints ($0.09/GiB-mo, 1 TiB free); single-node **8×H100** (`gpu="H100:8"`, under the 10-GPU Starter cap); FSDP; **checkpoint-every-N-steps + idempotent resume** (Modal GPUs are preemptible by default — engineer for it from day 1).
- Fork `GenieRedux`; stand up its three components; **reproduce its CoinRun result.**
- **Gate G0:** GenieRedux CoinRun reproduced (tokenizer PSNR ≈17–18, controllability metric present). *This proves the entire stack before we invest in data or scale.*
- Budget: ~$200–500.

### Phase 1 — The Data Engine *(the crux; do this properly)*
- Build a **6-stage Modal `spawn_map` pipeline**: (1) parallel **Procgen/CoinRun rollouts** on ~1000 CPU containers (~540M frames ≈ 10k-hr-equiv in *minutes* for ~$1, all action-labeled) → (2) **custom CC0-asset platformer renderer** for higher-res + diverse art → (3) **randomization** (textures/palettes/levels/physics) → (4) Genie-style **quality/diversity filter** (small classifier, apply Genie's own trick to our data) → (5) **pHash dedup** → (6) **VQ-VAE tokenize** to `uint16` token grids, sharded to Volume (token store is tiny: ~40–185 GB for 10k hrs, ~300× compression; raw frames are transient).
- Hold out a **`gym-super-mario-bros` labeled probe** (research-only) for latent-action validation.
- **Gate G1:** a curated, tokenized, **action-labeled** platformer corpus at target scale (start ~100–1000 hrs-equiv), with measured visual diversity. DIAMOND shows even ~90 clean hours trains a playable world model, so we start small and grow.
- Budget: ~$1–50 (Route B self-gen is nearly free; cost is CPU/tokenization).

### Phase 2 — Train the three components (faithful, our scale)
1. **Tokenizer** (ST-ViViT / Open-MAGVIT2): train + validate reconstruction PSNR. **Gate G2a.**
2. **LAM (8-code VQ) + Dynamics (MaskGIT)** co-training (additive action embeddings, MaskGIT masking schedule).
3. **Validation — the real test of "did it work":**
   - **FVD** (video fidelity),
   - **Δt-PSNR controllability** (Genie's metric: model-inferred vs random actions),
   - **the label check** — do the 8 learned latent codes map to real moves (left/right/jump/…)? Measured against the ground-truth-action probe. **This is the property that makes it *playable*, and self-generated data is what lets us verify it.**
- **Scale ladder:** CoinRun-scale → 100M → 300M → (if results justify) ~1B.
- **Gate G3:** controllable model — Δt-PSNR gap + interpretable action codes.
- Budget: ~$50–420 (100M) · ~$450–750 (300M) · ~$5k–42k (1B), on Modal H100 @ $3.95/GPU-hr (practical brackets fold in the multi-stage overhead + sub-40% MFU on video attention).

### Phase 3 — Make it playable (the ~1 FPS → ~20 FPS engineering track)
The research gives a clear recipe and the ~30× gap is *algorithmic*, not hardware:
- **Few-step distillation:** 25 MaskGIT steps → **3–4** (biggest lever; Matrix-Game 2.0 = 3 steps, GameNGen 4→1).
- **Causal attention + rolling KV-cache:** stop recomputing frame history every step (Self-Forcing, Matrix-Game 2.0).
- **Lower res + higher-compression tokenizer** (256–360p, fewer tokens/frame).
- **fp8 on Hopper** (~1.5–2× on H100/H200).
- **Gate G4:** **15–30 FPS on a single H100, <50 ms/frame, coherent over 1000+ frames** (long-horizon coherence — not FPS — is where these systems actually break; measure it à la PlayGen). Anchors: Matrix-Game 2.0 1.8B @25fps, Self-Forcing @17fps, Oasis 360p@20fps/47ms.
- Budget: additional distillation training, modest (~$hundreds).

### Phase 4 — Ship the product (Modal serving + web client)
- **Serving:** `modal.Cls` with `@modal.enter` (load weights once) + `@modal.asgi_app` FastAPI **WebSocket** endpoint. One WS connection = one container = **one GPU per active player** (you cannot batch real-time players — hard 50 ms/frame wall). Autoscale: `min_containers` warm pool + `buffer_containers`; `scaledown_window` 60–300s; **GPU memory snapshots** (alpha) to cut cold-start 45–90s → ~2–5s.
- **Transport:** **WebSocket for input** (keyboard → discrete action, tiny messages) + **WebRTC for the video stream** (H.264/VP9, sub-200ms; do *not* push raw frames over WebSocket). MJPEG-over-WS is an acceptable MVP shortcut.
- **Client:** browser input loop, "upload an image / pick a scene → play it" prompt, the 8 actions mapped to arrow keys/buttons.
- **Economics:** **~$2–4 per active player-hour** on H100 (~$1.20–2.50 on L40S/A10 for a smaller checkpoint) — dominated entirely by GPU seconds. This drives free-tier limits and pricing.
- **Commercial hygiene:** output-screening (substantial-similarity/trademark) + a per-asset license ledger, since it ships.
- **Gate G5:** playable in-browser, image-prompt → play, cost-per-player within target.

### Phase 5 — Iterate
Diversity (more CC0 art, more procedural variety), long-horizon coherence, scale up, memory/persistence, richer action interface (Genie-2-style keyboard beyond 8 codes).

---

## 4. Compute & cost budget (Modal, order-of-magnitude)

| Item | Estimate |
|---|---|
| Phase 0 correctness gate (CoinRun) | ~$200–500 |
| Phase 1 data engine (self-gen, Route B) | ~$1–50 |
| Phase 2 train — 100M / 300M / 1B | ~$50–420 / ~$450–750 / ~$5k–42k |
| Phase 3 distillation for real-time | ~$hundreds |
| Phase 4 serving (ongoing) | ~$2–4 / active player-hour |
| Storage (token shards) | tens of GB → ~free under 1 TiB Volume tier |

Modal facts: H100 **$3.95/GPU-hr** (preemptible-by-default = already the "spot" price; no cheaper tier), B200 ~30% cheaper/FLOP when available, A100 worst $/FLOP for training. Single 8×H100 node handles ≤1B with FSDP; multi-node (up to 64 GPU, InfiniBand) is a gated beta we won't need.

---

## 5. Honest risks & unknowns
- **Diversity gap.** Self-generated data sacrifices the cross-game visual variety Genie got from hundreds of real titles. Whether Procgen + CC0 renderer + heavy randomization yields comparable generalization is the **main empirical risk** — mitigated by mixing several open engines/asset packs.
- **Long-horizon coherence**, not FPS, is where world models break. Must be measured (mechanics-accuracy over 1000+ frames), not eyeballed.
- **Per-player GPU economics.** ~1 GPU per active player caps concurrency and sets a hard opex floor; free play must be rate-limited.
- **Preemption.** Modal GPUs are preemptible — checkpoint frequently, resume idempotently.
- **No open repro has proven Genie's scale-up.** We are reproducing the *method* at small scale, honestly — not promising 11B emergence.
- **Legal.** Even self-generated data needs CC0/owned assets + output-screening; a lawyer check before commercial launch is prudent (esp. EU AI Act training-content summary if serving the EU).

---

## 6. Open decisions to confirm before Phase 0
1. **Scale ambition / budget ceiling** — start CoinRun-scale then decide (cheapest) vs. commit to a ~300M single-style or ~1B multi-style target.
2. **Data route** — self-generated CC0 (recommended, shippable) vs. faithful-to-Genie scraped video (research/internal only).
3. **Green-light Phase 0** (fork GenieRedux + Modal setup + reproduce the CoinRun correctness gate) as the first concrete step.

---

### Key references
- Genie paper: arXiv:2402.15391 · GenieRedux: github.com/insait-institute/GenieRedux · Jasmine: github.com/p-doom/jasmine · Open-MAGVIT2: github.com/TencentARC/SEED-Voken · vector-quantize-pytorch: github.com/lucidrains/vector-quantize-pytorch
- Real-time serving: GameNGen (arXiv:2408.14837) · Oasis (decart.ai) · DIAMOND (arXiv:2405.12399) · MineWorld (arXiv:2504.08388) · Matrix-Game 2.0 (arXiv:2508.13009) · Self-Forcing (arXiv:2506.08009)
- Data: Procgen (github.com/openai/procgen, MIT) · VPT (openai.com/index/vpt) · Modal pricing/docs (modal.com/pricing)
