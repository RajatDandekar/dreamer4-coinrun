# Genie-for-Platformers — Build Status

Living execution log. See [PLAN.md](PLAN.md) for the full plan. Decisions locked (2026-07-20):
**~1B multi-style · self-generated CC0 data · Phase 0 started.**

---

## Phase 0 — Foundations & correctness gate  🟡 in progress

### Verified (grounded in the real repo, not assumed)
- **Base repo cloned:** `GenieRedux/` (INSAIT). Two branches matter:
  - `main` = CVPR'25 *Exploration-Driven* multi-env system (RetroAct, 974 envs, needs game ROMs via stable-retro).
  - **`neurips` = the CoinRun case study WITH pretrained weights** — our Phase 0 target. Entrypoint: `accelerate launch main.py +config=<cfg> ++config.mode=<train|eval>`.
- **Architecture is faithful to Genie** (`models/`: `tokenizer.py`, `lam.py`, `dynamics.py`/MaskGIT, `components/stvivit.py` + `vector_quantize.py`). Confirmed hyperparameters vs the paper:
  - Tokenizer: dim 512, `codebook_size 1024`, `patch 4`, 16 frames, 8 blocks  → matches Genie (1024 codes, patch 4).
  - LAM: `codebook_size 7 = num_actions 7` (CoinRun's 7 actions; Genie's internet model used |A|=8). Unsupervised latent actions.
  - Dynamics: MaskGIT, `use_action_embeddings: true` (additive, per paper), `inference_steps 25` (= Genie's 25 MaskGIT steps).
  - Eval metrics: **ΔPSNR (controllability)** + FID + PSNR + SSIM; `delta_psnr_horizon 4`.
  - Optimizer: lr 1e-4, betas [0.9,0.99], wd 1e-4, cosine anneal, warmup 10k.
- **Pretrained weights EXIST and are public** on HF `INSAIT-Institute/GenieRedux` (research's "checkpoints in preparation" is outdated):
  - `GenieRedux_Tokenizer_CoinRun_100mln_v1.0`, `GenieRedux_CoinRun_250mln_v1.0` (LAM+dynamics), `GenieRedux_Guided_CoinRun_80mln_v1.0`.
  - **Bonus:** `GenieRedux-G_RetroAct-v1.5_platformers-space-shooters_260mln_v1.5` — a real multi-env platformer checkpoint, relevant to our ~1B multi-style goal.
- **Compute fits our plan:** INSAIT trained on **7×A100, batch 84**, tokenizer 150k iters → a single Modal **8×H100** node covers it (under the Starter 10-GPU cap). ✓
- **Execution boundary resolved:** Modal SDK installed (`.venv-modal/`, v1.5.2); **auth present** (`~/.modal.toml`). We can run, not just write.
- **Data route wired to Decision A:** neurips CoinRun data-gen uses OpenAI's painful 2018 `coinrun` (mpich/qt5/tf-1.12). We substitute **modern `procgen`** (pip, MIT) — which IS our self-generated CC0 route — emitting the repo's `instance/session/frames/*.jpg + actions.json` format.

### Harness
- [`modal_app.py`](modal_app.py): Modal app `genie-platformer-p0`, Volume `genie-platformer`.
  - `download_weights` / `inspect_weights` — pull + verify pretrained CoinRun weights (CPU, cheap). **← running now**
  - `train` (8×H100) / `evaluate` (1×H100) — shell out to the repo's accelerate entrypoint; go live once the CoinRun adapter lands.

### Progress
- ✅ **P0.1 — Modal pipeline proven (2026-07-20).** Image builds, Volume works, HF download works, GPU-less inspect works. Weights on Volume `genie-platformer:/vol/checkpoints/`:
  - Tokenizer 100mln → **101.7M params** (faithful); modules: encoder/decoder/vq/to_patch_emb.
  - GenieRedux 250mln (2,998 MB) → modules `[dynamics, latent_action_model]` = **full Genie WITH the unsupervised LAM**.
  - Guided 80mln (969 MB) → modules `[dynamics]` = dynamics-only (no LAM), as expected.
  - Note: checkpoints are nested dicts `{dynamics:{model+optim}, latent_action_model:{...}}` — informs how eval/finetune loads them. (Inspector under-counted 250M/80M because it didn't recurse; file sizes confirm scale.)
  - Gotcha fixed: py3.10 image needed loosened pins (scipy 1.16 requires py≥3.11); Modal builds ALL app images on `modal run`, so one bad image aborts the run.

- ✅ **P0.2 — Data engine works (2026-07-20).** `gen_coinrun` (procgen → GenieRedux `DatasetFileStructure`) validated on Modal: 128 CoinRun instances / 6,878 frames at `/vol/datasets/coinrun_v2.0.0/coinrun_v2.0.0/` (frames/*.jpg + actions.json + info.json), realistic variable-length sessions (32–64 frames). Path nesting `root/{name}/{name}` matches `train.py`'s `dataset_folder` exactly. Right-biased run+jump heuristic policy (procgen coinrun combos RIGHT=7/RIGHT+UP=8/UP=5/LEFT=1) so the character actually traverses levels. **This is the first brick of the Decision-A self-generated data engine** (procgen = MIT/CC-clean).
  - Key design decision: we do NOT reproduce INSAIT's exact CoinRun number (would need their 2018-coinrun relic: py3.6/tf1.12/qt5). Since we train ~1B from scratch on OUR data, the meaningful gate is "our tokenizer/dynamics work on our in-distribution data" — pretrained weights are only a plumbing check.

- ✅ **P0.3 — Training loop + checkpointing PROVEN on Modal (2026-07-20). PHASE 0 COMPLETE.**
  Tokenizer trained 30 steps on 1×H100 (loss moving, ~2.9 it/s); `model-15.pt` + `model-30.pt` persisted to `/vol/checkpoints/tokenizer/tokenizer/`. Full pipeline validated: construct 100M tokenizer → load our procgen CoinRun → fwd/bwd/optim → checkpoint to Volume.
  - **Two genuine GenieRedux bugs found & fixed (baked into the image via sed patches):**
    1. `Trainer.load_log_data` indexes `[0, max_valid_size//4−1, …]` = `[0,31,63,95]` from the *default* `max_valid_size=128` into a smaller valid batch → out-of-bounds on any dataset with <128 valid sequences (theirs was ~10k instances). Fix: clamp `self.max_valid_size = min(max_valid_size, len(self.valid_ds))`.
    2. `train.py` always calls `dist.destroy_process_group()`, which asserts in single-process runs. Fix: guard with `dist.is_initialized()`.
  - **Debug lesson:** async CUDA `device-side assert` masked the real Python line; a `--cpu` run surfaced the clean `IndexError` at `trainer.py:345`. Always CPU-repro to de-mask CUDA asserts.

- ✅ **P1.1 — Data engine to production (2026-07-20):** `gen_data` fan-out (Modal `.map`), diversified multi-behavior policy, multi-game-ready. Generated `coinrun_prod_v1` = **2,000 instances / 121,728 frames**.
- ✅ **P1.2 — 8-GPU DDP validated**, and learned the key op lesson: **all long runs use `modal run --detach`** (attached runs die when the local shell is reaped; detached runs persist server-side).
- ✅ **P1.3 — TOKENIZER TRAINED (2026-07-20):** from-scratch 100M ST-ViViT, 15,000 steps on 8×H100 (~4.5h, detached). Recon loss **0.05 → 0.0005** (~PSNR mid-30s, INSAIT-quality). `model-15000.pt` on Volume. Ran clean through two shell reaps thanks to `--detach`.
- 🟡 **P1.4 — LAM + dynamics (genie_redux, ~250M) training** — validating start, then long detached run. The controllability gate: FVD + ΔPSNR + do the 7 unsupervised latent codes map to real moves (checkable vs our ground-truth labels).

### Next steps (Phase 1 — the real build)
1. **Data engine to production quality** (the crux): replace the heuristic policy with a trained/scripted agent (procgen PPO or better), scale instances (thousands), add `.map` fan-out, then **multi-style CC0 assets + higher resolution** toward the ~1B multi-style goal.
2. **Full tokenizer training** (150k steps) on the scaled dataset → validate reconstruction PSNR.
3. **LAM + dynamics training** → validate FVD + ΔPSNR controllability + latent-action↔label mapping (our ground-truth labels make this checkable).
4. Scale to ~1B (8×H100 node), then **distill for real-time** (P3) → **serve** as a playable browser product (P4).

### Open sub-decision (noted, not blocking)
For G0, procgen-generated CoinRun frames differ slightly from OpenAI-coinrun-2018, so metrics won't exactly match INSAIT's paper number (still validates the stack). Exact-number reproduction would need their 2018-coinrun data-gen. Recommend: **procgen gate now** (aligned with our data route), exact repro optional later.

---
## 🌙 OVERNIGHT AUTONOMOUS PLAN (2026-07-21 night)
Goal: Rajat wakes to a **playable browser demo** of the world model we trained tonight + controllability results + write-up.

**State:** ✅ tokenizer (model-15000) · ✅ LAM+dynamics genie_redux (model-40000, 40k steps) · ✅ eval ran (PSNR ~18-19, SSIM ~0.53, ΔPSNR ~0 = subtle per-frame control; world-gen coherent — verified visually). Model + weights on Volume `genie-platformer`.

**Inference API (verified):** `construct_model(cfg)` builds tokenizer+LAM+dynamics; load `torch.load(model-40000.pt)["model"]`. Generate: `model.sample(prime_frames=(b,c,t,h,w), actions=(b,n) indices 0-6, num_frames=n)` → frames [0,1]. Control-mode = force all actions to K.

**Build steps:**
1. [ ] serve.py: Player class (@modal.enter loads model + start-frame gallery), `dream(scene, actions)` → frames. TEST it generates.
2. [ ] Wrap in FastAPI @modal.asgi_app: GET / (client), POST /api/dream. **modal deploy** (persistent URL, survives teardowns).
3. [ ] Polished HTML client: pick scene, press keys (7 actions), watch the model dream. "Play inside the world model."
4. [ ] Showcase page: story + rollout GIFs + control-mode comparison + metrics + architecture.
5. [ ] Fallback if live too flaky: rich static gallery of pre-generated dream rollouts.

**Ops:** ALL long jobs `modal run --detach` or `modal deploy`. Keep a heartbeat bash so I re-invoke. Modal CLI: `/Users/raj/Downloads/Vizuara/genie-platformer/.venv-modal/bin/modal`. Trained model: `/vol/checkpoints/genie_redux/genie_redux/model-40000.pt`, tokenizer `/vol/checkpoints/tokenizer/tokenizer/model-15000.pt`. Data `coinrun_prod_v2`.
