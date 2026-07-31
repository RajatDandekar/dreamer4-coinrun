# Open Dreamer (Dreamer 4) on CoinRun — findings & recipe

Switched from GenieRedux (Genie-1, 2024) to **Open Dreamer** (`next-state/open-dreamer`,
a JAX/Flax impl of **Dreamer 4**, Hafner/Yan/Lillicrap, arXiv:2509.24527) because it fixes
the two things that made our demo flaky:

| Our symptom | Root cause (GenieRedux build) | Dreamer 4 fix |
|---|---|---|
| control barely responds | **unsupervised** latent actions (8-code VQ); measured ΔPSNR ≈ 0 | **action-conditioned** dynamics (ground-truth actions) |
| ~1 fps, turn-based | 16–25 MaskGIT steps/frame; real-time needs a separate distillation project | **shortcut forcing** — flow conditioned on noise level *and* step size ⇒ few-step generation learned in one phase |
| dreams drift | random-biased heuristic policy, 64-frame clips | full episodes + **reward-biased sampling** (`p_include_reward: 0.5`) |
| weak model | 122k frames, 40k dynamics steps | **9.6M frames**, 200k dynamics steps |

## ⚠️ Their CoinRun path is INCOMPLETE — 4 real breakages

The write-up says CoinRun is their single-GPU starting point, but the shipped code can't
reproduce it end to end. All four found by reading the source:

1. **`dreamer/data/generate_coinrun_dataset.py` crashes.** Calls
   `ShardWriter(..., serialization_format="pickle")`; `ShardWriter.__init__` accepts no
   such kwarg.
2. **Writer/reader format mismatch.** `ShardWriter` emits **msgpack**
   (`serialize_msgpack_record`), but the CoinRun reader
   (`transforms.py: ProcessEpisodeAndSlice`) does `pickle.loads(element)`. The pickle
   write path was lost in a refactor.
3. **No CoinRun tokenization.** `scripts/tokenize_minecraft_dataset.py` hardcodes
   `ProcessMinecraftEpisodeAndSlice(full_episode=True)` — Minecraft MP4 only.
4. **`scripts/train_dynamics.py` asserts the Minecraft action space** (lines 194–195):
   `assert cfg.dataset.num_binary_actions == NUM_BINARY_ACTIONS` /
   `== NUM_CAMERA_CLASSES`. CoinRun is (0 binary, 16 categorical) ⇒ instant failure.
   **The model is generic** (`models.py` builds embeddings from config and skips binary
   when 0) — the asserts are a stale guard, not a real constraint.

### Our fixes
- Write records **pickle**-serialized in the schema the reader parses:
  `{"raw_video": uint8 (T,64,64,3) bytes, "sequence_length": T, "actions": (T,), "rewards": (T,)}`
  into `shard-%05d.array_record`. **Proven** — their trainer consumes it.
- Skip #3 entirely: `train_dynamics.py` supports raw video
  (`use_latent_data = data_type == "latent"` → False → `tokenizer.encode(...)` on the fly).
- Patch #4 with sed at image build (2 asserts → `pass`).
- Compute `latent_mean`/`latent_std` ourselves (coinrun.yaml lacks them; dynamics
  normalizes by them and they normally come from the Minecraft-only tokenize script).

## Environment gotchas (Modal)
- **procgen needs py≤3.10; open-dreamer requires py==3.11.\*** ⇒ two images (datagen / train).
- procgen's `libenv.so` needs **glib**: `apt install libglib2.0-0` (+`libgl1`).
- **`decord==0.6.0` 500s on Modal's PyPI mirror.** Only used by the Minecraft path and
  guarded by `_require_decord()` ⇒ `sed -i '/decord/d' pyproject.toml && uv lock`.
- **Do NOT use an `nvidia/cuda` devel base with `jax[cuda12]`** — the plugin fails to
  resolve cuSPARSE and JAX *silently falls back to CPU*. Use `debian_slim` + pip CUDA
  wheels. Always assert on `jax.devices()`.
- `array-record>=0.8.3` needs py3.11 (py3.10 caps at 0.8.1) — cross-version ArrayRecord
  read-back verified OK.
- Modal lowercases CLI flags: a param named `B` becomes `--b` and breaks. Use `batch`.
- grain multiprocessing workers fail in standalone scripts
  (`'NoneType' has no attribute 'Empty'`) ⇒ set `num_workers=0` for small jobs.

## The recipe (their configs)
- **Tokenizer:** 10k steps, lr 3e-3, EMA 0.999, MAE masking p_max 0.9, encoder depth 12
  (d_model 1536) / decoder depth 8 (d_model 1024), `n_latents` 512, `d_bottleneck` 16,
  loss mse + LPIPS.
- **Dynamics:** 200k steps, **muon** optimizer, lr 3e-4 wsd (warmup 5%, decay 10%),
  depth 30 (d_model 1920), `packing_factor` 2, `n_register` 32, `k_max` 256,
  OT coupling (Sinkhorn/barycentric), `loss_weighting: v_space`, EMA 0.999,
  **`bootstrap_start` 100k, `bootstrap_fraction` 0.25**.
- **Data:** 10k train / 500 val / 500 test episodes, `chunk_size` 160,
  `min_episode_length` 1000, reward-biased slicing.

### Our CoinRun deviations (justified)
- **`n_latents` 512 → 32.** Their 512 targets Minecraft 360×640 (~920 patches/frame,
  ≈84× compression). CoinRun is 64×64/patch 8 = **64 patches/frame**, so 512 would be 8×
  *expansion* and would inflate dynamics cost (tokens ∝ n_latents/packing_factor).
  32×16 = 512 values vs 12,288 raw ⇒ **24× compression**; 32 divisible by packing_factor 2.
- Batch 32 for the tokenizer (their B=8 is Minecraft-sized; B=128 OOMs an H100 here).

## Results so far
- **Dataset:** 400/20/20 shards = 60k/3k/3k records = **9.6M / 480k / 480k frames**
  (10k/500/500 episodes; ~32% yield at `min_episode_length=1000`).
- **Tokenizer (10k steps, 1×H100, 1h35m, ~$6):**
  **PSNR 40.41** · MSE 0.0021 · LPIPS 0.0015 (from PSNR 15.9 @200 steps).
  > Beats **GenieRedux 38.25** and the **Genie paper's 35.7**.
  Reconstructions show crisp platforms, sprites, and distinct level art styles.

## Licensing
`next-state/open-dreamer` has **no LICENSE file** (GitHub API: `license: None` ⇒ all rights
reserved). Fine for internal evaluation; **ask upstream for Apache-2.0/MIT before shipping.**

## Commands
```bash
V=/Users/raj/Downloads/Vizuara/genie-platformer/.venv-modal/bin/modal
$V run --detach modal_dreamer.py::gen_data          # dataset (done)
$V run --detach modal_dreamer.py::train_tokenizer --steps 10000 --batch 32 --run tok-coinrun
$V run --detach modal_dreamer.py::latent_stats --run tok-coinrun
$V run --detach modal_dreamer.py::train_dynamics --steps 200000 --batch 16 --run dyn-coinrun
```
Volume `open-dreamer`: `/datasets/coinrun_episodes/{train,val,test}`, `/logs/<run>/{checkpoints,viz}`.

---
## 🌙 UNATTENDED RUN (launched 2026-07-25, Rajat away)

**Modal app `ap-eWlh3sWqqsjuW6wK9GxaYo` — self-chaining `pipeline()`, runs server-side.**

Stages: dynamics (200k steps, 1.57B params, bootstrap_start=100k) → FVD eval + rollouts.
Writes `/PIPELINE_STATUS.json` to the Volume after every stage.

### Why it survives a closed laptop
Modal `--detach` runs on Modal's cloud; the laptop is only a remote control. Verified
repeatedly (earlier 40k dynamics + 10k tokenizer runs survived several session teardowns).

### Resumability (important)
- `train_dynamics.py:255` does `start_step, bundle, rng = bundle.restore(checkpoint_manager, rng)`
  ⇒ **auto-resumes from the latest checkpoint**.
- Checkpoints every 5,000 steps ⇒ relaunching loses ≤5k steps.
- `pipeline()` skips dynamics entirely if the final checkpoint exists.
- Relaunch with: `modal run --detach modal_dreamer.py::pipeline`

### Gotcha fixed at launch
Modal caps `timeout` at **86400s (24h)**; `48*3600` raises
`InvalidError: Timeout must be between 10s and 86400s`. If 200k steps exceeds 24h the job is
killed — just relaunch, it resumes.

### Throughput estimate
1.57B params but only 16 spatial tokens/frame (vs Minecraft's 256) ⇒ ~6·N·tokens ≈ 7.7e13
FLOPs/step ⇒ roughly 5–7 it/s on 1×H100 ⇒ 200k steps ≈ 8–24h.

### Monitoring
Durable cron `4f7681f9` (:13 and :43 hourly) wakes Claude to check status, self-repair, and
relaunch on failure. **Limitation:** it only fires while Claude Code is actually running —
it cannot fire while the Mac is asleep. The Modal job continues regardless.

---
## Unattended run — course correction (2026-07-26)

### Failure 1: OOM on H100
`RESOURCE_EXHAUSTED: Out of memory while trying to allocate 26.00GiB [jit_train_step]`
after exactly 100 steps (= `log_every`, when metrics are materialized). 1.57B params in fp32
(params + muon momentum + EMA ≈ 19GB) + activations at batch 16 × 64 frames exceeds 80GB.
**Fix:** H100 → **H200 (141GB)** and batch 16 → **8** (which is also their `dynamics.yaml`
default — 16 was our deviation). Verified: passed step 194, no OOM.

### Failure 2 (the real one): throughput
| config | it/s | batch | seq/s |
|---|---|---|---|
| 1×H200 | 1.05 | 8 | 8.4 |
| 8×H200 | 1.15 | 32 | 36.8 |

**Step rate is ~1.1 it/s regardless of GPU count** — extra GPUs buy batch, not steps. So:
- 200k steps ≈ 48–53h, but **Modal caps a job at 24h** (~90k steps)
- `bootstrap_start=100k` ⇒ a 24h window **never reaches the shortcut/bootstrap phase**, which
  is the whole reason for Dreamer 4 (few-step ⇒ real-time generation)

**Fix — rescale the recipe to the constraint instead of the step count:**
`8×H200, batch 32, 60k steps, bootstrap_start=30k` ⇒ ~17h (fits one window) and
**60k × 32 = 1.92M sequences > their 200k × 8 = 1.6M** — *more* data than the original recipe,
with a full bootstrap phase.

Multi-GPU test also confirmed the bootstrap phase trains: `boot_mse=0.4238` (non-zero),
`flow_mse` 0.969 → 0.527.

### Ops gotchas
- `modal app stop <id>` needs `--yes` (no interactive TTY).
- **Never run two jobs against the same `run_name`** — both write the same checkpoint dir and
  would corrupt state. Stopped the superseded single-GPU run before launching `dyn-coinrun-mg`.
- Modal `timeout` max = 86400s.

### Live run
`ap-LcbpHthMb1ybD3NlcOBFDM` — 8×H200, run `dyn-coinrun-mg`, status `dynamics -> running`.
Monitoring cron `aa065884` (:17/:47 hourly).

### Failure 3: 8×H200 NCCL deadlock (2026-07-26)
The multi-GPU run froze at step 4187 after ~1h:
```
E rendezvous.cc:116] This thread has been waiting for [rank=7] Acquire clique: devices=8:[...
```
A JAX/NCCL collective rendezvous deadlock. **Modal still reported `ephemeral / Tasks 1` and the
pipeline status still said `running`** — a silent stall burning ~$36/hr. Step counter AND
elapsed time were both frozen, which is the only reliable tell.

**Verdict: multi-GPU is not reliable here. Stay on a SINGLE H200.**

**Final config:** 1×H200, batch 8, **80k steps, bootstrap_start=40k** ≈ 21h at the measured
1.05 it/s — fits Modal's 24h cap. 80k × 8 = 640k sequences.

**Monitoring hardened:** cron `57fac581` now compares step count against `/tmp/last_step.txt`
between check-ins and greps for `rendezvous`/`Acquire clique`, because "app is running" does
NOT mean "training is advancing".

Live: `ap-QLmJonKeDGJN59t7TcKoBc`.

---
## ✅ RESULTS — dynamics trained + evaluated (2026-07-28)

**Dynamics: 80,000/80,000 steps complete** (1×H200, batch 8, bootstrap_start 40k, ckpt 79999).
`flow_mse` 0.53 → **0.014**; `boot_mse` active from 40k → **0.004**.

### FVD on the held-out TEST split (576 clips × 16 frames, 144-frame rollouts from 4 context frames)
| metric | value | meaning |
|---|---|---|
| FVD(original, **pred**) | **32.19** | **end-to-end** (what you actually see) |
| FVD(original, gt_decoded) | 16.59 | tokenizer ceiling (recon floor) |
| FVD(gt_decoded, pred) | 23.29 | dynamics only |

Reference: the **Genie paper reports FVD 54.8** on its curated platformer set (580M dynamics).
Ours is 32.19 end-to-end — *not* a like-for-like comparison (different data/resolution/protocol),
but it is comfortably in the healthy range, and the decomposition is clean: the tokenizer floor
(16.59) plus a modest dynamics contribution (23.29).

### Visual check (the part that matters)
Filmstrips of `pred_000000.mp4` vs `gt_decoded_000000.mp4`, including frames **110–148**
(maximum accumulated drift):
- Crisp sky / stone platforms / crates / character throughout — **no mush, no collapse**
- Geometry stays valid CoinRun even at the end of a 144-frame autoregressive rollout
- Night-and-day vs the old GenieRedux build, whose dreams were brown blobs

### ⚠️ Still unmeasured: controllability
FVD measures visual + temporal quality, **not** whether actions steer the world. This eval ran
in replication mode (ground-truth actions). The old build's failure was ΔPSNR ≈ 0 — good-looking
frames that ignored input. **Next step: an action-swap test** (same context, true vs different
action; measure divergence) before claiming control is fixed.

### Failure 5 (their repo)
`eval_fvd.py` writes MP4 via `iio.imwrite(plugin="pyav")` but pyproject only declares
`imageio[ffmpeg]` ⇒ `ImportError` at save (generation itself was fine). Fix: `uv pip install av`.
Also `eval_fvd.yaml` has **no `tokenizer_ckpt`** key (checkpoint is self-contained) — passing it
is a Hydra strict-struct error. And Modal's click wrapper reserves `ctx` as a param name.
