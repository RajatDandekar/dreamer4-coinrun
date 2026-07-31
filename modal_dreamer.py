"""
Open Dreamer (Dreamer 4) on Modal — CoinRun from scratch.

Follows next-state/open-dreamer's documented recipe:
  1. generate CoinRun episodes  -> ArrayRecord shards (pickle)
  2. train_tokenizer.py         -> causal video tokenizer (MAE, 10k steps)
  3. tokenize_*_dataset.py      -> latent shards + latent_stats.npz
  4. train_dynamics.py          -> action-conditioned dynamics w/ shortcut forcing (200k steps)
  5. eval_fvd.py                -> FVD + rollouts

Why this replaces the GenieRedux build:
  - dynamics is ACTION-CONDITIONED (ground-truth actions) -> crisp control
    (our Genie-1 model used unsupervised latent actions; measured dPSNR ~= 0)
  - shortcut forcing (flow conditioned on noise level AND step size) -> few-step
    generation learned in ONE phase -> real-time, no separate distillation project

NOTE ON PYTHON VERSIONS: open-dreamer requires python==3.11.*, but `procgen`
(needed only by their data generator) ships wheels up to ~3.10. So we split:
  - DATAGEN image: py3.10 + procgen + array-record
  - TRAIN image:   py3.11 + jax[cuda12] + uv sync (faithful to their lockfile)

NOTE ON LICENSE: next-state/open-dreamer currently has NO LICENSE file
(GitHub API reports license: None => all rights reserved). Used here for
internal research evaluation only. Ask upstream to add Apache-2.0/MIT before
anything ships publicly.
"""
import modal

app = modal.App("open-dreamer")
vol = modal.Volume.from_name("open-dreamer", create_if_missing=True)
VOL = "/vol"
REPO = "https://github.com/next-state/open-dreamer.git"
REPO_DIR = "/root/open-dreamer"

# ---------------------------------------------------------------------------
# DATAGEN image: py3.10 (procgen wheels) + their shard writer deps
# ---------------------------------------------------------------------------
datagen = (
    modal.Image.debian_slim(python_version="3.10")
    # procgen's prebuilt libenv.so needs glib (libgthread-2.0.so.0) + GL.
    .apt_install("git", "build-essential", "libglib2.0-0", "libgl1", "ffmpeg")
    .pip_install(
        # array-record unpinned: py3.10 caps at 0.8.1 (>=0.8.3 needs py3.11).
        # The ArrayRecord container format is stable, but we verify explicitly
        # with `verify_shards` (read back in the py3.11 training image).
        "procgen==0.10.7", "gym3", "numpy<2", "tyro", "array-record",
        "absl-py", "etils[epath]",
    )
    .run_commands(f"git clone --depth 1 {REPO} {REPO_DIR}")
)

# ---------------------------------------------------------------------------
# TRAIN image: py3.11 + JAX CUDA12, installed via uv from their lockfile
# ---------------------------------------------------------------------------
train_img = (
    # NOTE: do NOT use an nvidia/cuda devel base here. jax[cuda12] ships its own
    # CUDA via pip wheels; mixing the two makes the plugin fail to resolve
    # cuSPARSE ("Unable to load cuSPARSE") and JAX silently falls back to CPU.
    # A slim base + pip CUDA wheels is the reliable combination.
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "curl", "build-essential")
    .pip_install("uv")
    .run_commands(
        f"git clone --depth 1 {REPO} {REPO_DIR}",
        # decord (MP4 decode) 0.6.0 persistently 500s on Modal's PyPI mirror.
        # It is ONLY used by the Minecraft VPT path: transforms.py imports it in
        # a try/except and guards use behind _require_decord(), so the CoinRun
        # path (raw uint8 bytes, no MP4) works fine without it. Drop + relock so
        # everything else stays pinned to their versions.
        f"cd {REPO_DIR} && sed -i '/decord/d' pyproject.toml && uv lock && uv sync --no-dev",
        # train_dynamics.py hardcodes the Minecraft VPT action space:
        #   assert cfg.dataset.num_binary_actions == NUM_BINARY_ACTIONS
        #   assert cfg.dataset.categorical_action_dim == NUM_CAMERA_CLASSES
        # CoinRun is (0 binary, 16 categorical) so these fail. The MODEL is
        # generic (models.py builds embeddings from config, skipping binary when
        # 0), so these asserts are a stale guard, not a real constraint.
        f"cd {REPO_DIR} && sed -i 's/^        assert cfg.dataset.num_binary_actions == NUM_BINARY_ACTIONS$/        pass  # patched: action space comes from dataset cfg/;"
        f"s/^        assert cfg.dataset.categorical_action_dim == NUM_CAMERA_CLASSES$/        pass  # patched: action space comes from dataset cfg/' scripts/train_dynamics.py",
        f"cd {REPO_DIR} && grep -n 'patched: action space' scripts/train_dynamics.py",
    )
)


@app.function(image=train_img, gpu="H100", volumes={VOL: vol}, timeout=1800)
def smoke():
    """Verify JAX sees the GPU and open-dreamer imports inside their uv venv."""
    import subprocess
    code = (
        "import jax, jaxlib; "
        "print('jax', jax.__version__, 'jaxlib', jaxlib.__version__); "
        "print('devices:', jax.devices()); "
        "import flax, grain, optax, ott; print('flax/grain/optax/ott ok'); "
        "import dreamer; from dreamer.data import data as dd; print('dreamer imports ok'); "
        "import jax.numpy as jnp; x=jnp.ones((1024,1024)); "
        "print('matmul ok:', float((x@x).sum()))"
    )
    r = subprocess.run(f"cd {REPO_DIR} && uv run python -c \"{code}\"",
                       shell=True, capture_output=True, text=True, executable="/bin/bash")
    print("STDOUT:\n", r.stdout[-4000:])
    print("STDERR:\n", r.stderr[-4000:])
    return r.returncode


@app.function(image=datagen, volumes={VOL: vol}, timeout=1800)
def datagen_smoke():
    """Check procgen works AND measure episode yield rate.

    Their generator sets min_episode_length=1000 and discards shorter episodes.
    With RANDOM actions CoinRun episodes usually end early (death/win), so this
    could resample forever. Measure before committing to a 10k-episode run.
    """
    import numpy as np
    from procgen import ProcgenGym3Env
    from gym3 import types_np

    lengths = []
    for trial in range(20):
        env = ProcgenGym3Env(num=1, env_name="coinrun", start_level=int(np.random.randint(0, 10000)))
        first_obs, n = True, 0
        for t in range(1000):
            rew, obs, first = env.observe()
            env.act(types_np.sample(env.ac_space, bshape=(env.num,)))
            n = t + 1
            if first and not first_obs:
                break
            first_obs = False
        lengths.append(n)
    lengths = np.array(lengths)
    print(f"episode lengths over {len(lengths)} trials:")
    print(f"  mean={lengths.mean():.1f} median={np.median(lengths):.0f} "
          f"min={lengths.min()} max={lengths.max()}")
    print(f"  fraction reaching 1000 (their min_episode_length): "
          f"{(lengths >= 1000).mean():.1%}")
    print(f"  action space n = {ProcgenGym3Env(num=1, env_name='coinrun', start_level=0).ac_space.eltype.n}")
    return {"mean": float(lengths.mean()), "frac_full": float((lengths >= 1000).mean())}


CHUNK = 160          # frames per record (their chunk_size default)
H = W = 64
C = 3


@app.function(image=datagen, volumes={VOL: vol}, timeout=6 * 3600, cpu=2.0)
def _gen_shard(spec):
    """Roll out CoinRun episodes and write ONE ArrayRecord shard of pickle records.

    Record schema is dictated by their reader (transforms.ProcessEpisodeAndSlice):
        {"raw_video": uint8 bytes (T,64,64,3), "sequence_length": T,
         "actions": (T,) int, "rewards": (T,) float}
    NOTE: their shipped generate_coinrun_dataset.py passes
    `serialization_format="pickle"` to ShardWriter, but ShardWriter takes no such
    kwarg and writes msgpack -> it would crash, and the reader wants pickle.
    So we write pickle records directly.
    """
    import os, pickle
    import numpy as np
    from procgen import ProcgenGym3Env
    from gym3 import types_np
    from array_record.python.array_record_module import ArrayRecordWriter

    split, shard_idx, n_eps, min_len, seed0 = (
        spec["split"], spec["shard_idx"], spec["n_eps"], spec["min_len"], spec["seed0"])
    out_dir = f"{VOL}/datasets/coinrun_episodes/{split}"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/shard-{shard_idx:05d}.array_record"
    writer = ArrayRecordWriter(path, "group_size:1")

    rng = np.random.RandomState(seed0)
    n_records = 0
    n_frames = 0
    ep_lens = []
    tries = 0
    got = 0
    while got < n_eps and tries < n_eps * 12:
        tries += 1
        lvl = int(rng.randint(0, 100000))
        env = ProcgenGym3Env(num=1, env_name="coinrun", start_level=lvl)
        obs_l, act_l, rew_l = [], [], []
        first_obs = True
        for _ in range(1000):
            rew, obs, first = env.observe()
            if first and not first_obs:
                break
            a = types_np.sample(env.ac_space, bshape=(env.num,))
            env.act(a)
            obs_l.append(obs["rgb"][0])
            act_l.append(int(a[0]))
            rew_l.append(float(rew[0]))
            first_obs = False
        ep_lens.append(len(obs_l))
        if len(obs_l) < min_len:
            continue
        got += 1
        vid = np.asarray(obs_l, dtype=np.uint8)          # (T,64,64,3)
        acts = np.asarray(act_l, dtype=np.int32)
        rews = np.asarray(rew_l, dtype=np.float32)
        # split episode into fixed CHUNK-frame records (drop remainder < CHUNK)
        for s in range(0, len(vid) - CHUNK + 1, CHUNK):
            rec = {
                "raw_video": vid[s:s + CHUNK].tobytes(),
                "sequence_length": CHUNK,
                "actions": acts[s:s + CHUNK],
                "rewards": rews[s:s + CHUNK],
            }
            writer.write(pickle.dumps(rec))
            n_records += 1
            n_frames += CHUNK
    writer.close()
    vol.commit()
    return {"shard": shard_idx, "split": split, "records": n_records, "frames": n_frames,
            "episodes_kept": got, "episodes_tried": tries,
            "mean_ep_len": float(np.mean(ep_lens)) if ep_lens else 0.0}


@app.function(image=train_img, volumes={VOL: vol}, timeout=1800)
def verify_shards(split: str = "train"):
    """Read shards back in the py3.11 training image using THEIR loader path.
    Proves cross-version ArrayRecord compat + correct record schema."""
    import subprocess
    code = f'''
import glob, pickle, numpy as np
from array_record.python.array_record_module import ArrayRecordReader
paths = sorted(glob.glob("{VOL}/datasets/coinrun_episodes/{split}/shard-*.array_record"))
print("shards:", len(paths))
r = ArrayRecordReader(paths[0]); n = r.num_records(); print("records in shard0:", n)
raw = r.read([0])[0]
d = pickle.loads(raw)
print("keys:", sorted(d.keys()))
T = d["sequence_length"]
v = np.frombuffer(d["raw_video"], dtype=np.uint8).reshape(T, {H}, {W}, {C})
a = np.asarray(d["actions"]); rw = np.asarray(d["rewards"])
print("video", v.shape, v.dtype, "range", int(v.min()), int(v.max()))
print("actions", a.shape, "min", int(a.min()), "max", int(a.max()))
print("rewards", rw.shape, "sum", float(rw.sum()))
total = 0
for p in paths:
    rr = ArrayRecordReader(p); total += rr.num_records()
print("TOTAL RECORDS:", total, "TOTAL FRAMES:", total*T)
print("VERIFY OK")
'''
    r = subprocess.run(f"cd {REPO_DIR} && uv run python -c '{code}'",
                       shell=True, capture_output=True, text=True, executable="/bin/bash")
    print(r.stdout[-3000:]); print("STDERR:", r.stderr[-2000:])
    return r.returncode


def _uv(cmd: str, timeout_note: str = ""):
    import subprocess, sys
    print("RUN:", cmd, flush=True)
    p = subprocess.Popen(f"cd {REPO_DIR} && {cmd}", shell=True, executable="/bin/bash",
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    for line in p.stdout:
        sys.stdout.write(line)
        tail.append(line)
        if len(tail) > 400:
            tail.pop(0)
    p.wait()
    vol.commit()
    if p.returncode != 0:
        raise RuntimeError(f"command failed rc={p.returncode}\n" + "".join(tail[-60:]))
    return p.returncode


# CoinRun tokenizer sizing: their n_latents=512 targets Minecraft 360x640 (patch 16
# => ~920 patches/frame, 512*16 dims ~= 84x compression). CoinRun is 64x64 with
# patch 8 => only 64 patches/frame, so 512 latents would be 8x EXPANSION and would
# blow up dynamics cost (dynamics tokens scale with n_latents / packing_factor).
# 32 latents * 16 dims = 512 values vs 12,288 raw => 24x compression, and 32 is
# divisible by the dynamics packing_factor of 2.
N_LATENTS = 32

# Measured by `latent_stats` over 163,840 latent vectors encoded by our trained
# tok-coinrun tokenizer. dynamics normalizes latents by these; coinrun.yaml
# doesn't define them (they normally come from the Minecraft-only tokenize script).
COINRUN_LATENT_MEAN = [
    -0.21121462, -0.12767113, 0.14287420, 0.09847240, -0.10590467, 0.22086225,
    0.66683005, 0.18730359, 0.00232001, -0.02899681, 0.09015914, -0.19754437,
    -0.13149353, 0.18965001, 0.12063031, 0.01544591,
]
COINRUN_LATENT_STD = [
    0.35802462, 0.41334503, 0.38582306, 0.35531719, 0.42243692, 0.31742605,
    0.50199441, 0.44736411, 0.44105706, 0.33971484, 0.40145092, 0.44624824,
    0.33804237, 0.34346987, 0.34309691, 0.40803373,
]


@app.function(image=train_img, gpu="H100", volumes={VOL: vol}, timeout=24 * 3600)
def train_tokenizer(steps: int = 10000, batch: int = 128, n_latents: int = N_LATENTS,
                    run: str = "tok-coinrun", extra: str = ""):
    """Stage 2: causal video tokenizer (MAE) on CoinRun."""
    _uv(
        f"uv run scripts/train_tokenizer.py "
        f"dataset=coinrun run_name={run} max_steps={steps} "
        f"dataset.array_record_path={VOL}/datasets/coinrun_episodes/train "
        f"dataset.dataloader_cfg.B={batch} "
        f"dataset.dataloader_cfg.short_T=16 dataset.dataloader_cfg.long_T=16 "
        f"tokenizer.encoder.n_latents={n_latents} "
        f"ckpt.save_interval_steps=1000 "
        f"hydra.run.dir={VOL}/logs/{run} {extra}"
    )


@app.function(image=train_img, gpu="H100", volumes={VOL: vol}, timeout=3600)
def latent_stats(run: str = "tok-coinrun", batches: int = 40, batch: int = 8):
    """Compute per-channel latent mean/std for CoinRun with OUR trained tokenizer.

    dynamics.yaml requires ${dataset.latent_mean}/${dataset.latent_std} (it
    normalizes latents by them), but coinrun.yaml doesn't define them -- their
    stats normally come from the Minecraft-only tokenize script. Compute directly.
    """
    script = f'''
import numpy as np, jax
from omegaconf import OmegaConf
from dreamer.data import build_iterator
from dreamer.checkpointing import TokenizerCheckpointBundle
from dreamer.parallel import build_parallel

cfg = OmegaConf.load("configs/dataset/coinrun.yaml")
cfg.array_record_path = "{VOL}/datasets/coinrun_episodes/train"
cfg.dataloader_cfg.B = {batch}
cfg.dataloader_cfg.short_T = 16
cfg.dataloader_cfg.long_T = 16
# grain multiprocessing workers fail in this standalone script
# ('NoneType' has no attribute 'Empty'/'DatasetIterator'); run in-process.
cfg.dataloader_cfg.num_workers = 0
cfg.dataloader_cfg.prefetch_buffer_size = 1
cfg.dataloader_cfg.device_prefetch_buffer_size = 0
mesh, data_sharding, mesh_rules = build_parallel("data")   # returns 3, per train_dynamics.py:174
with jax.set_mesh(mesh):
    b = TokenizerCheckpointBundle.from_pretrained("{VOL}/logs/{run}/checkpoints", mesh_rules=mesh_rules)
    tok = b.tokenizer
    it = build_iterator(cfg, seed=0)
    s = s2 = n = 0
    for i in range(int({batches})):
        batch_data = next(it)
        vids = batch_data["videos"] if isinstance(batch_data, dict) else batch_data
        z, _, _ = tok.encode(vids, deterministic=True)   # (B,T,n_latents,d_bottleneck)
        z = np.asarray(z, dtype=np.float64).reshape(-1, z.shape[-1])
        s = s + z.sum(0); s2 = s2 + (z**2).sum(0); n += z.shape[0]
    mean = s / n
    std = np.sqrt(np.maximum(s2 / n - mean**2, 1e-12))
    print("N_VECTORS", n)
    print("LATENT_MEAN=" + repr([float(x) for x in mean]))
    print("LATENT_STD=" + repr([float(x) for x in std]))
    np.savez("{VOL}/logs/{run}/latent_stats.npz", mean=mean, std=std, num_samples=n)
    import json
    json.dump({{"mean": [float(x) for x in mean], "std": [float(x) for x in std],
               "num_samples": int(n)}}, open("{VOL}/logs/{run}/latent_stats.json", "w"))
'''
    import subprocess, sys
    open("/tmp/ls.py", "w").write(script)
    r = subprocess.run(f"cd {REPO_DIR} && uv run python /tmp/ls.py",
                       shell=True, capture_output=True, text=True, executable="/bin/bash")
    print(r.stdout[-4000:]); print("STDERR:", r.stderr[-3000:])
    vol.commit()
    return r.returncode


@app.function(image=train_img, gpu="H100", volumes={VOL: vol}, timeout=24 * 3600)
def tokenize(run: str = "tok-coinrun", out: str = "coinrun_latent", extra: str = ""):
    """Stage 3: encode episodes -> latent shards + metadata/latent_stats.npz."""
    _uv(
        f"uv run scripts/tokenize_minecraft_dataset.py "
        f"dataset=coinrun "
        f"dataset.array_record_path={VOL}/datasets/coinrun_episodes/train "
        f"tokenizer_ckpt={VOL}/logs/{run}/checkpoints "
        f"output_dir={VOL}/datasets/{out} {extra}"
    )


@app.function(image=train_img, gpu="H100", volumes={VOL: vol}, timeout=24 * 3600)
def train_dynamics(steps: int = 200000, batch: int = 16, run: str = "dyn-coinrun",
                   tok: str = "tok-coinrun", seq: int = 64, extra: str = ""):
    """Stage 4: action-conditioned dynamics w/ shortcut forcing, on RAW CoinRun video.

    We use dataset=coinrun (data_type: video) so train_dynamics tokenizes on the
    fly (use_latent_data=False -> tokenizer.encode(...)). That sidesteps their
    Minecraft-only tokenize script, which has no CoinRun path.

    bootstrap_start = steps//2 mirrors their 100k-of-200k split: phase 1 is pure
    flow matching, phase 2 adds shortcut/bootstrap samples -- phase 2 is what
    makes few-step (real-time) generation possible.

    latent_mean/std come from `latent_stats` (coinrun.yaml doesn't define them;
    dynamics normalizes latents by these).
    """
    import json, os
    p = f"{VOL}/logs/{tok}/latent_stats.json"
    if os.path.exists(p):
        st = json.load(open(p))
        mean, std = st["mean"], st["std"]
    else:
        # Measured by `latent_stats` over 163,840 latent vectors from our
        # tok-coinrun tokenizer. NOTE our std ~0.4 vs their Minecraft ~0.09-0.19,
        # so reusing their values would misscale inputs ~4x.
        mean, std = COINRUN_LATENT_MEAN, COINRUN_LATENT_STD
    ms = "[" + ",".join(f"{x:.8f}" for x in mean) + "]"
    ss = "[" + ",".join(f"{x:.8f}" for x in std) + "]"
    _uv(
        f"uv run scripts/train_dynamics.py dataset=coinrun run_name={run} "
        f"max_steps={steps} bootstrap_start={steps // 2} "
        f"tokenizer_ckpt={VOL}/logs/{tok}/checkpoints "
        f"dataset.array_record_path={VOL}/datasets/coinrun_episodes/train "
        f"dataset.dataloader_cfg.B={batch} "
        f"dataset.dataloader_cfg.short_T={seq} dataset.dataloader_cfg.long_T={seq} "
        f"'+dataset.latent_mean={ms}' '+dataset.latent_std={ss}' "
        f"ckpt.save_interval_steps=5000 "
        f"hydra.run.dir={VOL}/logs/{run} {extra}"
    )


@app.function(image=train_img, gpu="H200", volumes={VOL: vol}, timeout=6 * 3600)
def eval_only(run: str = "dyn-coinrun-mg", ctx_len: int = 4, horizon: int = 144,
              num_videos: int = 64):
    """FVD eval + rollouts on the trained dynamics model.

    Fixes vs the pipeline's first attempt:
      - NO tokenizer_ckpt: eval_fvd.yaml has no such key (Hydra strict-struct
        error). The dynamics checkpoint is self-contained -- it bundles the
        tokenizer so rollouts can be decoded.
      - ctx+horizon must be <= record length (160). 4+144=148 OK, and
        144 % fvd_chunk_size(16) == 0 as their README requires.
      - video_dir on the Volume so rollouts persist.
    """
    ms = "[" + ",".join(f"{x:.8f}" for x in COINRUN_LATENT_MEAN) + "]"
    ss = "[" + ",".join(f"{x:.8f}" for x in COINRUN_LATENT_STD) + "]"
    # eval_fvd.py writes MP4 via iio.imwrite(plugin="pyav") but their pyproject
    # only declares imageio[ffmpeg] -> ImportError at the save step (generation
    # itself works fine). Install the `av` backend into their uv venv first.
    _uv("uv pip install av")
    _uv(
        f"uv run scripts/eval_fvd.py dataset=coinrun "
        f"dataset.array_record_path={VOL}/datasets/coinrun_episodes/test "
        f"'+dataset.latent_mean={ms}' '+dataset.latent_std={ss}' "
        f"dynamics_ckpt={VOL}/logs/{run}/checkpoints "
        f"ctx_length={ctx_len} horizon={horizon} num_videos={num_videos} "
        f"video_dir={VOL}/logs/{run}-eval/videos "
        f"hydra.run.dir={VOL}/logs/{run}-eval"
    )


@app.function(image=train_img, gpu="H200:8", volumes={VOL: vol}, timeout=3600)
def multi_gpu_test(steps: int = 300, batch: int = 32):
    """Verify 8xH200 data-parallel works + measure speed.

    Single H200 runs 1.05 it/s => 200k steps ~= 53h, which BREAKS the plan:
    Modal caps jobs at 24h (~90k steps), so we'd never reach bootstrap_start=100k
    -- the shortcut/bootstrap phase that enables few-step real-time generation.
    Data-parallel across 8 GPUs should fix that. batch must be divisible by device count.
    """
    ms = "[" + ",".join(f"{x:.8f}" for x in COINRUN_LATENT_MEAN) + "]"
    ss = "[" + ",".join(f"{x:.8f}" for x in COINRUN_LATENT_STD) + "]"
    _uv(
        f"uv run scripts/train_dynamics.py dataset=coinrun run_name=dyn-mgtest "
        f"max_steps={steps} bootstrap_start={steps // 2} "
        f"tokenizer_ckpt={VOL}/logs/tok-coinrun/checkpoints "
        f"dataset.array_record_path={VOL}/datasets/coinrun_episodes/train "
        f"'+dataset.latent_mean={ms}' '+dataset.latent_std={ss}' "
        f"dataset.dataloader_cfg.B={batch} "
        f"dataset.dataloader_cfg.short_T=64 dataset.dataloader_cfg.long_T=64 "
        f"ckpt.max_to_keep=0 hydra.run.dir={VOL}/logs/dyn-mgtest"
    )


@app.function(image=train_img, gpu="H200", volumes={VOL: vol}, timeout=86400)
def pipeline(steps: int = 80000, batch: int = 8, run: str = "dyn-coinrun-mg",
             tok: str = "tok-coinrun", seq: int = 64):
    """SELF-CHAINING unattended pipeline: dynamics -> FVD eval -> rollouts.

    Runs entirely server-side on Modal so it survives the laptop closing and any
    local session ending. Writes PIPELINE_STATUS.json to the Volume after every
    stage so progress/failures are inspectable without reading container logs.
    Resumable: skips dynamics if the final checkpoint already exists.
    """
    import json, os, subprocess, sys, time

    status_path = f"{VOL}/PIPELINE_STATUS.json"
    state = {"started": time.time(), "stages": {}}

    def save(stage, **kw):
        state["stages"][stage] = {**kw, "t": time.time()}
        with open(status_path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        vol.commit()

    def sh(stage, cmd):
        save(stage, status="running", cmd=cmd)
        print(f"\n=== [{stage}] {cmd}\n", flush=True)
        p = subprocess.Popen(f"cd {REPO_DIR} && {cmd}", shell=True, executable="/bin/bash",
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
        tail = []
        for line in p.stdout:
            sys.stdout.write(line)
            tail.append(line)
            if len(tail) > 300:
                tail.pop(0)
        p.wait()
        vol.commit()
        ok = p.returncode == 0
        save(stage, status="ok" if ok else "FAILED", rc=p.returncode,
             tail="".join(tail[-40:]) if not ok else "")
        return ok

    ms = "[" + ",".join(f"{x:.8f}" for x in COINRUN_LATENT_MEAN) + "]"
    ss = "[" + ",".join(f"{x:.8f}" for x in COINRUN_LATENT_STD) + "]"

    def _common(split):
        return (f"dataset=coinrun "
                f"dataset.array_record_path={VOL}/datasets/coinrun_episodes/{split} "
                f"'+dataset.latent_mean={ms}' '+dataset.latent_std={ss}' ")

    common = _common("train")

    # --- Stage 1: dynamics (skip if already finished) -----------------------
    done_ckpt = f"{VOL}/logs/{run}/checkpoints/{steps - 1}"
    if os.path.exists(done_ckpt):
        save("dynamics", status="skipped (already complete)")
    else:
        ok = sh("dynamics",
                f"uv run scripts/train_dynamics.py {common} run_name={run} "
                f"max_steps={steps} bootstrap_start={steps // 2} "
                f"tokenizer_ckpt={VOL}/logs/{tok}/checkpoints "
                f"dataset.dataloader_cfg.B={batch} "
                f"dataset.dataloader_cfg.short_T={seq} dataset.dataloader_cfg.long_T={seq} "
                f"ckpt.save_interval_steps=5000 hydra.run.dir={VOL}/logs/{run}")
        if not ok:
            save("pipeline", status="ABORTED at dynamics")
            return state

    # --- Stage 2: FVD eval + rollouts ---------------------------------------
    # ctx_length+horizon MUST be <= record length. Their defaults (4+240=244)
    # assume 256-frame Minecraft records; our CoinRun records are 160 frames, so
    # the eval would crash instantly. 4+144=148 <= 160, and 144 % fvd_chunk_size(16)
    # == 0 as required. Evaluate on the held-out TEST split, not train.
    sh("eval_fvd",
       f"uv run scripts/eval_fvd.py {_common('test')} "
       f"ctx_length=4 horizon=144 num_videos=64 "
       f"dynamics_ckpt={VOL}/logs/{run}/checkpoints "
       f"tokenizer_ckpt={VOL}/logs/{tok}/checkpoints "
       f"hydra.run.dir={VOL}/logs/{run}-eval")

    save("pipeline", status="COMPLETE")
    print("PIPELINE COMPLETE")
    return state


@app.local_entrypoint()
def gen_data(n_train: int = 10000, n_val: int = 500, n_test: int = 500,
             eps_per_shard: int = 25, min_len: int = 1000):
    """Generate the CoinRun dataset at their documented scale, fanned out on Modal.

    Their defaults: 10k train / 500 val / 500 test episodes, chunk_size=160,
    min_episode_length=1000 (measured yield: ~45% of random-action episodes).
    """
    specs, shard = [], 0
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        n_shards = max(1, n // eps_per_shard)
        for i in range(n_shards):
            specs.append({"split": split, "shard_idx": i, "n_eps": eps_per_shard,
                          "min_len": min_len, "seed0": 1234 + shard})
            shard += 1
    print(f"launching {len(specs)} shard jobs "
          f"({n_train}/{n_val}/{n_test} episodes, {eps_per_shard} per shard)...")
    res = list(_gen_shard.map(specs))
    for split in ("train", "val", "test"):
        rs = [r for r in res if r["split"] == split]
        if not rs:
            continue
        print(f"{split}: shards={len(rs)} records={sum(r['records'] for r in rs)} "
              f"frames={sum(r['frames'] for r in rs):,} "
              f"episodes_kept={sum(r['episodes_kept'] for r in rs)}/"
              f"{sum(r['episodes_tried'] for r in rs)} tried")


@app.local_entrypoint()
def main():
    # JAX/GPU first (higher-risk unknown); isolate failures so one doesn't hide the other.
    try:
        print("=== train smoke (JAX GPU + open-dreamer imports) ===")
        print("rc:", smoke.remote())
    except Exception as e:
        print("train smoke FAILED:", repr(e))
    try:
        print("=== datagen smoke (procgen + episode yield) ===")
        print(datagen_smoke.remote())
    except Exception as e:
        print("datagen smoke FAILED:", repr(e))
