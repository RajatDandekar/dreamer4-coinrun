"""
Genie-for-Platformers — Phase 0 Modal harness.

Phase 0 = correctness gate: prove we can run GenieRedux (the results-validated open
Genie reproduction) on Modal, using INSAIT's *pretrained CoinRun weights*, BEFORE we
spend anything on the ~1B multi-style run.

Repo:   github.com/insait-institute/GenieRedux  (branch `neurips` = CoinRun case study
        WITH pretrained weights; entrypoint = `accelerate launch main.py +config=...`)
Weights: HuggingFace `INSAIT-Institute/GenieRedux` (public):
   - GenieRedux_Tokenizer_CoinRun_100mln_v1.0   (ST-ViViT VQ tokenizer, 1024 codebook, patch 4)
   - GenieRedux_CoinRun_250mln_v1.0             (full Genie: LAM + MaskGIT dynamics)
   - GenieRedux_Guided_CoinRun_80mln_v1.0       (guided: ground-truth actions, no LAM)
   - GenieRedux-G_RetroAct-v1.5_...260mln_v1.5  (multi-env platformers+shooters — for later)

Run (from the project venv that has the modal SDK + ~/.modal.toml auth):
   ./.venv-modal/bin/modal run modal_app.py::download_weights
   ./.venv-modal/bin/modal run modal_app.py::inspect_weights

Later phases (train/eval) shell out to the repo's own `main.py` via accelerate on an
8xH100 node; those are wired below and become live once the CoinRun dataset adapter lands.
"""

import modal

app = modal.App("genie-platformer-p0")

# Persistent storage for weights, datasets, checkpoints (survives across runs).
vol = modal.Volume.from_name("genie-platformer", create_if_missing=True)
VOL = "/vol"

REPO_URL = "https://github.com/insait-institute/GenieRedux.git"
REPO_DIR = "/root/GenieRedux"

# HF weights we pull (public repo, not gated).
HF_REPO = "INSAIT-Institute/GenieRedux"
COINRUN_WEIGHTS = [
    "GenieRedux_Tokenizer_CoinRun_100mln_v1.0",
    "GenieRedux_CoinRun_250mln_v1.0",
    "GenieRedux_Guided_CoinRun_80mln_v1.0",
]

# ---------------------------------------------------------------------------
# Images
#   - `light`: just torch + hf_hub, for pulling/inspecting weights (no GPU, cheap).
#   - `full` : torch cu126 + GenieRedux deps + the repo (neurips branch), for eval/train.
#     Python 3.10 to match the neurips branch (tested on 3.10) and keep procgen wheels valid.
# ---------------------------------------------------------------------------
light = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("huggingface-hub==0.34.4", "numpy==2.2.6")
)

full = (
    # Match the neurips branch's TESTED env (torch cu121 era) — avoids accelerate/torch
    # API drift inside their Trainer. Python 3.10 (their tested version + procgen wheels).
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.4.1", "torchvision==0.19.1",
                 index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(
        # exact neurips GenieRedux genie_redux_env.yaml pip list (tested-to-work):
        "accelerate==0.29.1", "beartype==0.18.2", "einops==0.8.0", "einx==0.3.0",
        "lovely-numpy==0.2.10", "lovely-tensors==0.1.15", "matplotlib==3.9.2",
        "opencv-python==4.10.0.84", "pandas==2.1.4", "Pillow==10.0.1",
        "prettytable==3.11.0", "scipy==1.14.1", "torch-fidelity==0.3.0",
        "torchcache==0.5.2", "torchmetrics==1.2.1", "tqdm==4.66.1", "traitlets==5.7.1",
        "vector-quantize-pytorch==1.16.2", "wandb==0.16.2", "tyro==0.8.10",
        "hydra-core==1.3.2", "dill==0.3.8", "pyarrow==18.1.0", "huggingface-hub==0.25.1",
        "numpy==1.26.4",  # <2 for procgen/old-gym + the old-era stack
        # our Decision-A self-generated data route (procgen = MIT):
        "procgen==0.10.7",
    )
    .run_commands(
        f"git clone --depth 1 --branch neurips {REPO_URL} {REPO_DIR}",
        # Fix a genuine bug: Trainer defaults max_valid_size=128 and caps the valid Subset
        # to min(len,128), but leaves self.max_valid_size=128 — so load_log_data indexes
        # [0,31,63,95] into a smaller valid batch and crashes on any dataset with <128
        # valid sequences. Clamp to the actual valid-set size (correct for any dataset).
        f"sed -i 's/^        self.max_valid_size = max_valid_size$/        self.max_valid_size = min(max_valid_size, len(self.valid_ds))/' {REPO_DIR}/training/trainer.py",
        # Guard teardown: train.py always calls dist.destroy_process_group(), which asserts
        # in single-process runs (no group initialized). The 8-GPU run initializes one, so
        # this only bites single-GPU smokes — make it conditional.
        f"sed -i 's/^    dist.destroy_process_group()$/    dist.is_initialized() and dist.destroy_process_group()/' {REPO_DIR}/train.py",
    )
)


# ---------------------------------------------------------------------------
# Phase 0.1 — pull pretrained CoinRun weights to the Volume (cheap, no GPU)
# ---------------------------------------------------------------------------
@app.function(image=light, volumes={VOL: vol}, timeout=1800)
def download_weights():
    import os, shutil
    from huggingface_hub import hf_hub_download

    for name in COINRUN_WEIGHTS:
        dst = f"{VOL}/checkpoints/{name}"
        os.makedirs(dst, exist_ok=True)
        src = hf_hub_download(repo_id=HF_REPO, filename=f"{name}.pt", local_dir=dst)
        # match the repo's own convention: <name>/model.pt
        final = os.path.join(dst, "model.pt")
        if os.path.abspath(src) != os.path.abspath(final):
            shutil.move(src, final)
        size_mb = os.path.getsize(final) / 1e6
        print(f"  ✓ {name}: {size_mb:,.1f} MB -> {final}")
    vol.commit()
    print("download_weights: done")


# ---------------------------------------------------------------------------
# Phase 0.1b — inspect the weights (param count + module keys) to confirm they
# match the paper's spec (250M full model, 100M tokenizer) end-to-end on Modal.
# ---------------------------------------------------------------------------
@app.function(image=light, volumes={VOL: vol}, timeout=900)
def inspect_weights():
    import os, torch

    for name in COINRUN_WEIGHTS:
        p = f"{VOL}/checkpoints/{name}/model.pt"
        if not os.path.exists(p):
            print(f"  ! missing {p} — run download_weights first")
            continue
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        if hasattr(sd, "state_dict"):
            sd = sd.state_dict()
        # unwrap common wrappers
        for k in ("state_dict", "model_state_dict", "module"):
            if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
                sd = sd[k]
        n_params = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
        top = sorted({str(k).split(".")[0] for k in sd.keys()})
        print(f"\n== {name} ==")
        print(f"   params: {n_params/1e6:,.1f}M   tensors: {len(sd)}")
        print(f"   top-level modules: {top[:12]}")


# ---------------------------------------------------------------------------
# Phase 0.2 / P1 — CoinRun dataset generation (procgen -> GenieRedux DatasetFileStructure)
#   Writes  <root>/<instance:06d>/<session:06d>/frames/<frame:06d>.jpg  +  actions.json
#   and a root info.json — the exact layout data/data.py's EnvironmentDataset reads.
#   This is our Decision-A self-generated CC0 data route (procgen = MIT). The pretrained
#   INSAIT weights learned on 2018-coinrun, so procgen frames are slightly OOD for THEM
#   (fine — used only as a plumbing check); for OUR from-scratch training this IS the data.
#
#   Policy: right-biased run+jump heuristic (procgen coinrun combos: RIGHT=7, RIGHT+UP=8,
#   UP=5, LEFT=1) so the character actually traverses levels — uniform-random barely moves.
# ---------------------------------------------------------------------------
# procgen platformer-family games (2D, 64x64, shared 15-action space) — "multi-style".
PROCGEN_PLATFORMERS = ["coinrun", "jumper", "ninja", "leaper", "climber"]

# Per-episode behavior profiles for BROAD world-model coverage (procgen combos:
# LEFT=1, LEFT+UP=2, DOWN=3, NOOP=4, UP/JUMP=5, RIGHT=7, RIGHT+UP=8). Diversified so the
# corpus covers running both directions, jumping, falling, dying, idling — not one behavior.
BEHAVIOR_PROFILES = [
    [7, 7, 7, 7, 8, 8, 5, 1],   # runner-right
    [1, 1, 1, 1, 2, 2, 5, 7],   # runner-left
    [5, 5, 8, 8, 2, 2, 7, 1],   # jumpy
    [7, 1, 5, 8, 2, 3, 4, 7, 1],  # explorer (varied incl. noop/down)
]

# Drive the env with procgen's 15-action combos, but STORE a 7-action CoinRun label (0-6)
# so ground-truth actions match the model's action_dim=7 (GenieRedux CoinRun). Genie's LAM
# is unsupervised (ignores these at train time), but the generation/eval path conditions on
# them — a 15-valued label there indexes out of bounds. 0=noop 1=right 2=left 3=jump
# 4=right+jump 5=left+jump 6=down.
PROCGEN_TO_LABEL = {4: 0, 7: 1, 1: 2, 5: 3, 8: 4, 2: 5, 3: 6}


@app.function(image=full, volumes={VOL: vol}, timeout=2 * 3600, cpu=2.0)
def _gen_chunk(spec):
    """Generate one chunk of instances (procgen -> GenieRedux DatasetFileStructure)."""
    import os, json, random
    import numpy as np
    from PIL import Image
    from procgen import ProcgenGym3Env

    ds_name, n_steps = spec["ds_name"], spec["n_steps"]
    root = f"{VOL}/datasets/{ds_name}/{ds_name}"
    n_frames = 0
    for inst_id, game, level in spec["instances"]:
        rng = random.Random(inst_id)
        policy = rng.choice(BEHAVIOR_PROFILES)
        env = ProcgenGym3Env(num=1, env_name=game, start_level=level, num_levels=1,
                             distribution_mode="hard", rand_seed=level)
        sdir = f"{root}/{inst_id:06d}/{0:06d}"
        fdir = f"{sdir}/frames"
        os.makedirs(fdir, exist_ok=True)
        _, obs, first = env.observe()
        Image.fromarray(obs["rgb"][0]).save(f"{fdir}/{0:06d}.jpg", quality=95)
        actions = []
        t = 1
        while t < n_steps:
            a = rng.choice(policy)
            env.act(np.array([a], dtype=np.int32))
            _, obs, first = env.observe()
            Image.fromarray(obs["rgb"][0]).save(f"{fdir}/{t:06d}.jpg", quality=95)
            label = PROCGEN_TO_LABEL.get(int(a), 0)  # store 7-action CoinRun label (0-6)
            actions.append({"src_id": t - 1, "tgt_id": t, "action": label, "extras": {"game": game}})
            t += 1
            if bool(first[0]):
                break
        with open(f"{sdir}/actions.json", "w") as f:
            json.dump(actions, f)
        n_frames += len(actions) + 1
    vol.commit()
    return {"instances": len(spec["instances"]), "frames": n_frames}


@app.function(image=full, volumes={VOL: vol}, timeout=600)
def _write_info(ds_name: str, games: list):
    import os, json
    root = f"{VOL}/datasets/{ds_name}/{ds_name}"
    os.makedirs(root, exist_ok=True)
    info = {"info": {"action_space": [7], "observation_space": [64, 64],
                     "config": {"games": games, "source": "procgen", "policy": "diversified"}},
            "name": "coinrun", "generator_version": "0.1.0", "version": "2.0.0"}
    with open(f"{root}/info.json", "w") as f:
        json.dump(info, f, indent=2)
    vol.commit()


@app.local_entrypoint()
def gen_data(games: str = "coinrun", n_per_game: int = 2000, n_steps: int = 64,
             chunk: int = 50, ds_name: str = "coinrun_prod_v1"):
    """Production data engine: fan out chunked procgen rollouts across games via .map."""
    games_list = [g.strip() for g in games.split(",")]
    instances, gid = [], 0
    for game in games_list:
        for lvl in range(n_per_game):
            instances.append((gid, game, lvl))
            gid += 1
    specs = [{"instances": instances[i:i + chunk], "n_steps": n_steps, "ds_name": ds_name}
             for i in range(0, len(instances), chunk)]
    print(f"generating {len(instances)} instances across {games_list} in {len(specs)} chunks...")
    results = list(_gen_chunk.map(specs))
    ni = sum(r["instances"] for r in results)
    nf = sum(r["frames"] for r in results)
    _write_info.remote(ds_name, games_list)
    print(f"gen_data DONE: {ni} instances / {nf} frames -> datasets/{ds_name}")


# ---------------------------------------------------------------------------
# Phase 0.3 — eval / train shell out to the repo's own accelerate entrypoint.
#   These are the real mechanism; they go live once gen_coinrun lands.
# ---------------------------------------------------------------------------
@app.function(image=full, gpu="H100:8", volumes={VOL: vol}, timeout=24 * 3600)
def train(config: str = "tokenizer.yaml", dataset: str = "coinrun_prod_v1",
          num_processes: int = 8, batch_size: int = 8, grad_accum: int = 1,
          steps: int = 150000, save_every: int = 5000, validate_every: int = 5000,
          extra: str = ""):
    """Production training on one 8xH100 node (DDP), checkpoint/validate to the Volume."""
    import subprocess
    cmd = (
        f"cd {REPO_DIR} && accelerate launch --num_processes={num_processes} "
        f"--num_machines=1 --mixed_precision=bf16 main.py +config={config} ++config.mode=train "
        f"++config.train.dataset_root_dpath={VOL}/datasets ++config.train.dataset_name={dataset} "
        f"++config.train.save_root_dpath={VOL}/checkpoints ++config.train.wandb_dpath={VOL}/logs "
        f"++config.train.wandb_mode=disabled ++config.train.num_train_steps={steps} "
        f"++config.train.batch_size={batch_size} ++config.train.grad_accum={grad_accum} "
        f"++config.train.save_model_every={save_every} ++config.train.validate_every={validate_every} "
        f"++config.train.log_every=50 {extra}"
    )
    print("RUN:", cmd)
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")
    vol.commit()


@app.function(image=full, gpu="H100:1", volumes={VOL: vol}, timeout=3 * 3600)
def train_smoke(config: str = "tokenizer.yaml", steps: int = 300, batch_size: int = 2,
                dataset: str = "coinrun_v2.0.0", cpu: bool = False,
                validate_every: int = 100000, extra: str = ""):
    """Single-GPU training smoke: validates model construction + train loop + checkpoint on Modal.
    Bar: no crash, loss decreases, a checkpoint lands on the Volume.
    cpu=True => run on CPU (slow) for a clean Python traceback when a CUDA assert hides it."""
    import subprocess
    if cpu:
        launch = "accelerate launch --cpu --num_processes=1 --num_machines=1"
        env = ""
    else:
        launch = "accelerate launch --num_processes=1 --num_machines=1 --mixed_precision=bf16"
        env = "CUDA_LAUNCH_BLOCKING=1 "  # synchronous kernels -> assert points at the real line
    cmd = (
        f"cd {REPO_DIR} && {env}{launch} main.py +config={config} ++config.mode=train "
        f"++config.train.dataset_root_dpath={VOL}/datasets ++config.train.dataset_name={dataset} "
        f"++config.train.save_root_dpath={VOL}/checkpoints ++config.train.wandb_dpath={VOL}/logs "
        f"++config.train.wandb_mode=disabled ++config.train.num_train_steps={steps} "
        f"++config.train.batch_size={batch_size} ++config.train.grad_accum=1 "
        f"++config.train.log_every=10 ++config.train.validate_every={validate_every} "
        f"++config.train.save_model_every={max(steps // 2, 1)} "
        # short warmup so loss visibly moves within a smoke run:
        f"++config.optimizer.warmup_steps=20 ++config.optimizer.linear_warmup_total_iters=20 {extra}"
    )
    print("RUN:", cmd)
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")
    vol.commit()


@app.function(image=full, gpu="H100:1", volumes={VOL: vol}, timeout=6 * 3600)
def evaluate(config: str = "genie_redux.yaml", model_fpath: str = "", tokenizer_fpath: str = "",
             extra: str = ""):
    """Reproduce CoinRun ΔPSNR / FID / PSNR / SSIM on pretrained weights (correctness gate G0)."""
    import subprocess
    cmd = (
        f"cd {REPO_DIR} && accelerate launch --num_processes=1 --mixed_precision=bf16 "
        f"main.py +config={config} ++config.mode=eval "
        f"++config.eval.action_to_take=-1 ++config.eval.inference_method=one_go "
        f"++config.eval.model_fpath={model_fpath} ++config.tokenizer_fpath={tokenizer_fpath} {extra}"
    )
    print("RUN:", cmd)
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")
    vol.commit()


@app.local_entrypoint()
def main():
    """Default: pull + inspect the pretrained CoinRun weights (the Phase 0.1 smoke gate)."""
    download_weights.remote()
    inspect_weights.remote()
