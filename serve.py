"""
Genie-Play — serve the from-scratch world model as an interactive browser demo.

The trained model (tokenizer model-15000 + genie_redux model-40000) lives on the
`genie-platformer` Volume. This app loads it once per warm GPU container and exposes:
  - Player.dream(scene, actions)  -> generated frames (base64 PNG)   [test via local_entrypoint]
  - web()  -> FastAPI: GET / (client) + POST /api/dream               [added after dream() verified]

Run test:   modal run serve.py::test
Deploy web: modal deploy serve.py
"""
import modal

app = modal.App("genie-play")
vol = modal.Volume.from_name("genie-platformer")
VOL = "/vol"
REPO_URL = "https://github.com/insait-institute/GenieRedux.git"
REPO_DIR = "/root/GenieRedux"

TOKZ = f"{VOL}/checkpoints/tokenizer/tokenizer/model-15000.pt"
GENIE = f"{VOL}/checkpoints/genie_redux/genie_redux/model-40000.pt"
DATA = f"{VOL}/datasets/coinrun_prod_v2/coinrun_prod_v2"

# Same tested neurips stack as training (cached image; adds fastapi/uvicorn for serving).
full = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.4.1", "torchvision==0.19.1",
                 index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(
        "accelerate==0.29.1", "beartype==0.18.2", "einops==0.8.0", "einx==0.3.0",
        "lovely-numpy==0.2.10", "lovely-tensors==0.1.15", "matplotlib==3.9.2",
        "opencv-python==4.10.0.84", "pandas==2.1.4", "Pillow==10.0.1", "prettytable==3.11.0",
        "scipy==1.14.1", "torch-fidelity==0.3.0", "torchcache==0.5.2", "torchmetrics==1.2.1",
        "tqdm==4.66.1", "traitlets==5.7.1", "vector-quantize-pytorch==1.16.2", "wandb==0.16.2",
        "tyro==0.8.10", "hydra-core==1.3.2", "dill==0.3.8", "pyarrow==18.1.0",
        "huggingface-hub==0.25.1", "numpy==1.26.4", "procgen==0.10.7",
        "fastapi==0.115.0",
    )
    .run_commands(f"git clone --depth 1 --branch neurips {REPO_URL} {REPO_DIR}")
    .add_local_file("/Users/raj/Downloads/Vizuara/genie-platformer/client.html", "/root/client.html")
)


def build_config():
    """Reconstruct the exact config construct_model needs (values from training config dump)."""
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "model": "genie_redux", "tokenizer_fpath": TOKZ,
        "train": {"wandb_mode": "disabled"},
        "tokenizer": {"dim": 512, "num_frames": 16, "codebook_size": 1024, "image_size": 64,
                      "patch_size": 4, "temporal_patch_size": 1, "num_blocks": 8, "dim_head": 64,
                      "heads": 8, "ff_mult": 4, "vq_loss_weight": 1.0, "recons_loss_weight": 1.0},
        "lam": {"dim": 512, "num_frames": 16, "codebook_size": 7, "num_actions": 7, "image_size": 64,
                "patch_size": 4, "temporal_patch_size": 1, "num_blocks": 8, "dim_head": 64,
                "heads": 8, "ff_mult": 4, "vq_loss_weight": 1.0, "recons_loss_weight": 1.0},
        "dynamics": {"dim": 512, "action_dim": 7, "image_size": 64, "patch_size": 4,
                     "temporal_patch_size": 1, "num_blocks": 12, "dim_head": 64, "heads": 8,
                     "ff_mult": 4, "max_seq_len": 8000, "sample_temperature": 1.0,
                     "sample_num_frames": 15},
    })


@app.cls(image=full, gpu="H100", volumes={VOL: vol}, min_containers=1,
         scaledown_window=600, timeout=1200)
class Player:
    @modal.enter()
    def load(self):
        import os, sys, glob, torch, numpy as np
        from PIL import Image
        os.chdir(REPO_DIR)
        sys.path.insert(0, REPO_DIR)
        from models import construct_model
        self.torch, self.np = torch, np

        model = construct_model(build_config())
        sd = torch.load(GENIE, map_location="cpu")
        model.load_state_dict(sd["model"])  # custom loader; nested {dynamics, latent_action_model}
        print("loaded genie_redux (dynamics+lam; tokenizer from construct_model)")
        self.model = model.cuda().eval()

        # start-frame "scenes" = first 2 real frames of several dataset sessions
        self.scenes = []
        for i in range(12):
            fs = sorted(glob.glob(f"{DATA}/{i:06d}/000000/frames/*.jpg"))[:2]
            if len(fs) < 2:
                continue
            fr = [np.array(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0 for f in fs]
            self.scenes.append(np.stack(fr))  # (2,64,64,3)
        print(f"loaded {len(self.scenes)} scenes")

    def _dream(self, scene_idx, actions, steps=16):
        torch, np = self.torch, self.np
        prime = self.scenes[scene_idx % len(self.scenes)]              # (2,64,64,3)
        prime_t = torch.tensor(prime).permute(3, 0, 1, 2).unsqueeze(0).cuda()  # (1,3,2,64,64)
        n = len(actions)
        # model needs actions of length prime_frames(2)+num_frames-1; generate n frames -> n+1 actions.
        acts_list = [actions[0]] + list(actions)  # prepend one for the prime transition
        acts = torch.tensor([acts_list], dtype=torch.long).cuda()  # (1,n+1)
        with torch.no_grad():
            video = self.model.sample(prime_frames=prime_t, actions=acts,
                                      num_frames=n, inference_steps=steps)
        return video.clamp(0, 1)[0].permute(1, 2, 3, 0).cpu().numpy()   # (n,64,64,3)

    def _frames_b64(self, video, size=256):
        import base64, io
        from PIL import Image
        out = []
        for fr in video:
            img = Image.fromarray((fr * 255).astype("uint8")).resize((size, size), Image.NEAREST)
            buf = io.BytesIO(); img.save(buf, format="PNG")
            out.append(base64.b64encode(buf.getvalue()).decode())
        return out

    @modal.method()
    def dream(self, scene_idx=0, actions=None, steps=16):
        if actions is None:
            actions = [1] * 15
        video = self._dream(scene_idx, actions, steps)
        return self._frames_b64(video)

    def _scenes_b64(self):
        import base64, io
        from PIL import Image
        out = []
        for sc in self.scenes:
            img = Image.fromarray((sc[0] * 255).astype("uint8")).resize((80, 80), Image.NEAREST)
            buf = io.BytesIO(); img.save(buf, format="PNG")
            out.append(base64.b64encode(buf.getvalue()).decode())
        return out

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, Request
        from fastapi.responses import HTMLResponse, JSONResponse
        from fastapi.middleware.cors import CORSMiddleware
        html = open("/root/client.html").read()
        api = FastAPI()
        api.add_middleware(CORSMiddleware, allow_origins=["*"],
                           allow_methods=["*"], allow_headers=["*"])

        @api.get("/", response_class=HTMLResponse)
        def index():
            return html

        @api.get("/api/scenes")
        def scenes():
            return JSONResponse({"thumbs": self._scenes_b64()})

        @api.post("/api/dream")
        async def dream_ep(req: Request):
            b = await req.json()
            acts = [int(a) for a in b.get("actions", [1])][-15:] or [1]
            video = self._dream(int(b.get("scene", 0)), acts, int(b.get("steps", 16)))
            return JSONResponse({"frames": self._frames_b64(video)})

        @api.post("/api/dream_all")
        async def dream_all_ep(req: Request):
            b = await req.json()
            sc, steps = int(b.get("scene", 0)), int(b.get("steps", 16))
            rollouts = [self._frames_b64(self._dream(sc, [a] * 15, steps)) for a in range(7)]
            return JSONResponse({"rollouts": rollouts})

        return api


@app.function(image=full, gpu="H100", volumes={VOL: vol}, timeout=1200)
def probe():
    """Server-side verification with a step-by-step status file on the Volume (reliable diagnostics).
    Run: modal run --detach serve.py::probe ; then read /vol/demo_out/status.txt"""
    import os, sys, glob, traceback
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # clean traceback if a CUDA assert fires
    os.makedirs(f"{VOL}/demo_out", exist_ok=True)
    log = []

    def L(m):
        log.append(str(m))
        with open(f"{VOL}/demo_out/status.txt", "w") as fh:
            fh.write("\n".join(log))
        vol.commit()

    try:
        import torch, numpy as np
        from PIL import Image
        os.chdir(REPO_DIR); sys.path.insert(0, REPO_DIR)
        from models import construct_model
        L("imports ok")
        model = construct_model(build_config()); L("model constructed")
        sd = torch.load(GENIE, map_location="cpu")
        model.load_state_dict(sd["model"])  # custom loader handles nested {dynamics, latent_action_model}
        L("state loaded (dynamics + lam; tokenizer from construct_model)")
        model = model.cuda().eval(); L("model on cuda")
        fs = sorted(glob.glob(f"{DATA}/000000/000000/frames/*.jpg"))[:2]
        fr = [np.array(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0 for f in fs]
        prime = torch.tensor(np.stack(fr)).permute(3, 0, 1, 2).unsqueeze(0).cuda()
        L(f"prime shape {tuple(prime.shape)}")
        for action in [0, 1, 3, 5]:
            # actions length = num_first_frames(2) + num_frames(15) - 1 = 16
            acts = torch.tensor([[action] * 16], dtype=torch.long).cuda()
            with torch.no_grad():
                video = model.sample(prime_frames=prime, actions=acts, num_frames=15, inference_steps=16)
            video = video.clamp(0, 1)[0].permute(1, 2, 3, 0).cpu().numpy()
            L(f"action {action}: video {video.shape} range[{video.min():.3f},{video.max():.3f}]")
            imgs = [Image.fromarray((f * 255).astype("uint8")).resize((256, 256), Image.NEAREST) for f in video]
            imgs[0].save(f"{VOL}/demo_out/probe_a{action}.gif", save_all=True,
                         append_images=imgs[1:], duration=140, loop=0)
        L("PROBE OK")
    except Exception as e:
        L("ERROR: " + repr(e))
        L(traceback.format_exc())
    vol.commit()
