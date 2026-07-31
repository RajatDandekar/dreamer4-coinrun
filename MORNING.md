# ☀️ Good morning — you can play inside a world model you trained overnight

## → Share this with the team: **https://dreamer-worldmodel.vercel.app**
Public (no login), explains what we built + how to play, with the live interactive demo embedded.
Backend = the Modal GPU app `https://teamvizuara--genie-play-player-web.modal.run` (CORS-enabled; the
Vercel page calls it). Cleaner Vercel landing lives in `site/index.html`.

Pick a starting scene, then press the **arrow keys** (or click). Every frame you see is
**hallucinated by a neural network we trained from scratch tonight** — there is no game engine
behind it. Hit **"✨ Compare all 7 actions"** to watch the same scene dream differently under each
of the model's learned latent actions.

*(It's kept warm on a GPU, so it should respond immediately. Each dream takes ~1s to form —
the model is un-distilled, 16 MaskGIT steps per frame, i.e. real Genie-at-this-scale speed.)*

---

## What actually got built overnight (end to end, on Modal)

A faithful **Genie-style world model**, trained **from scratch on our own legally-clean data**:

1. **Data engine** — `procgen` CoinRun (MIT) → our GenieRedux dataset format, parallel fan-out,
   diversified policy. **2,000 episodes / 121,728 frames**, with 7-action labels. (Decision A: no
   scraped video — self-generated CC0 data, shippable.)
2. **Video tokenizer** (100M, ST-ViViT VQ-VAE) — trained from scratch, 15k steps on 8×H100.
   Reconstruction loss **0.05 → 0.0005** (~PSNR mid-30s, INSAIT-grade).
3. **World model** (~250M) — unsupervised **Latent Action Model** + **MaskGIT dynamics**, trained
   to completion (**40,000 steps** on 8×H100, ~4.5h + ~14h detached). It dreams coherent CoinRun
   trajectories frame-by-frame, conditioned on actions.
4. **Served it** — a deployed Modal GPU web app that reloads the model and generates dreams live,
   with a browser client so you can play inside it.

**Metrics:** frame PSNR ~18–19, SSIM ~0.53. Controllability ΔPSNR ≈ 0 (honest: per-frame control
is *subtle* at this scale — see below).

## The honest part (so you're not surprised)

- **The visuals are simple.** 64×64 CoinRun with our policy = brown ground, sky, and dark
  character/platform blocks. It's recognizably CoinRun and the dreams are temporally coherent, but
  it's abstract, not photoreal. That's the CoinRun case-study scale (what fits an overnight run),
  not the ~1B multi-style endgame.
- **Control is subtle.** The 7 actions were learned *unsupervised*; they do steer the dream, but
  gently (one CoinRun frame barely differs between "right" and "jump"). The "compare 7 actions"
  view makes the differences easiest to see. Clear, snappy control is what the **guided** variant
  and more training/scale buy — that's the next lever.
- **It's a research artifact, not a shipped game.** Un-distilled (~1 fps), single-style, 250M.
  The point tonight was to prove the *whole pipeline works from scratch on our own data* — it does.

## Where this sits on the plan
✅ data engine · ✅ tokenizer · ✅ LAM+dynamics (trained) · ✅ **served & playable**
⬜ scale toward ~1B multi-style · ⬜ train the **guided** model for crisp control · ⬜ few-step
distillation for real-time (25→3-4 steps) · ⬜ richer/higher-res + multi-game data.

## Files & ops
- Demo URL: https://teamvizuara--genie-play-player-web.modal.run  (Modal app `genie-play`)
- Code: `serve.py` (server), `client.html` (UI), `modal_app.py` (data+train+eval), `STATUS.md` (full log).
- Trained weights on Volume `genie-platformer`: tokenizer `model-15000.pt`, world model `model-40000.pt`.
- **Cost note:** the demo keeps 1×H100 warm (`min_containers=1`) so it's instant for you. To stop it:
  `modal app stop genie-play`. Everything else is idle.

I'll keep polishing (a story/showcase landing + more scenes) while you sleep and note anything below.
