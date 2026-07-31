#!/usr/bin/env python3
"""
Figures for the CoinRun world-model write-up, via the PaperBanana multi-agent pipeline.

Why this exists instead of the shared skill script: that script hardcodes
vlm_model="gemini-2.0-flash", which this API key cannot see (the planner agent
fails with a ClientError after 3 retries). The key does expose gemini-3.5-flash
and gemini-3-pro-image-preview, so we drive PaperBananaPipeline directly.

PaperBanana runs Retrieve -> Plan -> Style -> Render -> Critique (x N), which is
what makes it better than a single-shot image call for diagrams with structure.

Usage:
    GOOGLE_API_KEY=... python make_figures_pb.py [figure_key ...]
"""
import asyncio, os, shutil, sys, traceback

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_pb")

STYLE = (
    "Publication-quality academic methodology diagram for a machine-learning paper. "
    "Warm cream background (#F0EEE6). Charcoal (#1A1A1A) line work and text. "
    "Restrained accents: terracotta (#C15F3C) for the highlighted path, muted slate "
    "(#5A6472) for secondary elements. Clean sans-serif labels, small and legible. "
    "Flat vector illustration, orthogonal layout, generous whitespace, clear arrows. "
    "No 3D, no gradients, no drop shadows, no photorealism, no decorative clutter. "
    "Every text label must be spelled correctly and be readable at small size."
)

FIGURES = {
    "architecture": (
        "A world model's three components laid out left to right with labelled arrows between them. "
        "STAGE 1 'Video Tokenizer': a small pixel-art platformer frame enters; it is encoded into a "
        "row of 32 small rounded rectangles representing continuous latent vectors; annotate "
        "'32 latents x 16 dims, tanh bottleneck, 24x compression'. "
        "STAGE 2 'Dynamics Model': a stack of transformer blocks receives BOTH the latent row AND a "
        "separate highlighted ACTION token drawn as a small arrow-key glyph; annotate "
        "'block-causal transformer, 1.57B parameters, flow matching + shortcut forcing'. "
        "STAGE 3 'Decoder': the predicted latents are decoded back into a platformer frame. "
        "A prominent curved feedback arrow runs from the output frame back to the Stage 2 input, "
        "labelled 'autoregressive rollout'. The action token and the feedback arrow are the two "
        "elements drawn in terracotta; everything else is charcoal and slate."
    ),
    "genie_vs_dreamer": (
        "A two-column comparison with a vertical divider down the middle. "
        "LEFT column headed 'Genie (2024) - what we tried first': a frame is converted into a grid of "
        "DISCRETE integer codes drawn as numbered squares, labelled 'VQ codebook, 1024 entries, "
        "quantisation loss'; below it an action symbol with a large question mark, labelled "
        "'actions inferred, unsupervised'; at the bottom a small red-toned result box reading "
        "'control barely responded'. "
        "RIGHT column headed 'Dreamer 4 (2025) - what worked': the same frame becomes SMOOTH "
        "continuous vectors with a small tanh curve icon, labelled 'continuous latents, no "
        "quantisation'; below it a clear arrow-key glyph feeding straight into the model, labelled "
        "'ground-truth actions'; at the bottom a terracotta result box reading 'action-conditioned'. "
        "Keep both columns visually parallel so the two differences are obvious."
    ),
    "data_engine": (
        "A horizontal data pipeline of five numbered stages connected by arrows. "
        "(1) a game controller icon labelled 'procgen CoinRun (MIT)'; "
        "(2) parallel stacked boxes labelled '10,000 episodes rolled out on CPU containers'; "
        "(3) a film strip cut into segments labelled '160-frame records'; "
        "(4) a database cylinder labelled 'ArrayRecord shards'; "
        "(5) a large emphasised number card reading '9,600,000 frames'. "
        "Below stage 4, a small callout box showing the record fields as a list: "
        "raw_video, sequence_length, actions, rewards. "
        "Add a short side note near stage 2: 'self-generated - no scraped video, actions known exactly'."
    ),
    "shortcut": (
        "An explanatory two-row diagram about sampling speed. "
        "TOP ROW: a curved trajectory from a scribbled noise cloud on the left to a clean game frame "
        "on the right, traversed by MANY small identical arrows placed along the curve; label the row "
        "'standard flow matching: hundreds of small steps (slow)'. "
        "BOTTOM ROW: the identical curve traversed by only THREE large terracotta arrows; label the "
        "row 'shortcut model: a few big steps (real-time capable)'. "
        "To the right of both rows, a boxed rule reading: 'train so that ONE step of size 2d equals "
        "TWO steps of size d'. Keep the two curves vertically aligned so the contrast is immediate."
    ),
    "two_phase": (
        "A training schedule shown as one long horizontal bar spanning 0 to 80,000 steps with tick "
        "marks. The left half (0 to 40,000) is lightly shaded and labelled 'Phase 1: flow matching "
        "only - learn the velocity field'. The right half (40,000 to 80,000) is shaded terracotta and "
        "labelled 'Phase 2: add shortcut bootstrap - learn to take big steps'. A vertical dashed "
        "marker at 40,000 is labelled 'bootstrap_start'. Beneath the bar, two small aligned line "
        "charts sharing the same x-axis: the first labelled 'flow_mse' descending smoothly from 0.53 "
        "to 0.014 across the whole range; the second labelled 'boot_mse' which is flat at zero until "
        "40,000 and then appears and descends to 0.004."
    ),
    "rollout_drift": (
        "A diagram explaining compounding error in autoregressive rollouts. A left-to-right chain of "
        "small platformer frames, each connected to the next through a small box labelled 'dynamics'. "
        "Above the first four frames, a bracket labelled 'real context frames (4)'. Above all the "
        "remaining frames, a longer bracket labelled 'model now feeds on its own predictions (144)'. "
        "Running along the top, a subtle wedge that widens from left to right, labelled "
        "'error accumulates - this is where world models usually collapse'. "
        "At the far right, a small tick mark and the note 'ours stayed coherent'."
    ),
    "fvd": (
        "A measurement decomposition figure. Three horizontal bars sharing a common left baseline, "
        "drawn to relative length. Bar 1, shortest, labelled 'tokenizer ceiling: original vs "
        "reconstruction' with the value 16.59. Bar 2, medium, labelled 'dynamics only: reconstruction "
        "vs dream' with the value 23.29. Bar 3, longest and drawn in terracotta, labelled "
        "'end-to-end: original vs dream' with the value 32.19. Under the bars a caption reads "
        "'FVD - lower is better'. To the left of each bar, a tiny film-frame icon."
    ),
}


async def build(key: str, prompt: str) -> bool:
    from paperbanana import PaperBananaPipeline, GenerationInput, DiagramType
    from paperbanana.core.config import Settings

    settings = Settings(
        # gemini-2.0-flash (the library/skill default) is NOT visible to this key.
        vlm_model="gemini-3.5-flash",
        image_model="gemini-3-pro-image-preview",
        refinement_iterations=2,
        output_dir=OUT,
    )
    result = await PaperBananaPipeline(settings=settings).generate(
        GenerationInput(
            source_context=prompt,
            communicative_intent=STYLE,
            diagram_type=DiagramType.METHODOLOGY,
        )
    )
    dest = os.path.join(OUT, f"pb_{key}.png")
    if result.image_path and os.path.exists(result.image_path):
        shutil.copy2(result.image_path, dest)
        print(f"  OK  {key}  ({len(result.iterations)} refinement rounds) -> {dest}")
        return True
    print(f"  FAIL {key}: pipeline returned no image")
    return False


def main():
    os.makedirs(OUT, exist_ok=True)
    if not os.environ.get("GOOGLE_API_KEY"):
        sys.exit("set GOOGLE_API_KEY")
    wanted = sys.argv[1:] or list(FIGURES)
    ok = 0
    for key in wanted:
        if key not in FIGURES:
            print(f"  skip {key}: unknown"); continue
        print(f"[paperbanana] {key} ...", flush=True)
        try:
            ok += bool(asyncio.run(build(key, FIGURES[key])))
        except Exception as e:
            print(f"  FAIL {key}: {type(e).__name__}: {str(e)[:200]}")
            traceback.print_exc(limit=1)
    print(f"\n{ok}/{len(wanted)} figures generated into {OUT}")


if __name__ == "__main__":
    main()
