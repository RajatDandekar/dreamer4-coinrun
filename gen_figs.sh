#!/bin/bash
cd /Users/raj/Downloads/Vizuara/genie-platformer
V=.venv-modal/bin/python
G="/Users/raj/.claude/skills/write-substack/scripts/generate_figure.py"
export GOOGLE_API_KEY="${GOOGLE_API_KEY:?set GOOGLE_API_KEY in your environment}"
STYLE="clean academic diagram on a warm cream background (#faf7f2), thin charcoal strokes, muted teal and terracotta accent colours only, elegant sans-serif labels, generous whitespace, flat vector look, publication quality, no 3D, no gradients, no drop shadows"

gen () { $V "$G" --description "$1" --style "$STYLE" --output "figures/$2" --method gemini >/dev/null 2>&1 && echo "OK $2" || echo "FAIL $2"; }

gen "Two contrasting approaches side by side, split down the middle. LEFT half titled 'Genie (2024)': a frame becomes a grid of DISCRETE coloured code numbers from a codebook of 1024, below it an icon of a lock labelled 'quantised - information lost', and an action symbol with a question mark labelled 'actions guessed, unsupervised'. RIGHT half titled 'Dreamer 4 (2025)': the same frame becomes smooth CONTINUOUS vectors with a tanh curve symbol, labelled 'continuous, no quantisation', and a clear labelled arrow key symbol labelled 'true actions given'. A vertical divider between them." fig2_genie_vs_dreamer.png

gen "A data pipeline flowing left to right in five labelled stages with arrows: 1) a game controller icon labelled 'procgen CoinRun, 10,000 episodes'; 2) 'random policy rollout' with dice; 3) a film strip labelled '160-frame chunks'; 4) a database cylinder labelled 'ArrayRecord shards, pickle records'; 5) a large number card reading '9,600,000 frames'. Under stage 4 a small callout box lists the record schema: raw_video, sequence_length, actions, rewards." fig3_data_engine.png

gen "A two-phase training timeline drawn as a long horizontal bar from step 0 to step 80,000. The first half, 0 to 40,000, is shaded lightly and labelled 'Phase 1: flow matching only - learn the velocity field'. The second half, 40,000 to 80,000, is shaded darker and labelled 'Phase 2: + shortcut bootstrap - learn to take big steps'. A vertical marker at 40,000 reads 'bootstrap_start'. Below the bar, two small line charts share the axis: one labelled flow_mse falling from 0.53 to 0.014, one labelled boot_mse appearing only after 40,000 and falling to 0.004." fig4_two_phase.png

gen "An explanatory diagram of shortcut models. TOP row: a curved path from a noise cloud to a clean game frame, traversed by MANY small numbered arrows, labelled 'standard flow matching: 256 small steps, slow'. BOTTOM row: the same curved path traversed by only 3 or 4 LARGE arrows, labelled 'shortcut model: few big steps, real-time'. To the right, a small box showing the self-consistency rule: 'one step of size 2d must equal two steps of size d'." fig5_shortcut.png

gen "A diagram about compounding error in autoregressive rollouts. A horizontal chain of small game frames, each feeding into the next through a box labelled 'dynamics'. Above the first four frames a bracket labelled 'real context frames (4)'. Above the remaining frames a longer bracket labelled 'model feeds on its own predictions (144)'. A subtle widening wedge above the chain labelled 'error accumulates - this is where world models break'." fig6_rollout_drift.png

gen "A clean measurement decomposition diagram, three horizontal bars of different lengths sharing a left axis. Bar one labelled 'tokenizer ceiling - original vs reconstruction' value 16.59. Bar two labelled 'dynamics only - reconstruction vs dream' value 23.29. Bar three, the longest, labelled 'end to end - original vs dream' value 32.19. A caption underneath reads 'lower FVD is better'. Small icons of a film frame beside each bar." fig7_fvd.png
