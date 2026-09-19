---
name: user_preferences
description: User's working style, preferences, and technical background for the air hockey project
type: user
---

- Senior engineer comfortable with RL concepts, asks good questions about algorithm internals
- Often works remotely from phone via Tailscale — web UI must work well on mobile
- Prefers to see training replays visually rather than just metrics
- Wants to understand why things work, not just have them work (asked about SAC internals, on-policy vs off-policy, network sizing)
- Impatient with slow training — always looking to optimize throughput
- Training machine has an RTX 4090 (24 GB); `nvidia-smi` to check it is free before launching a run
- Likes to iterate quickly: make a change, run training, watch replays, adjust
- Prefers me to commit periodically without asking
- Hardware: a built 4-motor CDPR on Teknic ClearPath-SC servos (two models, see CLAUDE.md), Teensy step/dir firmware, FLIR camera
- Never initiate robot motion; the user runs `ai/bin/play.sh`. Building, flashing, restarting the UI are fine.
- Sim-to-real alignment is the thing he cares most about: table runs reproduce training by default, and every deviation (planner iterations, horizon, caps, timeouts, opponent, referee) is stated up front, not discovered later
- Flags or positional arguments, never environment-variable settings (`POLICY=... bash play.sh` was rejected)
- Versioned run names `<major>.<minor>-<description>-<stage>`, registry in `ai/RUNS.md`
- Wants simple, direct explanations; judges checkpoints by watching replays and the table, not by counters -- "it stops the puck but never shoots it" overrode a page of numbers
- Commands runnable from the repo root, no `cd`
