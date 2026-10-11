# Physical hardware authorization

Never activate, enable, or move the physical robot without the user's
express permission for that specific session/action. Authorization to train,
evaluate in simulation, or run the web UI is not hardware authorization.
Prior hardware authorization does not carry over to a new session.

Run commands from the repository root. Keep experimental training limits
scoped to the run; do not change physical deployment defaults to launch an
experiment.

# Commits and pushes

Commit and push completed, validated milestones periodically during work;
do not leave days of work uncommitted. The user authorizes routine commits
and pushes to this repository's existing remote without asking each time.
Review the diff and stage coherent changes; do not include unrelated edits,
secrets, recordings, build outputs, or model weights by accident. Do not
force-push or rewrite published history without explicit authorization.

Use the user's configured Git identity for both author and committer. Never
add Codex, OpenAI, or another assistant as an author/co-author, or add
AI-attribution trailers or generated-by text to commits. This preference
applies to all future sessions in this repository. Report failed commits or
pushes explicitly; never imply that local edits have been pushed.

# Service ownership

When you start a web server or background service, handle its required
restarts and verify the running version when code changes. Do not assume
that editing source updates an already-running process. If the current
environment cannot reach the process, explain that concrete limitation.
Service-management authorization never authorizes enabling or moving the
physical robot.

# Long-running experiment ownership

When the user gives hours or overnight time for an experiment, own the full
research cycle: run it, monitor it, inspect completed results and replays,
diagnose regressions, and perform evidence-driven follow-up within the
user's scope and time/resource budget. Do not end the task merely because
a background process started successfully or a fixed sweep completed.
An unattended script is useful execution machinery, not a substitute for
reviewing its results and choosing the next experiment.

Keep a durable experiment record with the objective, hypotheses, exact run
names, results, current interpretation, next actions, and stopping criteria.
Use that record to continue across context compaction or session handoffs.
Keep live status and periodic evaluation replays discoverable in the web UI.
If work must stop before review/iteration is complete, state what remains
and the concrete reason; do not imply follow-up is happening automatically.
Research authorization still never authorizes physical robot activation or
changing deployment defaults.
