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
