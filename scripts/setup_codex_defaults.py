#!/usr/bin/env python3
"""Install the user's requested full-access Codex defaults; no Codex launch.

Run once from a normal terminal. Existing files receive timestamped backups.
"""
import os
from pathlib import Path
import re
import shutil
import time
import tomllib


SHELL_BLOCK = '''# BEGIN personal Codex launch defaults
unalias codex 2>/dev/null || true
alias codex='codex --no-daemon --dangerously-bypass-approvals-and-sandbox'
# END personal Codex launch defaults
'''
ATTRIBUTION_BLOCK = '''<!-- BEGIN personal Git attribution preference -->
Use my configured Git identity for both author and committer. Never add
Codex, OpenAI, or any assistant as an author/co-author, and never add
AI-attribution trailers or generated-by text to commits.
<!-- END personal Git attribution preference -->
'''


def config_text(original):
    parsed = tomllib.loads(original)
    if isinstance(parsed.get('approval_policy'), dict):
        raise ValueError('Existing structured approval_policy needs manual migration')
    # Only change root scalars, preserving project/model/TUI/plugin settings.
    table = re.search(r'^\s*\[', original, re.MULTILINE)
    end = table.start() if table else len(original)
    root, tables = original[:end], original[end:]
    root = re.sub(
        r'^[ \t]*(sandbox_mode|approval_policy|default_permissions)[ \t]*=.*(?:\n|$)',
        '', root, flags=re.MULTILINE)
    result = ('sandbox_mode = "danger-full-access"\n'
              'approval_policy = "never"\n' + root + tables)
    checked = tomllib.loads(result)
    assert checked['sandbox_mode'] == 'danger-full-access'
    assert checked['approval_policy'] == 'never'
    return result


def replace_block(original, block, start, end):
    pattern = re.escape(start) + r'.*?' + re.escape(end) + r'\n?'
    if start in original:
        if end not in original:
            raise ValueError('Incomplete managed block; refusing to overwrite')
        return re.sub(pattern, lambda _: block, original, flags=re.DOTALL)
    return original + ('\n' if original and not original.endswith('\n') else '') + '\n' + block


def install(config_dir, shell_rc):
    config = config_dir / 'config.toml'
    override = config_dir / 'AGENTS.override.md'
    agents = override if override.exists() and override.read_text().strip() else config_dir / 'AGENTS.md'
    read = lambda p: p.read_text() if p.exists() else ''
    # Construct and validate every update before writing any file.
    changes = [
        (config, config_text(read(config))),
        (shell_rc, replace_block(read(shell_rc), SHELL_BLOCK,
                                '# BEGIN personal Codex launch defaults',
                                '# END personal Codex launch defaults')),
        (agents, replace_block(read(agents), ATTRIBUTION_BLOCK,
                              '<!-- BEGIN personal Git attribution preference -->',
                              '<!-- END personal Git attribution preference -->')),
    ]
    stamp = time.strftime('%Y%m%d-%H%M%S') + '-' + str(time.time_ns())
    changed = []
    for path, content in changes:
        if read(path) == content:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            backup = path.with_name(path.name + '.before-codex-defaults-' + stamp)
            shutil.copy2(path, backup)
            print(f'Backup: {backup}')
        path.write_text(content)
        changed.append(path)
        print(f'Updated: {path}')
    return changed


if __name__ == '__main__':
    config_dir = Path(os.environ.get('CODEX_HOME') or Path.home() / '.codex')
    shell_rc = Path(os.environ.get('ZDOTDIR') or Path.home()) / '.zshrc'
    install(config_dir, shell_rc)
    print('Installed. Open a new terminal, then type codex.')
    print('Defaults: full access, no approval prompts, no shared daemon; user-only Git attribution.')
