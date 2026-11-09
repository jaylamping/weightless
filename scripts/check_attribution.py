#!/usr/bin/env python3
"""No AI attribution in commits or PR descriptions. The repo is public: a session link opens a chat.

  range <a>..<b>       every commit in the range: message, author and committer
  text                 stdin, such as a PR description
  commit-msg <file>    a commit message file, plus the identity git will commit as (the commit-msg hook)
  install              install the commit-msg hook into this clone

Exits non-zero and names each offending line if anything matches.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Each pattern is one kind of attribution an AI tool adds by default. Matched case-insensitively, per line.
PATTERNS = [
    (re.compile(r"^\s*co-authored-by:.*(claude|anthropic|openai|copilot|cursor|codex|gemini)", re.I), "AI co-author trailer"),
    (re.compile(r"^\s*claude-session:", re.I), "session trailer"),
    (re.compile(r"claude\.ai/", re.I), "claude.ai link"),
    (re.compile(r"generated (with|by) \[?claude", re.I), "\"Generated with\" footer"),
    (re.compile(r"claude-projects-attribution", re.I), "attribution marker"),
    (re.compile(r"\U0001F916"), "robot emoji footer"),
]

# Agent tools commit as these identities unless told otherwise.
BAD_IDENTITY = re.compile(r"noreply@anthropic\.com|@users\.noreply\.anthropic|noreply@openai\.com", re.I)

HOOK = """#!/bin/sh
# Installed by scripts/check_attribution.py install.
exec python3 "$(git rev-parse --show-toplevel)/scripts/check_attribution.py" commit-msg "$1"
"""


def text_problems(text: str, where: str) -> list[str]:
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        for pat, what in PATTERNS:
            if pat.search(line):
                out.append(f"{where} line {n}: {what}: {line.strip()}")
    return out


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def check_range(rng: str) -> list[str]:
    out = []
    for sha in git("rev-list", rng).split():
        short = sha[:10]
        ident = git("log", "-1", "--format=%an <%ae>%n%cn <%ce>", sha)
        if BAD_IDENTITY.search(ident):
            out.append(f"commit {short}: author or committer is an AI tool identity: {' / '.join(ident.split(chr(10))[:2])}")
        out += text_problems(git("log", "-1", "--format=%B", sha), f"commit {short}")
    return out


def check_commit_msg(path: str) -> list[str]:
    out = text_problems(Path(path).read_text(encoding="utf-8", errors="replace"), "commit message")
    for var in ("GIT_AUTHOR_IDENT", "GIT_COMMITTER_IDENT"):
        ident = git("var", var)
        if BAD_IDENTITY.search(ident):
            out.append(f"{var} is an AI tool identity: {ident.strip()}. Set git config user.name and user.email.")
    return out


def install() -> None:
    hook = Path(git("rev-parse", "--git-path", "hooks").strip())
    if not hook.is_absolute():
        hook = ROOT / hook
    hook.mkdir(parents=True, exist_ok=True)
    target = hook / "commit-msg"
    if target.exists() and target.read_text() != HOOK:
        sys.exit(f"{target} exists and is not ours; merge it by hand")
    target.write_text(HOOK)
    target.chmod(0o755)
    print(f"installed {target}")


def main() -> None:
    args = sys.argv[1:]
    if args == ["install"]:
        install()
        return
    if len(args) == 2 and args[0] == "range":
        problems = check_range(args[1])
    elif args == ["text"]:
        problems = text_problems(sys.stdin.read(), "text")
    elif len(args) == 2 and args[0] == "commit-msg":
        problems = check_commit_msg(args[1])
    else:
        sys.exit(__doc__)
    for p in problems:
        print(p, file=sys.stderr)
    if problems:
        sys.exit("AI attribution found; remove it")


if __name__ == "__main__":
    main()
