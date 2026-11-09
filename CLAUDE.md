# Commits and pull requests

This repo is public. Commits, PR descriptions and PR comments describe the change only:

- No AI attribution: no `Co-Authored-By:` for an AI tool, no agent session or project links (they open private
  conversations), no "Generated with ..." footers, no "Requested by / project thread" block.
- Commit as Joey Lamping <35737718+jaylamping@users.noreply.github.com>, never as an AI tool identity.

`.claude/settings.json` turns Claude Code's attribution off and sets the identity at session start.
`scripts/check_attribution.py` enforces it twice: as a `commit-msg` hook (`python3 scripts/check_attribution.py install`,
once per clone) and in CI (`.github/workflows/attribution.yml`), which fails a push or PR whose commits or description
carry attribution.
